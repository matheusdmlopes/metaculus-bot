"""
Camada do Claude: toda chamada de modelo do bot passa por aqui.

Usa o SDK oficial da Anthropic (não o litellm do forecasting-tools), sempre em
streaming, e cuida de quatro coisas que o resto do bot não precisa saber:

- recusa: a chamada pede `fallbacks="default"`, então numa recusa por política
  a própria API refaz o pedido no modelo substituto; se ainda assim recusar,
  vira `ClaudeRefused`;
- `pause_turn`: turno longo de busca na web pausa e é retomado aqui;
- custo: cada resposta é convertida em dólar pelo `usage` e somada no total da
  execução e no gerenciador de custo do forecasting-tools (é o número que
  aparece no comentário publicado);
- teto: `check_budget()` barra pergunta nova quando a execução passou do limite.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import anthropic
from forecasting_tools import MonetaryCostManager

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_PAUSE_RESUMES = 8
CALL_TIMEOUT_SECONDS = 20 * 60

# US$ por milhão de tokens: (entrada, saída, leitura de cache).
PRICES_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-fable-5-2": (8.00, 40.00, 0.20),
    # Leitura de cache do Haiku é aproximação (10% da entrada).
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
CACHE_WRITE_MULTIPLIER = 1.25
USD_PER_WEB_SEARCH = 0.01  # US$ 10 por 1.000 buscas


class ClaudeError(RuntimeError):
    """Erro da camada do Claude que o bot trata como falha daquela chamada."""


class ClaudeRefused(ClaudeError):
    """O modelo pedido e o substituto recusaram o pedido."""


class ClaudeTruncated(ClaudeError):
    """A resposta acabou antes do fim (max_tokens ou janela de contexto)."""


class BudgetExceeded(ClaudeError):
    """A execução passou do teto de gasto; perguntas novas ficam para a próxima."""


@dataclass
class Reply:
    text: str
    model: str
    cost_usd: float
    web_searches: int = 0

    def as_json(self) -> dict[str, Any]:
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as error:
            raise ClaudeError(
                f"Resposta estruturada não é JSON válido: {self.text[:200]!r}"
            ) from error


class Claude:
    def __init__(self, *, max_usd_per_run: float, parallel_calls: int) -> None:
        self.max_usd_per_run = max_usd_per_run
        self.spent_usd = 0.0
        self.calls = 0
        self.web_searches = 0
        self._parallel_calls = parallel_calls
        # Cliente e semáforo nascem dentro do event loop, na primeira chamada.
        self._client: anthropic.AsyncAnthropic | None = None
        self._slots: asyncio.Semaphore | None = None

    def check_budget(self) -> None:
        if self.spent_usd >= self.max_usd_per_run:
            raise BudgetExceeded(
                f"Teto de US$ {self.max_usd_per_run:.2f} por execução atingido "
                f"(gasto: US$ {self.spent_usd:.2f}). Pergunta fica para a próxima execução."
            )

    async def ask(
        self,
        prompt: str,
        *,
        system: str,
        model: str,
        effort: str,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 32000,
        cache: bool = False,
    ) -> Reply:
        """
        Uma pergunta ao Claude, uma resposta em texto.

        Com `schema`, o texto devolvido é JSON válido naquele formato. Saída
        estruturada não combina com busca na web (as citações da busca quebram
        o formato), por isso `tools` e `schema` não vão juntos.
        """
        if tools and schema:
            raise ValueError("tools e schema não podem ir na mesma chamada")
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(max_retries=4)
            self._slots = asyncio.Semaphore(self._parallel_calls)
        assert self._slots is not None

        output_config: dict[str, Any] = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "output_config": output_config,
        }
        if tools:
            request["tools"] = tools
        if cache:
            request["cache_control"] = {"type": "ephemeral"}
        if not model.startswith("claude-haiku"):  # Haiku não aceita fallback
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        total_cost = 0.0
        total_searches = 0

        async with self._slots:
            for _ in range(MAX_PAUSE_RESUMES + 1):
                async with asyncio.timeout(CALL_TIMEOUT_SECONDS):
                    async with self._client.beta.messages.stream(
                        messages=messages, **request
                    ) as stream:
                        message = await stream.get_final_message()

                cost, searches = _cost_of(model, message)
                total_cost += cost
                total_searches += searches
                self._register(cost, searches)

                if message.stop_reason == "pause_turn":
                    # Turno de ferramenta do servidor pausou: devolve o que veio
                    # como turno do assistente e a API continua de onde parou.
                    messages.append({"role": "assistant", "content": message.content})
                    continue
                break
            else:
                raise ClaudeTruncated(
                    f"Turno pausou mais de {MAX_PAUSE_RESUMES} vezes sem terminar"
                )

        if message.stop_reason == "refusal":
            details = message.stop_details
            raise ClaudeRefused(
                "Pedido recusado mesmo com fallback "
                f"(categoria: {getattr(details, 'category', None)}; "
                f"explicação: {getattr(details, 'explanation', None)})"
            )

        text = _final_text(message.content)
        if message.stop_reason in ("max_tokens", "model_context_window_exceeded"):
            if schema is not None or not text:
                raise ClaudeTruncated(f"Resposta cortada ({message.stop_reason})")
            logger.warning("Resposta cortada (%s); usando o texto parcial", message.stop_reason)
        if not text:
            raise ClaudeError(f"Resposta sem texto (stop_reason={message.stop_reason})")

        return Reply(
            text=text,
            model=message.model,
            cost_usd=total_cost,
            web_searches=total_searches,
        )

    async def model_is_reachable(self, model: str) -> None:
        """Confere a chave e o acesso ao modelo sem gastar token."""
        client = anthropic.AsyncAnthropic(max_retries=1)
        try:
            await client.models.retrieve(model)
        finally:
            await client.close()

    def _register(self, cost: float, searches: int) -> None:
        self.calls += 1
        self.spent_usd += cost
        self.web_searches += searches
        MonetaryCostManager.increase_current_usage_in_parent_managers(cost)


def _final_text(content: list[Any]) -> str:
    """
    Junta o último trecho contínuo de blocos de texto da resposta.

    Num turno com busca, o modelo intercala frases curtas ("vou procurar...")
    com chamadas de ferramenta; o relatório é o texto depois do último
    resultado. As citações quebram esse texto em vários blocos seguidos.
    """
    parts: list[str] = []
    for block in reversed(content):
        if block.type == "text":
            parts.append(block.text)
        elif parts and block.type != "thinking":
            break
    return "".join(reversed(parts)).strip()


def _cost_of(requested_model: str, message: Any) -> tuple[float, int]:
    usage = message.usage
    prices = PRICES_PER_MTOK.get(message.model) or PRICES_PER_MTOK.get(requested_model)
    if prices is None:
        logger.warning(
            "Sem preço para %s; custo estimado com o preço do Opus 5.5", message.model
        )
        prices = PRICES_PER_MTOK["claude-opus-5-5"]
    input_price, output_price, cache_read_price = prices

    tokens_cost = (
        usage.input_tokens * input_price
        + (usage.cache_creation_input_tokens or 0) * input_price * CACHE_WRITE_MULTIPLIER
        + (usage.cache_read_input_tokens or 0) * cache_read_price
        + usage.output_tokens * output_price
    ) / 1_000_000
    searches = usage.server_tool_use.web_search_requests if usage.server_tool_use else 0
    return tokens_cost + searches * USD_PER_WEB_SEARCH, searches
