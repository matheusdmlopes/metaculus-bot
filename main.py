"""
Bot de previsão para o torneio de bots da Metaculus (FutureEval), movido a Claude.

Para cada pergunta aberta:
  1. uma pesquisa na web (Claude com busca e leitura de página) vira um briefing;
  2. cinco previsões independentes leem o briefing, cada uma por um ângulo;
  3. a biblioteca forecasting-tools agrega as cinco (mediana), publica a
     previsão e posta o comentário com a pesquisa e os raciocínios.

Uso:
  uv run python main.py --checar                 confere as chaves
  uv run python main.py --texto "Will X ...?"    prevê uma pergunta solta, sem publicar
  uv run python main.py --pergunta <URL>         prevê uma pergunta da Metaculus, sem publicar
  uv run python main.py --mode test_questions    roda na área de teste (não pontua)
  uv run python main.py                          roda no torneio e no MiniBench (publica)

Regra do torneio: nada de humano no meio. Não se roda o bot numa pergunta
aberta do torneio para olhar a resposta e ajustar o bot em cima dela; por isso
`--pergunta` recusa perguntas do torneio em andamento.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import anthropic
import dotenv

from bot_helpers import has_env, silence_noisy_dependencies

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DateQuestion,
    ForecastBot,
    ForecastReport,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    PredictionAffirmed,
    PredictionTypes,
    ReasonedPrediction,
    clean_indents,
)

import prompts  # noqa: E402
from claude_llm import BudgetExceeded, Claude, ClaudeError  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

TEST_TOURNAMENT = "bot-testing-area"
REPORTS_FOLDER = Path(__file__).parent / "relatorios"


@dataclass(frozen=True)
class Settings:
    """Ajustes do bot. Cada um pode ser trocado por variável de ambiente."""

    research_model: str = "claude-opus-5-5"
    forecast_model: str = "claude-opus-5-5"
    research_effort: str = "high"
    forecast_effort: str = "high"
    max_searches: int = 8
    max_fetches: int = 5
    forecasts_per_question: int = 5
    max_usd_per_run: float = 25.0
    parallel_calls: int = 4
    parallel_questions: int = 2

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        values: dict[str, Any] = {}
        for name, default in vars(defaults).items():
            raw = os.getenv(f"BOT_{name.upper()}")
            values[name] = type(default)(raw) if raw else default
        return cls(**values)


class ClaudeForecastBot(ForecastBot):
    def __init__(
        self,
        *,
        claude: Claude,
        settings: Settings,
        question_limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.claude = claude
        self.settings = settings
        self.question_limit = question_limit
        self._lens_counter: dict[int, int] = {}
        self._question_slots: asyncio.Semaphore | None = None
        super().__init__(
            research_reports_per_question=1,
            predictions_per_research_report=settings.forecasts_per_question,
            llms={
                "default": settings.forecast_model,
                "researcher": settings.research_model,
            },
            **kwargs,
        )

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, Any]:
        # Só para o nome dos modelos sair certo no comentário publicado; as
        # chamadas de verdade passam pelo claude_llm, não pelo litellm.
        defaults = Settings()
        return {
            "default": defaults.forecast_model,
            "researcher": defaults.research_model,
        }

    ##################################### FLUXO #####################################

    async def forecast_questions(  # type: ignore[override]
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        if self.skip_previously_forecasted_questions:
            questions = [q for q in questions if not q.already_forecasted]
        # As perguntas do torneio ficam abertas só por algumas horas: primeiro
        # as que fecham antes.
        far_future = datetime.max.replace(tzinfo=timezone.utc)
        ordered = sorted(questions, key=lambda q: q.close_time or far_future)
        if self.question_limit:
            ordered = ordered[: self.question_limit]
        return await super().forecast_questions(ordered, return_exceptions)

    async def _run_individual_question(
        self, question: MetaculusQuestion
    ) -> ForecastReport:
        if self._question_slots is None:
            self._question_slots = asyncio.Semaphore(self.settings.parallel_questions)
        async with self._question_slots:
            self.claude.check_budget()
            return await super()._run_individual_question(question)

    ##################################### PESQUISA #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        if isinstance(question, ConditionalQuestion):
            description = (
                "This is a conditional pair. The forecaster will estimate the child "
                "question assuming the parent resolves Yes, and assuming it resolves No.\n\n"
                f"Parent question:\n{prompts.describe_question(question.parent)}\n\n"
                f"Child question:\n{prompts.describe_question(question.child)}"
            )
        else:
            description = prompts.describe_question(question)

        reply = await self.claude.ask(
            prompts.RESEARCH_TASK.format(today=prompts.today(), question=description),
            system=prompts.RESEARCH_SYSTEM,
            model=self.settings.research_model,
            effort=self.settings.research_effort,
            tools=prompts.research_tools(
                self.settings.max_searches, self.settings.max_fetches
            ),
            cache=True,
        )
        logger.info(
            "Pesquisa pronta: %s (%d buscas, US$ %.2f)",
            question.page_url,
            reply.web_searches,
            reply.cost_usd,
        )
        return reply.text

    async def summarize_research(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        # O briefing já abre com a seção "Summary": aproveita, sem outra chamada.
        match = re.search(r"##\s*Summary\s*\n(.*?)(?=\n##\s|\Z)", research, flags=re.S)
        return match.group(1).strip() if match else research[:1200]

    ##################################### PREVISÃO #####################################

    async def _ask_forecaster(
        self,
        question: MetaculusQuestion,
        research: str,
        answer_instructions: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        key = question.id_of_question or id(question)
        turn = self._lens_counter.get(key, 0)
        self._lens_counter[key] = turn + 1
        lens_name, lens_text = prompts.LENSES[turn % len(prompts.LENSES)]

        disclaimer = self._get_conditional_disclaimer_if_necessary(question)
        prompt = prompts.FORECAST_TASK.format(
            today=prompts.today(),
            question=prompts.describe_question(question),
            research=research,
            lens=f"{lens_text}\n\n{disclaimer}".strip(),
            answer_instructions=answer_instructions,
        )
        reply = await self.claude.ask(
            prompt,
            system=prompts.FORECAST_SYSTEM,
            model=self.settings.forecast_model,
            effort=self.settings.forecast_effort,
            schema=schema,
        )
        return reply.as_json(), lens_name

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        answer, lens = await self._ask_forecaster(
            question, research, prompts.BINARY_ANSWER, prompts.BINARY_SCHEMA
        )
        percent = float(answer["probability_percent"])
        if not 0 <= percent <= 100:
            raise ClaudeError(f"Probabilidade fora de 0 a 100: {percent}")
        probability = min(0.99, max(0.01, percent / 100))
        return ReasonedPrediction(
            prediction_value=probability,
            reasoning=_reasoning(lens, answer, f"Probability: {probability:.0%}"),
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        answer, lens = await self._ask_forecaster(
            question,
            research,
            prompts.MULTIPLE_CHOICE_ANSWER.format(options=question.options),
            prompts.multiple_choice_schema(question.options),
        )
        given = {
            item["option"]: float(item["probability_percent"])
            for item in answer["options"]
        }
        missing = [option for option in question.options if option not in given]
        total = sum(given.values())
        if missing or total <= 0 or min(given.values()) < 0:
            raise ClaudeError(
                f"Resposta de múltipla escolha inválida (faltando: {missing}, soma: {total})"
            )
        # Piso por opção: nenhuma fica em zero, porque zero numa opção que
        # acontece custa caro na pontuação.
        floor = min(0.01, 0.2 / len(question.options))
        floored = [max(given[option] / total, floor) for option in question.options]
        probabilities = [value / sum(floored) for value in floored]
        prediction = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=option, probability=probability)
                for option, probability in zip(question.options, probabilities)
            ]
        )
        final = "\n".join(
            f"- {option}: {probability:.1%}"
            for option, probability in zip(question.options, probabilities)
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=_reasoning(lens, answer, final)
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_message, lower_message = self._create_upper_and_lower_bound_messages(
            question
        )
        answer, lens = await self._ask_forecaster(
            question,
            research,
            prompts.NUMERIC_ANSWER.format(
                bounds=f"{lower_message}\n{upper_message}",
                units=question.unit_of_measure or "not stated, infer from the question",
            ),
            prompts.NUMERIC_SCHEMA,
        )
        values = [float(answer[prompts.percentile_key(p)]) for p in prompts.PERCENTILES]
        prediction = _distribution(values, question)
        final = "\n".join(
            f"- Percentile {p}: {value:g}" for p, value in zip(prompts.PERCENTILES, values)
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=_reasoning(lens, answer, final)
        )

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_message, lower_message = self._create_upper_and_lower_bound_messages(
            question
        )
        answer, lens = await self._ask_forecaster(
            question,
            research,
            prompts.DATE_ANSWER.format(bounds=f"{lower_message}\n{upper_message}"),
            prompts.DATE_SCHEMA,
        )
        dates = [answer[prompts.percentile_key(p)] for p in prompts.PERCENTILES]
        prediction = _distribution([_timestamp(date) for date in dates], question)
        final = "\n".join(
            f"- Percentile {p}: {date}" for p, date in zip(prompts.PERCENTILES, dates)
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=_reasoning(lens, answer, final)
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, DateQuestion):
            upper: Any = question.upper_bound.date().isoformat()
            lower: Any = question.lower_bound.date().isoformat()
            unit = ""
        else:
            upper = (
                question.nominal_upper_bound
                if question.nominal_upper_bound is not None
                else question.upper_bound
            )
            lower = (
                question.nominal_lower_bound
                if question.nominal_lower_bound is not None
                else question.lower_bound
            )
            unit = question.unit_of_measure or ""

        if question.open_upper_bound:
            upper_message = f"The question creator thinks the answer is likely not higher than {upper} {unit}."
        else:
            upper_message = f"The answer cannot be higher than {upper} {unit}."
        if question.open_lower_bound:
            lower_message = f"The question creator thinks the answer is likely not lower than {lower} {unit}."
        else:
            lower_message = f"The answer cannot be lower than {lower} {unit}."
        return upper_message.strip(), lower_message.strip()

    ############################# PERGUNTAS CONDICIONAIS #############################
    # Fluxo do template oficial, sem mudança: prevê pai, filho e os dois ramos
    # reaproveitando as funções de previsão acima.

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return (
            "This is a conditional question with a parent and a child. Forecast only "
            "the CHILD question, given how the parent resolved. Do not re-forecast the parent."
        )


def _reasoning(lens: str, answer: dict[str, Any], final_answer: str) -> str:
    return f"*Angle: {lens}*\n\n{str(answer['rationale']).strip()}\n\n{final_answer}"


def _timestamp(date_text: str) -> float:
    parsed = datetime.fromisoformat(date_text.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _distribution(
    values: list[float], question: NumericQuestion | DateQuestion
) -> NumericDistribution:
    """Monta a distribuição a partir dos valores nos percentis de prompts.PERCENTILES."""
    if isinstance(question, DateQuestion):
        lower, upper = question.lower_bound.timestamp(), question.upper_bound.timestamp()
    else:
        lower, upper = question.lower_bound, question.upper_bound

    if values != sorted(values):
        logger.warning("Percentis vieram fora de ordem; reordenando")
        values = sorted(values)

    # Limite fechado: nada passa dele. Limite aberto: pode passar, mas a
    # biblioteca rejeita valor absurdamente longe da faixa da pergunta.
    slack = 1.5 * (upper - lower)
    floor = lower - slack if question.open_lower_bound else lower
    ceiling = upper + slack if question.open_upper_bound else upper
    fitted = [min(ceiling, max(floor, value)) for value in values]

    return NumericDistribution.from_question(
        [
            Percentile(percentile=p / 100, value=value)
            for p, value in zip(prompts.PERCENTILES, fitted)
        ],
        question,
    )


############################### LINHA DE COMANDO ###############################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bot de previsão da Metaculus movido a Claude"
    )
    parser.add_argument(
        "--mode",
        choices=["tournament", "test_questions"],
        default="tournament",
        help="tournament: torneio de bots + MiniBench (padrão). test_questions: área de teste, não pontua",
    )
    parser.add_argument(
        "--pergunta",
        action="append",
        metavar="URL",
        help="prevê esta pergunta da Metaculus sem publicar (pode repetir)",
    )
    parser.add_argument(
        "--texto",
        metavar="PERGUNTA",
        help="prevê uma pergunta de sim ou não escrita na hora, sem publicar",
    )
    parser.add_argument(
        "--sem-publicar",
        action="store_true",
        help="na área de teste, só mostra a previsão em vez de enviar",
    )
    parser.add_argument(
        "--limite", type=int, metavar="N", help="no máximo N perguntas nesta execução"
    )
    parser.add_argument(
        "--checar", action="store_true", help="confere as chaves e sai"
    )
    args = parser.parse_args()
    if args.sem_publicar and args.mode != "test_questions":
        parser.error(
            "--sem-publicar só vale com --mode test_questions. Rodar o bot em pergunta "
            "aberta do torneio só para olhar a resposta fere a regra do torneio."
        )
    return args


def missing_env(*names: str) -> list[str]:
    return [name for name in names if not has_env(name)]


async def check_keys(settings: Settings, claude: Claude) -> int:
    """Confere as duas chaves e faz uma chamada mínima igual às do bot."""
    problems = 0

    if not has_env("ANTHROPIC_API_KEY"):
        print("Anthropic: falta ANTHROPIC_API_KEY no .env")
        problems += 1
    else:
        try:
            for model in sorted({settings.research_model, settings.forecast_model}):
                await claude.model_is_reachable(model)
            reply = await claude.ask(
                "Is water wet? This is a connectivity test.",
                system="Answer the test question.",
                model=settings.forecast_model,
                effort="low",
                schema=prompts.BINARY_SCHEMA,
                max_tokens=2000,
            )
            reply.as_json()
            print(f"Anthropic: chave válida, {settings.forecast_model} respondeu")
            search = await claude.ask(
                "Search the web once for today's top world news headline and report it in one sentence.",
                system="Answer briefly.",
                model=settings.research_model,
                effort="low",
                tools=prompts.research_tools(1, 1),
                max_tokens=4000,
            )
            if search.web_searches:
                print("Anthropic: busca na web funcionando")
            else:
                print(
                    "Anthropic: a chamada passou, mas a busca na web não rodou. "
                    "Confira se a busca está liberada na organização, no Console."
                )
                problems += 1
            print(f"Anthropic: custo deste teste: US$ {claude.spent_usd:.2f}")
        except anthropic.AuthenticationError:
            print("Anthropic: chave inválida (ANTHROPIC_API_KEY)")
            problems += 1
        except anthropic.NotFoundError as error:
            print(f"Anthropic: modelo não encontrado para esta chave: {error.message}")
            problems += 1
        except anthropic.APIConnectionError:
            print("Anthropic: sem conexão com a API")
            problems += 1
        except (anthropic.APIStatusError, ClaudeError) as error:
            print(f"Anthropic: a chamada de teste falhou: {error}")
            problems += 1

    if not has_env("METACULUS_TOKEN"):
        print("Metaculus: falta METACULUS_TOKEN no .env")
        problems += 1
    else:
        try:
            questions = await asyncio.to_thread(
                MetaculusClient().get_all_open_questions_from_tournament, TEST_TOURNAMENT
            )
            print(
                f"Metaculus: token válido, {len(questions)} pergunta(s) aberta(s) na área de teste"
            )
        except Exception as error:  # a biblioteca levanta tipos variados aqui
            print(f"Metaculus: não consegui ler a área de teste: {error}")
            problems += 1

    print("Tudo certo." if problems == 0 else f"{problems} problema(s) acima.")
    return 1 if problems else 0


def is_live_tournament_question(question: MetaculusQuestion) -> bool:
    live = {
        str(MetaculusClient.CURRENT_AI_COMPETITION_ID),
        str(MetaculusClient.CURRENT_MINIBENCH_ID),
        "fall-futureeval-2026",
    }
    ids = {str(slug) for slug in question.tournament_slugs}
    ids.add(str(question.default_project_id))
    return bool(live & ids)


def save_report(report: ForecastReport) -> Path:
    REPORTS_FOLDER.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", report.question.question_text.lower())[:60].strip("-")
    path = REPORTS_FOLDER / f"{datetime.now():%Y-%m-%d-%H%M%S}-{slug}.md"
    path.write_text(report.explanation, encoding="utf-8")
    return path


def print_summary(
    results: Sequence[ForecastReport | BaseException],
    claude: Claude,
    show_content: bool,
) -> int:
    """
    Resumo da execução. Devolve o código de saída: 1 se alguma pergunta falhou.

    No torneio (`show_content=False`) não sai previsão nem raciocínio, só o
    endereço e o estado de cada pergunta: o log do GitHub Actions é público em
    repositório público, e ninguém deve ajustar o bot olhando resposta de
    pergunta aberta.
    """
    reports = [r for r in results if isinstance(r, ForecastReport)]
    failures = [r for r in results if isinstance(r, BaseException)]
    postponed = [f for f in failures if "BudgetExceeded" in f"{type(f).__name__} {f}"]
    failures = [f for f in failures if f not in postponed]

    print("\n" + "=" * 80)
    if not results:
        print("Nenhuma pergunta nova nesta execução.")
    for report in reports:
        note = f" ({len(report.errors)} erro(s) menor(es))" if report.errors else ""
        if show_content:
            readable = type(report).make_readable_prediction(report.prediction)
            print(f"\n{report.question.question_text}")
            print(f"  Previsão: {readable.strip()}")
            print(
                f"  Custo: US$ {report.price_estimate or 0:.2f} | "
                f"Tempo: {report.minutes_taken or 0:.1f} min{note}"
            )
            print(f"  Relatório completo: {save_report(report)}")
        else:
            print(f"ok     {report.question.page_url}{note}")
    for failure in failures:
        message = str(failure)
        print(f"FALHOU {type(failure).__name__}: {message[:300]}")
    if postponed:
        print(f"{len(postponed)} pergunta(s) adiada(s): teto de gasto da execução atingido.")

    print(
        f"\nTotal: {len(reports)} prevista(s), {len(failures)} com falha | "
        f"{claude.calls} chamada(s) ao Claude, {claude.web_searches} busca(s), "
        f"US$ {claude.spent_usd:.2f}"
    )
    print("=" * 80)
    return 1 if failures else 0


async def run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    claude = Claude(
        max_usd_per_run=settings.max_usd_per_run, parallel_calls=settings.parallel_calls
    )
    if args.checar:
        return await check_keys(settings, claude)

    local_only = bool(args.texto or args.pergunta)
    publish = not (local_only or args.sem_publicar)
    in_tournament = args.mode == "tournament" and not local_only
    if in_tournament:
        # No torneio o log fica só com avisos e erros, sem conteúdo de previsão.
        logging.getLogger("forecasting_tools").setLevel(logging.WARNING)

    needed = ["ANTHROPIC_API_KEY"] if args.texto else ["ANTHROPIC_API_KEY", "METACULUS_TOKEN"]
    if missing := missing_env(*needed):
        print(f"Falta no .env: {', '.join(missing)}. Confira com: uv run python main.py --checar")
        return 1

    bot = ClaudeForecastBot(
        claude=claude,
        settings=settings,
        question_limit=args.limite,
        publish_reports_to_metaculus=publish,
        skip_previously_forecasted_questions=in_tournament,
        extra_metadata_in_explanation=True,
    )
    print(
        f"Modo: {'pergunta avulsa' if local_only else args.mode} | "
        f"publica: {'sim' if publish else 'não'} | "
        f"pesquisa: {settings.research_model} | previsão: {settings.forecast_model} x"
        f"{settings.forecasts_per_question} | teto: US$ {settings.max_usd_per_run:.2f}\n"
    )

    results: list[ForecastReport | BaseException]
    if args.texto:
        question = BinaryQuestion(question_text=args.texto)
        results = await bot.forecast_questions([question], return_exceptions=True)
    elif args.pergunta:
        questions = []
        for url in args.pergunta:
            question = await asyncio.to_thread(bot.metaculus_client.get_question_by_url, url)
            if is_live_tournament_question(question):
                print(
                    f"Recusado: {url} é do torneio em andamento. O bot prevê essa pergunta "
                    "sozinho, no modo torneio; olhar a resposta antes fere a regra."
                )
                return 1
            questions.append(question)
        results = await bot.forecast_questions(questions, return_exceptions=True)
    elif args.mode == "test_questions":
        results = await bot.forecast_on_tournament(TEST_TOURNAMENT, return_exceptions=True)
    else:
        results = []
        for tournament in (
            MetaculusClient.CURRENT_AI_COMPETITION_ID,
            MetaculusClient.CURRENT_MINIBENCH_ID,
        ):
            try:
                results += await bot.forecast_on_tournament(
                    tournament, return_exceptions=True
                )
            except Exception as error:  # um torneio fora do ar não derruba o outro
                logger.error("Não consegui ler o torneio %s: %s", tournament, error)
                results.append(error)

    return print_summary(results, claude, show_content=not in_tournament)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.exit(asyncio.run(run(parse_args())))
