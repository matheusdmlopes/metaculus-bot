"""
Testes sem rede e sem gasto: um cliente falso da Anthropic devolve respostas
prontas, e o bot roda o fluxo inteiro (pesquisa, cinco previsões, agregação,
comentário) sem publicar nada.

    uv run pytest
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    DateQuestion,
    DiscreteQuestion,
    ForecastReport,
    MultipleChoiceQuestion,
    NumericQuestion,
)

import main  # noqa: E402
import prompts  # noqa: E402
from claude_llm import Claude, ClaudeRefused  # noqa: E402

BRIEFING = """## Summary
Things stand roughly where they stood last month.

## Status against the resolution criteria
Not met yet as of today.

## Sources
- Example source, 2026-11-10, https://example.com/a"""


def block(kind: str, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(type=kind, text=text)


def message(
    content: list[SimpleNamespace],
    stop_reason: str = "end_turn",
    searches: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber", explanation="declined"),
        model="claude-opus-5-5",
        usage=SimpleNamespace(
            input_tokens=10_000,
            output_tokens=2_000,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            server_tool_use=SimpleNamespace(web_search_requests=searches),
        ),
    )


def research_message() -> SimpleNamespace:
    half = len(BRIEFING) // 2
    return message(
        [
            block("text", "I'll look this up."),
            block("server_tool_use"),
            block("web_search_tool_result"),
            block("thinking"),
            # A busca quebra o texto final em vários blocos por causa das citações.
            block("text", BRIEFING[:half]),
            block("text", BRIEFING[half:]),
        ],
        searches=3,
    )


class FakeAnthropic:
    """Imita só o que o claude_llm usa: client.beta.messages.stream(...)."""

    def __init__(self, answer: Callable[[dict[str, Any], int], SimpleNamespace]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._answer = answer
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **request: Any) -> Any:
        self.requests.append(request)
        reply = self._answer(request, len(self.requests))

        class Stream:
            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *error: Any) -> bool:
                return False

            async def get_final_message(self) -> SimpleNamespace:
                return reply

        return Stream()


def scripted(forecasts: list[dict[str, Any]]) -> Callable[[dict[str, Any], int], SimpleNamespace]:
    """Pesquisa devolve o briefing; cada previsão devolve o próximo item da lista."""
    remaining = list(forecasts)

    def answer(request: dict[str, Any], _count: int) -> SimpleNamespace:
        if "tools" in request:
            return research_message()
        return message([block("thinking"), block("text", json.dumps(remaining.pop(0)))])

    return answer


def run_bot(
    questions: list[Any],
    answer: Callable[[dict[str, Any], int], SimpleNamespace],
    max_usd: float = 100.0,
) -> tuple[list[Any], Claude, FakeAnthropic]:
    async def go() -> tuple[list[Any], Claude, FakeAnthropic]:
        claude = Claude(max_usd_per_run=max_usd, parallel_calls=4)
        fake = FakeAnthropic(answer)
        claude._client = fake  # type: ignore[assignment]
        claude._slots = asyncio.Semaphore(4)
        bot = main.ClaudeForecastBot(
            claude=claude,
            settings=main.Settings(),
            publish_reports_to_metaculus=False,
            extra_metadata_in_explanation=True,
        )
        results = await bot.forecast_questions(questions, return_exceptions=True)
        return list(results), claude, fake

    return asyncio.run(go())


def only_report(results: list[Any]) -> ForecastReport:
    assert len(results) == 1
    assert isinstance(results[0], ForecastReport), results[0]
    return results[0]


def test_binary_takes_the_median_of_five_angles() -> None:
    forecasts = [
        {"rationale": f"Reason {i}.", "probability_percent": percent}
        for i, percent in enumerate([10, 30, 35, 60, 99.9])
    ]
    question = BinaryQuestion(question_text="Will it rain in Lucca tomorrow?")
    results, claude, fake = run_bot([question], scripted(forecasts))
    report = only_report(results)

    assert report.prediction == pytest.approx(0.35)
    for lens_name, _ in prompts.LENSES:
        assert f"Angle: {lens_name}" in report.explanation
    assert "Things stand roughly where they stood" in report.explanation
    assert "I'll look this up" not in report.explanation
    # 1 pesquisa + 5 previsões, todas com fallback de recusa ligado.
    assert len(fake.requests) == 6
    assert all(request["fallbacks"] == "default" for request in fake.requests)
    assert claude.web_searches == 3
    assert report.price_estimate == pytest.approx(claude.spent_usd)
    assert claude.spent_usd == pytest.approx(6 * 0.08 + 3 * 0.01)


def test_binary_is_clamped_to_1_99() -> None:
    forecasts = [{"rationale": "Sure thing.", "probability_percent": 100}] * 5
    question = BinaryQuestion(question_text="Will the sun rise tomorrow?")
    report = only_report(run_bot([question], scripted(forecasts))[0])
    assert report.prediction == pytest.approx(0.99)


def test_multiple_choice_keeps_a_floor_on_every_option() -> None:
    options = ["Red", "Green", "Blue"]
    forecasts = [
        {
            "rationale": "Red is far ahead.",
            "options": [
                {"option": "Red", "probability_percent": 100},
                {"option": "Green", "probability_percent": 0},
                {"option": "Blue", "probability_percent": 0},
            ],
        }
    ] * 5
    question = MultipleChoiceQuestion(question_text="Which colour wins?", options=options)
    report = only_report(run_bot([question], scripted(forecasts))[0])

    probabilities = report.prediction.to_dict()
    assert list(probabilities) == options
    assert sum(probabilities.values()) == pytest.approx(1)
    assert probabilities["Green"] >= 0.009
    assert probabilities["Red"] < 0.99


def test_multiple_choice_missing_option_fails_that_forecast() -> None:
    incomplete = {
        "rationale": "Forgot one.",
        "options": [{"option": "Red", "probability_percent": 100}],
    }
    question = MultipleChoiceQuestion(
        question_text="Which colour wins?", options=["Red", "Green"]
    )
    results, _, _ = run_bot([question], scripted([incomplete] * 5))
    assert isinstance(results[0], BaseException)


def numeric_answer(values: list[float]) -> dict[str, Any]:
    answer: dict[str, Any] = {"rationale": "A spread of outcomes."}
    for percentile, value in zip(prompts.PERCENTILES, values):
        answer[prompts.percentile_key(percentile)] = value
    return answer


def test_numeric_builds_a_submittable_distribution() -> None:
    values = [12, 15, 20, 24, 28, 31, 35, 40, 48, 60, 75]
    out_of_order = [15, 12, *values[2:]]
    beyond_closed_bound = [-5, *values[1:]]
    question = NumericQuestion(
        question_text="How many launches in 2027?",
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=True,
        unit_of_measure="launches",
    )
    forecasts = [numeric_answer(v) for v in (values, values, out_of_order, beyond_closed_bound, values)]
    report = only_report(run_bot([question], scripted(forecasts))[0])

    cdf = report.prediction.get_cdf()
    assert len(cdf) == 201
    assert cdf[0].percentile == pytest.approx(0)  # limite de baixo fechado
    assert cdf[-1].percentile < 1  # limite de cima aberto guarda massa acima
    assert "Percentile 50: 31" in report.explanation


def test_discrete_question_uses_the_numeric_path() -> None:
    question = DiscreteQuestion(
        question_text="How many rate cuts in 2027?",
        lower_bound=-0.5,
        upper_bound=10.5,
        open_lower_bound=False,
        open_upper_bound=False,
        cdf_size=12,
    )
    forecasts = [numeric_answer([0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5])] * 5
    report = only_report(run_bot([question], scripted(forecasts))[0])
    assert len(report.prediction.get_cdf()) == 12


def test_date_question() -> None:
    dates = [f"2027-{month:02d}-15" for month in range(1, 12)]
    answer: dict[str, Any] = {"rationale": "Sometime in 2027."}
    for percentile, date in zip(prompts.PERCENTILES, dates):
        answer[prompts.percentile_key(percentile)] = date
    question = DateQuestion(
        question_text="When will the bridge open?",
        lower_bound=datetime(2026, 12, 1, tzinfo=timezone.utc),
        upper_bound=datetime(2028, 1, 1, tzinfo=timezone.utc),
        open_lower_bound=False,
        open_upper_bound=True,
    )
    report = only_report(run_bot([question], scripted([answer] * 5))[0])
    assert len(report.prediction.get_cdf()) == 201
    assert "Percentile 50: 2027-06-15" in report.explanation


def test_paused_research_turn_is_resumed() -> None:
    forecasts = [{"rationale": "Fine.", "probability_percent": 40}] * 5

    def answer(request: dict[str, Any], count: int) -> SimpleNamespace:
        if "tools" not in request:
            return message([block("text", json.dumps(forecasts[0]))])
        if count == 1:
            return message([block("server_tool_use")], stop_reason="pause_turn", searches=2)
        return research_message()

    question = BinaryQuestion(question_text="Will the pause be resumed?")
    results, claude, fake = run_bot([question], answer)
    only_report(results)

    resumed = fake.requests[1]["messages"]
    assert [turn["role"] for turn in resumed] == ["user", "assistant"]
    assert claude.web_searches == 5


def test_refusal_becomes_an_error() -> None:
    async def go() -> None:
        claude = Claude(max_usd_per_run=100, parallel_calls=1)
        claude._client = FakeAnthropic(  # type: ignore[assignment]
            lambda request, count: message([], stop_reason="refusal")
        )
        claude._slots = asyncio.Semaphore(1)
        with pytest.raises(ClaudeRefused):
            await claude.ask("q", system="s", model="claude-opus-5-5", effort="low")
        assert claude.spent_usd > 0  # o que foi cobrado até a recusa entra na conta

    asyncio.run(go())


def test_budget_postpones_the_remaining_questions() -> None:
    forecasts = [{"rationale": "Fine.", "probability_percent": 40}] * 10
    questions = [
        BinaryQuestion(question_text=f"Question {i}?", id_of_question=i) for i in (1, 2, 3)
    ]

    async def go() -> list[Any]:
        claude = Claude(max_usd_per_run=0.30, parallel_calls=4)
        claude._client = FakeAnthropic(scripted(forecasts))  # type: ignore[assignment]
        claude._slots = asyncio.Semaphore(4)
        settings = main.Settings(parallel_questions=1)
        bot = main.ClaudeForecastBot(
            claude=claude, settings=settings, publish_reports_to_metaculus=False
        )
        results = list(await bot.forecast_questions(questions, return_exceptions=True))
        assert main.print_summary(results, claude, show_content=False) == 0
        return results

    results = asyncio.run(go())
    assert isinstance(results[0], ForecastReport)
    assert all("BudgetExceeded" in str(result) for result in results[1:])


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def day_capped_bot(ledger: Path, forecasts: list[dict[str, Any]]) -> tuple[Any, Claude, FakeAnthropic]:
    claude = Claude(
        max_usd_per_run=100, parallel_calls=4, max_usd_per_day=5.0, day_ledger=ledger
    )
    fake = FakeAnthropic(scripted(forecasts))
    claude._client = fake  # type: ignore[assignment]
    claude._slots = asyncio.Semaphore(4)
    bot = main.ClaudeForecastBot(
        claude=claude, settings=main.Settings(), publish_reports_to_metaculus=False
    )
    return bot, claude, fake


def test_day_cap_only_starts_a_question_that_fits(tmp_path: Path) -> None:
    ledger = tmp_path / ".gasto" / "gasto-do-dia.json"
    ledger.parent.mkdir()
    ledger.write_text(json.dumps({"dia": today_utc(), "usd": 3.9}))
    forecasts = [{"rationale": "Fine.", "probability_percent": 40}] * 15
    questions = [
        BinaryQuestion(question_text=f"Question {i}?", id_of_question=i) for i in (1, 2, 3)
    ]

    async def go() -> tuple[list[Any], Claude]:
        bot, claude, _fake = day_capped_bot(ledger, forecasts)
        results = list(await bot.forecast_questions(questions, return_exceptions=True))
        return results, claude

    results, claude = asyncio.run(go())
    question_cost = 6 * 0.08 + 3 * 0.01
    # 1ª: 3,90 + 1,00 de reserva cabe em 5. 2ª: 4,41 + 0,51 cabe. 3ª: 4,92 + 0,51 não cabe.
    assert isinstance(results[0], ForecastReport)
    assert isinstance(results[1], ForecastReport)
    assert "por dia" in str(results[2])
    assert claude.spent_today_usd == pytest.approx(3.9 + 2 * question_cost)
    assert claude.spent_today_usd <= 5.0
    saved = json.loads(ledger.read_text())
    assert saved["dia"] == today_utc()
    assert saved["usd"] == pytest.approx(claude.spent_today_usd, abs=1e-4)
    assert saved["maior_pergunta"] == pytest.approx(question_cost, abs=1e-4)


@pytest.mark.parametrize(
    "record",
    [
        {"usd": 4.5},  # sem pergunta hoje ainda: reserva US$ 1 e não cabe
        {"usd": 2.0, "maior_pergunta": 3.5},  # a pergunta mais cara do dia não caberia
    ],
)
def test_day_cap_blocks_before_any_call(tmp_path: Path, record: dict[str, float]) -> None:
    ledger = tmp_path / "gasto-do-dia.json"
    ledger.write_text(json.dumps({"dia": today_utc(), **record}))

    async def go() -> tuple[list[Any], FakeAnthropic]:
        bot, claude, fake = day_capped_bot(ledger, [])
        question = BinaryQuestion(question_text="Question?", id_of_question=1)
        results = list(await bot.forecast_questions([question], return_exceptions=True))
        assert main.print_summary(results, claude, show_content=False) == 0
        return results, fake

    results, fake = asyncio.run(go())
    assert "BudgetExceeded" in str(results[0])
    assert fake.requests == []


@pytest.mark.parametrize(
    "content",
    ['{"dia": "2020-01-01", "usd": 999, "maior_pergunta": 9}', "{}", "not json", None],
)
def test_day_count_starts_at_zero_without_a_record_for_today(
    tmp_path: Path, content: str | None
) -> None:
    ledger = tmp_path / "gasto-do-dia.json"
    if content is not None:
        ledger.write_text(content)
    claude = Claude(
        max_usd_per_run=100, parallel_calls=1, max_usd_per_day=5.0, day_ledger=ledger
    )
    assert claude.spent_today_usd == 0.0
    claude.check_budget()
    assert json.loads(ledger.read_text()) == {
        "dia": today_utc(),
        "usd": 0.0,
        "maior_pergunta": 0.0,
    }


def test_defaults_stay_inside_the_five_dollar_day() -> None:
    settings = main.Settings()
    assert settings.max_usd_per_day <= 5.0
    assert settings.max_usd_per_run <= settings.max_usd_per_day


def test_live_tournament_question_is_refused_for_dry_runs() -> None:
    live = BinaryQuestion(question_text="Live?", tournament_slugs=["minibench"])
    other = BinaryQuestion(question_text="Other?", tournament_slugs=["bot-testing-area"])
    assert main.is_live_tournament_question(live)
    assert not main.is_live_tournament_question(other)
