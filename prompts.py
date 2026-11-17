"""
Textos que vão para o Claude e os formatos de resposta.

Os prompts ficam em inglês porque as perguntas e os comentários da Metaculus
são em inglês. O bot faz uma pesquisa por pergunta e depois cinco previsões
independentes, cada uma com uma "lente" (um ângulo de análise diferente); a
biblioteca agrega as cinco pela mediana.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from forecasting_tools import (
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

PERCENTILES = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


################################### PESQUISA ###################################

RESEARCH_SYSTEM = """\
You are the research analyst on a forecasting team that competes in a forecasting tournament. A forecaster will read your briefing and turn it into a probability, so your job is to find the evidence they need. You do not produce a forecast yourself.

The forecaster only sees your briefing, not the pages you read, so every fact that matters has to be in the briefing itself, with its date and source.

Your background knowledge may be out of date. For anything time-sensitive, rely on what you find on the web today, and say so plainly when you could not verify something."""

RESEARCH_TASK = """\
Today is {today} (UTC).

<question>
{question}
</question>

Research this question with web search and web fetch, then write the briefing. What the forecaster needs, in order of importance:

1. Where things stand today against the resolution criteria. If the criteria name a resolution source (a dataset, an official page, a specific outlet), go to that source and report its latest value and date.
2. Base rates: how often this kind of outcome has happened in comparable situations or time windows, with the numbers.
3. Developments from the last few weeks, each with its date, and events scheduled between now and the resolution deadline.
4. What prediction markets, bookmakers or analyst consensus currently say, if anything, with the date and the source. Report them as they are; the forecaster decides how much weight they deserve.
5. The main uncertainties, and anything in the resolution criteria or fine print that could make the question resolve in a way a casual reader would not expect.

Prefer primary sources and recent reporting. When sources disagree, show the disagreement instead of averaging it away. If the question has already been effectively decided by something that happened, say so and show the evidence.

Write the briefing in markdown with exactly these sections, and nothing before the first heading:

## Summary
Three to five sentences on where things stand today.

## Status against the resolution criteria

## Base rates

## Recent developments

## Markets and other forecasts

## Uncertainties and resolution traps

## Sources
One line per source: what it is, its date, and the URL.

Keep the briefing dense and factual, around 600 to 900 words."""


def research_tools(max_searches: int, max_fetches: int) -> list[dict[str, Any]]:
    return [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches},
        {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": max_fetches},
    ]


################################### PREVISÃO ###################################

FORECAST_SYSTEM = """\
You are a forecaster in a tournament scored with a proper scoring rule: your score depends on the probability you gave to what actually happened, compared with the other forecasters. The best strategy is to report your honest, calibrated belief. Overconfidence is expensive: a 99% that fails costs far more than a 90% that succeeds earns, so keep extreme probabilities for cases where the evidence is overwhelming.

How to work:
- Forecast what the resolution criteria and fine print say, read literally, not what would be true in spirit.
- Start from a base rate for this kind of question, then adjust for the specific evidence.
- Give the status quo its due: over a short window, most things do not change.
- Scale your estimate to the time left before the deadline.
- The research briefing was gathered from the web today and is your source for anything recent. Your own memory may be out of date; when it conflicts with the briefing about a recent fact, trust the briefing. If the briefing is thin or could not verify something important, widen your uncertainty instead of guessing.
- Market prices and other forecasts in the briefing are evidence, not an answer to copy. If you depart from them, say why.

Your rationale is posted publicly next to your forecast. Write it as short markdown, 150 to 250 words: the considerations that moved your number, most important first."""

# Cada previsão recebe uma lente. Cinco previsões por pergunta, uma de cada.
LENSES: list[tuple[str, str]] = [
    (
        "Base rates first",
        "For this pass, lead with the outside view. Identify the best reference class, state its base rate with numbers, and only then adjust for what is specific to this case. Keep the adjustment modest unless the specific evidence is strong.",
    ),
    (
        "Causal chain",
        "For this pass, lead with the inside view. Lay out what concretely has to happen for each outcome, who decides it, and on what timeline. Multiply through the steps that are genuinely uncertain rather than judging the story as a whole.",
    ),
    (
        "Status quo and time left",
        "For this pass, ask what happens if nothing changes, and how much time is left for something to change. Think in terms of a rate of change per week or month and scale it to the remaining window.",
    ),
    (
        "Devil's advocate",
        "For this pass, take the conclusion the briefing seems to point to and argue against it. Imagine the question resolved the other way and explain the most plausible reason why. Then give the number you actually believe after weighing that.",
    ),
    (
        "Resolution fine print",
        "For this pass, read the resolution criteria and fine print like a contract. Look for edge cases, thresholds, the exact source and the exact dates, and for ways the literal wording could resolve differently from the intuitive answer. Weigh any market prices or outside forecasts in the briefing.",
    ),
]

FORECAST_TASK = """\
Today is {today} (UTC).

<question>
{question}
</question>

<research_briefing>
{research}
</research_briefing>

{lens}

{answer_instructions}"""

BINARY_ANSWER = """\
Give the probability that this question resolves Yes, as a percentage between 1 and 99."""

MULTIPLE_CHOICE_ANSWER = """\
Give a probability, as a percentage, for every one of these options: {options}. The percentages must add up to 100. Leave a small probability on options that look unlikely, since surprises happen."""

NUMERIC_ANSWER = """\
Give your forecast as a probability distribution: the value at each of the percentiles 5, 10, 20, 30, 40, 50, 60, 70, 80, 90 and 95. The value at percentile 5 means you see a 5% chance the outcome is below it. Values must increase from one percentile to the next.

{bounds}
Answer in these units: {units}. Write plain numbers (for example 1500000, not 1.5 million and not scientific notation).

Good forecasters set wide intervals: the outcome should fall outside your 5 to 95 range about one time in ten."""

DATE_ANSWER = """\
Give your forecast as a probability distribution over dates: the date at each of the percentiles 5, 10, 20, 30, 40, 50, 60, 70, 80, 90 and 95. The date at percentile 5 means you see a 5% chance the outcome falls before it. Dates must be in chronological order, written as YYYY-MM-DD.

{bounds}

Good forecasters set wide intervals: the outcome should fall outside your 5 to 95 range about one time in ten."""


def describe_question(question: MetaculusQuestion) -> str:
    """A pergunta como o Claude a lê: título, tipo, critérios e prazos."""
    lines = [f"Title: {question.question_text}"]

    if isinstance(question, MultipleChoiceQuestion):
        lines.append(f"Type: multiple choice. Options: {question.options}")
    elif isinstance(question, DateQuestion):
        lines.append("Type: date (the answer is a date)")
    elif isinstance(question, NumericQuestion):
        units = question.unit_of_measure or "not stated, infer from the question"
        lines.append(f"Type: numeric (the answer is a number). Units: {units}")
    else:
        lines.append("Type: binary (Yes or No)")

    if question.group_question_option:
        lines.append(
            f'This is one item of a group of related questions. It is about: "{question.group_question_option}"'
        )
    if question.close_time:
        lines.append(f"Forecasting closes: {question.close_time:%Y-%m-%d}")
    if question.scheduled_resolution_time:
        lines.append(f"Scheduled resolution: {question.scheduled_resolution_time:%Y-%m-%d}")

    for label, value in (
        ("Background", question.background_info),
        ("Resolution criteria", question.resolution_criteria),
        ("Fine print", question.fine_print),
    ):
        if value and value.strip():
            lines.append(f"\n{label}:\n{value.strip()}")
    return "\n".join(lines)


################################### FORMATOS ###################################


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_RATIONALE = {
    "type": "string",
    "description": "Markdown rationale, 150 to 250 words, written before the numbers.",
}

BINARY_SCHEMA = _object(
    {
        "rationale": _RATIONALE,
        "probability_percent": {
            "type": "number",
            "description": "Probability of Yes, between 1 and 99.",
        },
    }
)


def multiple_choice_schema(options: list[str]) -> dict[str, Any]:
    return _object(
        {
            "rationale": _RATIONALE,
            "options": {
                "type": "array",
                "description": "One entry per option, covering every option.",
                "items": _object(
                    {
                        "option": {"type": "string", "enum": options},
                        "probability_percent": {"type": "number"},
                    }
                ),
            },
        }
    )


def percentile_key(percentile: int) -> str:
    return f"p{percentile:02d}"


NUMERIC_SCHEMA = _object(
    {
        "rationale": _RATIONALE,
        **{
            percentile_key(p): {
                "type": "number",
                "description": f"Value at percentile {p}.",
            }
            for p in PERCENTILES
        },
    }
)

DATE_SCHEMA = _object(
    {
        "rationale": _RATIONALE,
        **{
            percentile_key(p): {
                "type": "string",
                "description": f"Date at percentile {p}, as YYYY-MM-DD.",
            }
            for p in PERCENTILES
        },
    }
)
