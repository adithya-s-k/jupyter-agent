"""Structured-output schema for the LLM-judge tier of the grader.

Used by the `final_answer` ORS tool when tiers 1 (exact) and 2 (numeric)
both miss. Calling `client.beta.chat.completions.parse(response_format=JudgeOut, ...)`
guarantees we get back a `JudgeOut` instance with a parsed enum verdict —
no regex, no hand-rolled prompt-output parsing.

Verdict semantics follow the OpenAI simple-evals SimpleQA grader:
- CORRECT       — semantically equivalent to the gold answer
- INCORRECT     — committed to a different value
- NOT_ATTEMPTED — hedged / refused / never committed

Only CORRECT maps to reward 1.0; everything else maps to 0.0.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JudgeVerdict(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class JudgeOut(BaseModel):
    # Field order matters: with structured output, the model fills fields in
    # declaration order. Putting `reasoning` first forces chain-of-thought
    # before committing to a verdict — Evidently / Databricks / Patronus all
    # report 10–15% better judge–human agreement when reasoning precedes label.
    reasoning: str = Field(
        description=(
            "One short sentence walking through: (a) what core value the gold "
            "expresses, (b) whether the prediction's core value matches it."
        ),
        default="",
    )
    verdict: JudgeVerdict = Field(
        description=(
            "CORRECT if the predicted answer is semantically equivalent to the gold; "
            "INCORRECT if it commits to a different value; NOT_ATTEMPTED if it hedges."
        )
    )


JUDGE_PROMPT = """You are grading short answers from a data-science agent.
Decide whether the predicted answer is semantically equivalent to the gold answer.

Rules:
- Case, punctuation, whitespace, articles ("the"), and trailing units don't matter.
- For numbers: predicted must match to the last significant figure of the gold
  (gold "0.544341", predicted "0.544" -> CORRECT; "0.5" -> INCORRECT).
- Extra surrounding prose is fine if the gold value is clearly stated
  (gold "5", predicted "There are 5 distinct categories" -> CORRECT).
- Common synonyms/abbreviations count (gold "Not Applicable", predicted "N/A" -> CORRECT).
- **Parenthetical annotations in the gold are equivalence hints, not required.**
  (gold "Gandalf (Ainur)", predicted "Gandalf" -> CORRECT;
   gold "No (correlation coefficient = 0.02)", predicted "No" -> CORRECT).
- **Percent + qualifier**: if the gold is "X% in YEAR", a prediction of X or X%
  is CORRECT as long as the core numeric value matches
  (gold "21.334% in 2014", predicted "21.334" -> CORRECT;
   gold "21.334% in 2014", predicted "21.334% in 2013" -> INCORRECT — year mismatch).
- **Multi-part gold**: if the gold lists two values joined by "and" but the
  question only asks for one (e.g. "which model wins?" → gold "0.987 (XGBoost and LGBM)"),
  predicting either model name is CORRECT. If the question explicitly asks for
  both halves, predicting only one is INCORRECT.
- If the predicted answer hedges without committing to the gold value -> NOT_ATTEMPTED.
- If it commits to a different value -> INCORRECT.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}"""
