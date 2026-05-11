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
    verdict: JudgeVerdict = Field(
        description=(
            "CORRECT if the predicted answer is semantically equivalent to the gold; "
            "INCORRECT if it commits to a different value; NOT_ATTEMPTED if it hedges."
        )
    )
    reasoning: str = Field(
        description="One sentence explaining the verdict.", default=""
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
- If the predicted answer hedges without committing to the gold value -> NOT_ATTEMPTED.
- If it commits to a different value -> INCORRECT.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}"""
