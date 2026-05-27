import re
from typing import Callable

from langfuse import Langfuse, Evaluation
from openai import OpenAI


def _parse_float_score(verdict: str) -> float:
    match = re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", verdict.strip())
    if not match:
        raise ValueError(f"Judge output is not a 0-1 score: {verdict!r}")

    score = float(match.group(0))
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Judge score is outside 0-1: {score}")
    return score


def judge_by_llm(*, prompt, input, output, model_name="gpt-4o-mini", **kwargs):
    langfuse = Langfuse()
    openai_client = OpenAI()
    # Use any model you prefer to act as your judge
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    verdict = completion.choices[0].message.content

    score = _parse_float_score(verdict or "")

    # Return a strongly-typed Langfuse Evaluation object
    return Evaluation(
        name="llm_correctness_judge",
        value=score,
        data_type="NUMERIC",
        comment=verdict,
    )


def run_experiment(
    *, dataset_name: str, name: str, task: Callable, judge_by_llm: Callable
):
    langfuse = Langfuse()
    dataset = langfuse.get_dataset(dataset_name)

    dataset.run_experiment(
        name=name,
        task=task,
        evaluators=[judge_by_llm],
    )
