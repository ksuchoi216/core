from langfuse import Langfuse, Evaluation
from openai import OpenAI
from typing import Callable


def judge_by_llm(*, prompt, input, output, expected_output, model_name="gpt-4o-mini"):
    langfuse = Langfuse()
    openai_client = OpenAI()
    # Use any model you prefer to act as your judge
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    verdict = completion.choices[0].message.content

    # Parse the judgment criteria
    is_pass = "PASS" in verdict.upper()

    # Return a strongly-typed Langfuse Evaluation object
    return Evaluation(
        name="llm_correctness_judge",
        value=1.0 if is_pass else 0.0,
        data_type="BOOLEAN",
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
