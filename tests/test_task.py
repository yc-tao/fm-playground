from myeval.tasks.registry import get_task
from myeval.datasets.base import Sample


def test_task_prompt_and_metric():
    task = get_task("qa_em")
    sample = Sample(input="1+1?", target="2")
    item = task.to_lighteval_item(sample)
    assert "Question:" in item["prompt"]
    pred = "2"
    metrics = task.metric_fn(pred, item["reference"])
    assert metrics["em"] == 1.0
