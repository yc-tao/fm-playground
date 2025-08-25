from fm_playground.datasets.base import load_jsonl, Sample


def test_load_jsonl(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"input":"hi","target":"hello"}\n', encoding="utf-8")
    samples = list(load_jsonl(str(data)))
    assert len(samples) == 1
    assert isinstance(samples[0], Sample)
    assert samples[0].input == "hi"
    assert samples[0].target == "hello"
