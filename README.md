# myeval

Thin wrapper around [Lighteval](https://github.com/huggingface/lighteval) for simple LLM evaluations.

## Installation

```bash
pip install -e .
```

## Running

```bash
python -m myeval.cli --config configs/example.yaml
```

Outputs are written to `./evals_out/`.

## Switching backends

Set `backend: vllm` in the config to use [vLLM](https://github.com/vllm-project/vllm) instead of `accelerate` (requires GPU and installing the optional `vllm` dependency).

## Docker

Build CPU image:

```bash
docker build -f docker/Dockerfile.cpu -t myeval:cpu .
```

Run:

```bash
docker run --rm -v $PWD:/app myeval:cpu python -m myeval.cli --config configs/example.yaml
```

For CUDA:

```bash
docker build -f docker/Dockerfile.cuda -t myeval:cuda .
docker run --rm --gpus all --ipc=host -v $PWD:/app myeval:cuda python -m myeval.cli --config configs/example.yaml
```

## Configuration

See `configs/example.yaml` for all options. Data sources can be JSONL, CSV or HuggingFace datasets. For CSV/JSONL you can customise column names via `input_col` and `target_col`.

## Common issues

- **Out of memory**: reduce `max_new_tokens` or use a smaller model.
- **Missing GPU**: use the CPU Docker image or run with `backend: accelerate` on CPU.
- **HF token**: set `HF_HOME` or login with `huggingface-cli login` if accessing private models.

## License

MIT
