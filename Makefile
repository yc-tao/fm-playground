.PHONY: setup test run docker-cpu docker-cuda

setup:
pip install -e .[test]

test:
pytest -q

run:
python -m myeval.cli --config $(CONFIG)

docker-cpu:
docker build -f docker/Dockerfile.cpu -t myeval:cpu .

docker-cuda:
docker build -f docker/Dockerfile.cuda -t myeval:cuda .
