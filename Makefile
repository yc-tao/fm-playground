.PHONY: setup test run docker-cpu docker-cuda

setup:
pip install -e .[test]

test:
pytest -q

run:
    python -m fm_playground.cli --config $(CONFIG)

docker-cpu:
    docker build -f docker/Dockerfile.cpu -t fm-playground:cpu .

docker-cuda:
    docker build -f docker/Dockerfile.cuda -t fm-playground:cuda .
