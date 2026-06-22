.PHONY: test test-integration test-all vet build export-model

test:
	go test -v -count=1 ./...

test-integration:
	go test -tags integration -v -count=1 -timeout 10m ./...

test-all: vet test-integration

vet:
	go vet ./...

build:
	bash build/build.sh

export-model:
	pip install -r server/requirements-build.txt -r server/requirements.txt
	python server/export_onnx.py models
