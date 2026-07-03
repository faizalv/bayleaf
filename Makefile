.PHONY: test test-integration test-all vet build build-clean export-model

test:
	go test -v -count=1 ./...

test-integration:
	go test -tags integration -v -count=1 -timeout 10m ./...

test-all: vet test-integration

vet:
	go vet ./...

build:
	bash build/build.sh

# Wipe the build cache and rebuild from scratch.
build-clean:
	rm -rf ~/.bayleaf/build-cache
	bash build/build.sh

export-model:
	pip install --extra-index-url https://download.pytorch.org/whl/cpu \
		-r server/requirements-build.txt
	python server/export_onnx.py models
