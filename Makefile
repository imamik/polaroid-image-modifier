.PHONY: all build run test clean lint install

# Binary name
BINARY_NAME=polaroid

# Build directory
BUILD_DIR=bin

all: build

build:
	@echo "Building..."
	@mkdir -p $(BUILD_DIR)
	go build -o $(BUILD_DIR)/$(BINARY_NAME) ./cmd/polaroid

run:
	go run ./cmd/polaroid

test:
	@echo "Running tests..."
	go test -v ./...

clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR)
	@go clean

lint:
	@echo "Linting..."
	golangci-lint run

install:
	@echo "Installing..."
	go install ./cmd/polaroid
