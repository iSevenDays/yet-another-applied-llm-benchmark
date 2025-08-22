# Makefile for building the thinking handler Go shared library

.PHONY: all clean test build

# Default target
all: build

# Build the shared library
build:
	go build -buildmode=c-shared -o libthinking_handler.so thinking_handler.go
	go build -buildmode=c-shared -o libevaluators.so evaluators.go

# Build for different platforms
build-linux:
	GOOS=linux GOARCH=amd64 go build -buildmode=c-shared -o libthinking_handler_linux.so thinking_handler.go

build-darwin:
	GOOS=darwin GOARCH=amd64 go build -buildmode=c-shared -o libthinking_handler_darwin.so thinking_handler.go
	GOOS=darwin GOARCH=arm64 go build -buildmode=c-shared -o libthinking_handler_darwin_arm64.so thinking_handler.go

build-windows:
	GOOS=windows GOARCH=amd64 go build -buildmode=c-shared -o libthinking_handler.dll thinking_handler.go

# Build for all platforms
build-all: build-linux build-darwin build-windows

# Test the Go code
test:
	go test -v

# Clean build artifacts
clean:
	rm -f libthinking_handler.*
	rm -f thinking_handler.h

# Install dependencies (none currently, but useful for future)
deps:
	go mod tidy
	go mod download

# Format code
fmt:
	go fmt ./...

# Vet code
vet:
	go vet ./...