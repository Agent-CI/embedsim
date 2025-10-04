.PHONY: help test test-coverage

help:
	@echo "EmbedSim - Makefile Commands"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all tests"
	@echo "  make test-coverage     - Run tests with coverage report"

# Testing
test:
	uv run pytest tests/ -v

test-coverage:
	uv run pytest tests/ --cov=embedsim --cov-report=html --cov-report=term
