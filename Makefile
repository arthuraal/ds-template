.PHONY: tests docs

initialize_git:
	@echo "Initializing Git..."
	git init

dependencies: 
	@echo "Installing dependencies..."
	poetry install --no-root
	poetry run pre-commit install

env: dependencies
	@echo "Activating virtual environment..."
	poetry shell

tests:
	pytest

docs:
	@echo Save documentation to docs... 
	pdoc src -o docs --force
	@echo View API documentation... 
	pdoc src --http localhost:8080	
