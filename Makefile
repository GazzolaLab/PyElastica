#* Variables
PYTHON := python3
PYTHONPATH := `pwd`
AUTOFLAKE8_ARGS := -r --exclude '__init__.py' --keep-pass-after-docstring
#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org/ | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org/ | $(PYTHON) - --uninstall

#* Installation
.PHONY: install
install:
	poetry install

.PHONY: install_examples_dependencies
install_examples_dependencies:
	poetry install -E examples
	# sadly pip ffmpeg doesnt work, hence we use conda for ffmpeg
	conda install -c conda-forge ffmpeg

.PHONY: install_with_new_dependency
install_with_new_dependency:
	poetry lock
	poetry install

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

#* Formatters
.PHONY: black
black:
	poetry run black --version
	poetry run black --config pyproject.toml --required-version 21.12b0 elastica tests examples

.PHONY: black-check
black-check:
	poetry run black --version
	poetry run black --diff --check --config pyproject.toml elastica tests examples

.PHONY: flake8
flake8:
	poetry run flake8 --version
	poetry run flake8 elastica tests

.PHONY: autoflake8-check
autoflake8-check:
	poetry run autoflake8 --version
	poetry run autoflake8 $(AUTOFLAKE8_ARGS) elastica tests examples
	poetry run autoflake8 --check $(AUTOFLAKE8_ARGS) elastica tests examples

.PHONY: autoflake8-format
autoflake8-format:
	poetry run autoflake8 --version
	poetry run autoflake8 --in-place $(AUTOFLAKE8_ARGS) elastica tests examples

.PHONY: format-codestyle
format-codestyle: black flake8

.PHONY: test
test:
	poetry run pytest

.PHONY: test_coverage
test_coverage:
	NUMBA_DISABLE_JIT=1 poetry run pytest --cov=elastica

.PHONY: test_coverage_xml
test_coverage_xml:
	NUMBA_DISABLE_JIT=1 poetry run pytest --cov=elastica --cov-report=xml

.PHONY: check-codestyle
check-codestyle: black-check flake8 autoflake8-check

.PHONY: formatting
formatting: format-codestyle

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D pytest@latest coverage@latest pytest-html@latest pytest-cov@latest black@latest

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove ipynbcheckpoints-remove pytestcache-remove

all: format-codestyle cleanup test

ci: check-codestyle
