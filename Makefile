#* Variables
PYTHON := python3
PYTHONPATH := `pwd`
AUTOFLAKE_ARGS := -r

#* Installation
.PHONY: install
install:
	uv sync

.PHONY: install-dev-deps
install-dev-deps:
	uv sync --all-groups --all-extras


.PHONY: install_examples_dependencies
install_examples_dependencies:
	uv pip install -e ".[examples]"
	# sadly pip ffmpeg doesnt work, hence we use conda for ffmpeg
	conda install -c conda-forge ffmpeg

.PHONY: pre-commit-install
pre-commit-install:
	pre-commit install

#* Formatters
.PHONY: black
black:
	black --version
	black --config pyproject.toml --required-version 24.3.0 elastica tests examples

.PHONY: black-check
black-check:
	black --version
	black --diff --check --config pyproject.toml elastica tests examples

.PHONY: flake8
flake8:
	flake8 --version
	flake8 elastica tests

.PHONY: autoflake-check
autoflake-check:
	autoflake --version
	autoflake --check $(AUTOFLAKE_ARGS) elastica tests examples

.PHONY: autoflake-format
autoflake-format:
	autoflake --version
	autoflake --in-place $(AUTOFLAKE_ARGS) elastica tests examples

.PHONY: format-codestyle
format-codestyle: black autoflake-format

.PHONY: mypy
mypy:
	uv run mypy --config-file pyproject.toml elastica
	uv run mypy --config-file pyproject.toml --explicit-package-bases \
		examples/AxialStretchingCase \
		examples/ButterflyCase \
		examples/CatenaryCase

.PHONY: test
test:
	pytest

.PHONY: test_coverage
test_coverage:
	NUMBA_DISABLE_JIT=1 pytest --cov=elastica

.PHONY: test_coverage_xml
test_coverage_xml:
	NUMBA_DISABLE_JIT=1 pytest --cov=elastica --cov-report=xml

.PHONY: check-codestyle
check-codestyle: black-check flake8 autoflake-check

.PHONY: formatting
formatting: format-codestyle

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

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
cleanup: pycache-remove dsstore-remove ipynbcheckpoints-remove pytestcache-remove mypycache-remove

all: format-codestyle cleanup test

ci: check-codestyle
