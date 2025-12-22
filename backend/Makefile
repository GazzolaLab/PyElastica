#* Variables
PYTHON := python3
PYTHONPATH := `pwd`/..
PYTHON_VERSION :=

#* Installation
.PHONY: install
install:
	source ../.venv/bin/activate && uv pip install -e "."

#* C++ Build (direct CMake, no pip)
.PHONY: build-cpp
build-cpp:
	@echo "Building C++ code with CMake..." && \
	source ../.venv/bin/activate && \
	VENV_PYTHON=$$(which python) && \
	PYBIND11_DIR=$$(python -c "import pybind11; import os; print(os.path.join(os.path.dirname(pybind11.__file__), 'share', 'cmake', 'pybind11'))") && \
	mkdir -p build && \
	cd build && \
		cmake .. \
			-DCMAKE_BUILD_TYPE=Release \
			-DPython3_ROOT_DIR=$$(cd .. && pwd)/../.venv \
			-DPython3_EXECUTABLE="$$VENV_PYTHON" \
			-DPYTHON_EXECUTABLE="$$VENV_PYTHON" \
			-DPython_EXECUTABLE="$$VENV_PYTHON" \
			-Dpybind11_DIR="$$PYBIND11_DIR" && \
		cmake --build . --parallel

#* C++ Tests (build and run, no pip)
.PHONY: test-cpp-direct
test-cpp-direct: build-cpp
	@echo "Running C++ tests..."
	@cd build && ./cpp_tests

#* Testing
.PHONY: test
test:
	echo "Building tests first..."; \
	source ../.venv/bin/activate
	$(MAKE) install; \
	cd build && ./cpp_tests
	pytest tests/py -v

#* Cleaning
.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

.PHONY: clean-build
clean-build:
	rm -rf build/ dist/ *.egg-info
