#* Variables
PYTHON := python3
PYTHONPATH := `pwd`/..
PYTHON_VERSION :=

#* Installation
.PHONY: install
install:
	uv pip install "." -v

#* C++ Build (direct CMake, no pip)
.PHONY: build-cpp
build-cpp:
	@echo "Building C++ code with CMake..." && \
	VENV_PYTHON=$$(which python) && \
	PYBIND11_DIR=$$(python -c "import pybind11; import os; print(os.path.join(os.path.dirname(pybind11.__file__), 'share', 'cmake', 'pybind11'))") && \
	mkdir -p build && \
	cd build && \
		LIBOMP_PREFIX=$$(brew --prefix libomp 2>/dev/null || echo "") && \
		LLVM_PREFIX=$$(brew --prefix llvm 2>/dev/null || echo "") && \
		cmake .. \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_C_COMPILER=$$LLVM_PREFIX/bin/clang \
			-DCMAKE_CXX_COMPILER=$$LLVM_PREFIX/bin/clang++ \
			-DPython3_ROOT_DIR=$$(cd .. && pwd)/../.venv \
			-DPython3_EXECUTABLE="$$VENV_PYTHON" \
			-DPYTHON_EXECUTABLE="$$VENV_PYTHON" \
			-DPython_EXECUTABLE="$$VENV_PYTHON" \
			-Dpybind11_DIR="$$PYBIND11_DIR" \
			-DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$$LIBOMP_PREFIX/include" \
			-DOpenMP_CXX_LIB_NAMES="omp" \
			-DOpenMP_omp_LIBRARY="$$LIBOMP_PREFIX/lib/libomp.dylib" \
			-DOpenMP_CXX_LIBRARIES="$$LIBOMP_PREFIX/lib/libomp.dylib" \
			-DELASTICAPP_VECTORIZATION_REPORTS=ON \
			-DELASTICAPP_SAVE_ASSEMBLY=ON \
			-DELASTICAPP_BUILD_TESTS=OFF; \
		cmake --build . --parallel 2>&1 | tee ../vectorization_report.txt

#* C++ Tests (build and run, no pip)
.PHONY: test-cpp-direct
test-cpp-direct: build-cpp
	@echo "Running C++ tests..."
	@cd build && ./cpp_tests

#* Testing
.PHONY: test
test:
	@echo "Running C++ and Python tests..."
	ELASTICAPP_BUILD_TESTS=ON make install && cd build && ./cpp_tests
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
