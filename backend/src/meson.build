example_1 = py.extension_module(
    'example_1',
    sources: ['example_1.cpp'],
    dependencies: [py_dep, pybind11_dep],
    link_language: 'cpp',
    install: true,
    subdir: package,
)

_linalg = py.extension_module(
    '_linalg',
    sources: ['_linalg.cpp'],
    dependencies: [py_dep, pybind11_dep, blaze_dep],
    link_language: 'cpp',
    install: true,
    subdir: package,
)
