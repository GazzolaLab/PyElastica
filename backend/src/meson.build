example_1 = py.extension_module(
    'example_1',
    sources: ['example_1.cpp'],
    dependencies: py_deps,
    link_language: 'cpp',
    install: true,
    subdir: package,
)

_linalg = py.extension_module(
    '_linalg',
    sources: ['_linalg.cpp'],
    dependencies: py_deps + blaze_deps,
    link_language: 'cpp',
    install: true,
    subdir: package,
)

_linalg_numpy = py.extension_module(
    '_linalg_numpy',
    sources: ['_linalg_numpy.cpp'],
    dependencies: py_deps + blaze_deps,
    link_language: 'cpp',
    install: true,
    subdir: package,
)

_PyArrays = py.extension_module(
    '_PyArrays',
    sources: [
        'Utilities/Math/Python/Bindings.cpp',
        'Utilities/Math/Python/BlazeVector.cpp',
        'Utilities/Math/Python/BlazeMatrix.cpp',
        'Utilities/Math/Python/BlazeTensor.cpp',
    ],
    dependencies: py_deps + blaze_deps,
    link_language: 'cpp',
    install: true,
    subdir: package,
)

_PyTags = py.extension_module(
    '_PyTags',
    sources: [
        'Systems/Python/Bindings.cpp',
        'Systems/Python/BindTags.cpp',
    ],
    dependencies: py_deps + blaze_deps + brigand_dep,
    link_language: 'cpp',
    install: true,
    subdir: package,
)
