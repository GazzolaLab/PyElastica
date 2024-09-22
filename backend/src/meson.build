# first bundle all pure-python files
python_sources = files(
    '__init__.py',
)
py.install_sources(python_sources, subdir: package)

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

_rotations = py.extension_module(
    '_rotations',
    sources: ['_rotations.cpp'],
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
        'Utilities/PrettyType.cpp',
        'Utilities/Demangle.cpp',
    ],
    dependencies: py_deps + blaze_deps + brigand_dep,
    link_language: 'cpp',
    install: true,
    subdir: package,
)

_PyExamples = py.extension_module(
    '_PyExamples',
    sources: [
        'BindExamples/Bindings.cpp',
        'Utilities/Math/Orthogonal.cpp',
        'Utilities/ConvertCase/ConvertCase.cpp',
        'Utilities/PrettyType.cpp',
        'Utilities/Demangle.cpp',
    ],
    dependencies: py_deps + blaze_deps + brigand_dep,
    link_language: 'cpp',
    install: true,
    subdir: package,
)


# _PyCosseratRods = py.extension_module(
#     '_PyCosseratRods',
#     sources: [
#         'Systems/CosseratRods/Python/BindCosseratRod.cpp',
#         'Systems/CosseratRods/Python/Bindings.cpp',
#         'Utilities/ConvertCase/ConvertCase.cpp',
#     ],
#     dependencies: py_deps + blaze_deps + brigand_dep,
#     link_language: 'cpp',
#     install: true,
#     subdir: package,
# )
