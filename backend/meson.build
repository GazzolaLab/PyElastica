project(
    'elasticapp',
    ['cpp'],
    version: '0.0.3',
    default_options: ['cpp_std=c++17'],
)

package = 'elasticapp'

cc = meson.get_compiler('cpp')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# find dependencies and create dep objects
pybind11_dep = declare_dependency(
    include_directories: run_command(
        py,
        '-c',
        'import pybind11; print(pybind11.get_include());',
        check: true,
    ).stdout().strip(),
)

subdir('src')
