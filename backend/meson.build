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

fs = import('fs')

message('Running buildscripts/build-all.sh (might take a while)')
buildscripts = files('buildscripts/build-all.sh')
res = run_command('bash', buildscripts, check: false)
message(res.stdout().strip())
if res.returncode() != 0
    error(res.stderr().strip())
endif

deps_installed = fs.parent(meson.current_source_dir()) / 'deps' / 'installed'

deps_inc_dirs = [deps_installed / 'include']
deps_lib_dirs = [deps_installed / 'lib']

# see https://github.com/mesonbuild/meson/issues/7943
# The strategy for the below required dependencies are as follows
# - First, meson tries to resolve the dependency by searching on the system
# - if it doesn't find it, it fallbacks to the subproject wrap system or the
#   ones installed in ./buildscripts 

blaze_dep = dependency('blaze')
blaze_tensor_dep = dependency('BlazeTensor')
sleef_dep = dependency('sleef', required: false)
if not sleef_dep.found()
    sleef_dep = declare_dependency(
        include_directories: deps_inc_dirs,
        dependencies: cc.find_library('sleef', dirs: deps_lib_dirs),
    )
endif

brigand_dep = dependency('brigand', fallback : 'brigand')
cxxopts_dep = dependency('cxxopts', fallback : 'cxxopts')

# meson-python: error: Could not map installation path to an equivalent wheel directory:
# tbb_dep = dependency('tbb', fallback : 'tbb')
# yaml_cpp_dep = dependency('yaml-cpp', fallback : 'yaml-cpp')

# TODO: are these required?
# setup_library "HighFive" "https://github.com/BlueBrain/HighFive"
# setup_library "spline" "https://github.com/tp5uiuc/spline.git"

subdir('src')
