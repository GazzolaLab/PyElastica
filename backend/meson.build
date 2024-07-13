project(
    'elasticapp',
    ['cpp'],
    version: '0.0.3',
    default_options: ['cpp_std=c++14'],
)

add_global_arguments(['-Wno-unused'],
                     language: 'cpp')

package = 'elasticapp'

cc = meson.get_compiler('cpp')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# set up global defines to optimize blaze usages
add_global_arguments(
    # - Enable external BLAS kernels
    '-DBLAZE_BLAS_MODE=0',
    # - Set default matrix storage order to column-major, since many of our
    #   functions are implemented for column-major layout. This default reduces
    #   conversions.
    '-DBLAZE_DEFAULT_STORAGE_ORDER=blaze::rowMajor',
    # - Disable SMP parallelization. This disables SMP parallelization for all
    #   possible backends (OpenMP, C++11 threads, Boost, HPX):
    #   https://bitbucket.org/blaze-lib/blaze/wiki/Serial%20Execution#!option-3-deactivation-of-parallel-execution
    '-DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0',
    # - Disable MPI parallelization
    '-DBLAZE_MPI_PARALLEL_MODE=0',
    # - Using the default cache size, which may have been configured automatically
    #   by the Blaze CMake configuration for the machine we are running on. We
    #   could override it here explicitly to tune performance.
    # BLAZE_CACHE_SIZE
    '-DBLAZE_USE_PADDING=1',
    # - Always enable non-temporal stores for cache optimization of large data
    #   structures: https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20Files#!streaming-non-temporal-stores
    '-DBLAZE_USE_STREAMING=1',
    # - Initializing default-constructed structures for fundamental types
    '-DBLAZE_USE_DEFAULT_INITIALIZATON=1',
    # Use Sleef for vectorization of more math functions
    '-DBLAZE_USE_SLEEF=1',
    # Set inlining settings
    '-DBLAZE_USE_STRONG_INLINE=1',
    '-DBLAZE_USE_ALWAYS_INLINE=1',
    # Set vectorization (leave to 1, else there is no use for blaze)
    '-DBLAZE_USE_VECTORIZATION=1',
    language: 'cpp'
)

# find dependencies and create dep objects
pybind11_dep = declare_dependency(
    include_directories: run_command(
        py,
        '-c',
        'import pybind11; print(pybind11.get_include());',
        check: true,
    ).stdout().strip(),
)

# python and pybind deps are used together
py_deps = [py_dep, pybind11_dep]

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

# blaze and deps commonly used together
blaze_deps = [blaze_dep, blaze_tensor_dep, sleef_dep]

brigand_dep = dependency('brigand', required: false)
if not brigand_dep.found()
    brigand_dep = declare_dependency(include_directories: deps_inc_dirs)
endif

cxxopts_dep = dependency('cxxopts', fallback : 'cxxopts')

# meson-python: error: Could not map installation path to an equivalent wheel directory:
# tbb_dep = dependency('tbb', fallback : 'tbb')
# yaml_cpp_dep = dependency('yaml-cpp', fallback : 'yaml-cpp')

# TODO: are these required?
# setup_library "HighFive" "https://github.com/BlueBrain/HighFive"
# setup_library "spline" "https://github.com/tp5uiuc/spline.git"

subdir('src')
