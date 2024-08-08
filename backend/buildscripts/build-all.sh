
set -e -x

OLD_CWD=$(pwd)

export ELASTICA_DEP_PREFIX="$OLD_CWD/../deps"

mkdir -p $ELASTICA_DEP_PREFIX && cd $ELASTICA_DEP_PREFIX

export ELASTICA_INSTALL_PREFIX="$ELASTICA_DEP_PREFIX/installed"

# With this we
# 1) Force install prefix to $ELASTICA_INSTALL_PREFIX
# 2) use lib directory within $ELASTICA_INSTALL_PREFIX (and not lib64)
# 3) make release binaries
# 4) build shared libraries
# 5) not have @rpath in the linked dylibs (needed on macs only)
# 6) tell cmake to search in $ELASTICA_INSTALL_PREFIX for sub dependencies
export ELASTICA_BASE_CMAKE_FLAGS="-DCMAKE_INSTALL_PREFIX=$ELASTICA_INSTALL_PREFIX \
    -DCMAKE_INSTALL_LIBDIR:PATH=lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=true \
    -DCMAKE_INSTALL_NAME_DIR=$ELASTICA_INSTALL_PREFIX/lib \
    -DCMAKE_PREFIX_PATH=$ELASTICA_INSTALL_PREFIX"

mkdir -p $ELASTICA_INSTALL_PREFIX

# set pkg_config_path so that system can find dependencies
export PKG_CONFIG_PATH="$ELASTICA_INSTALL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# blaze
if [ ! -d blaze ]; then
    git clone --depth 1 https://bitbucket.org/blaze-lib/blaze.git && cd blaze

    mkdir build
    cmake -B build $ELASTICA_BASE_CMAKE_FLAGS \
        -DBLAZE_SHARED_MEMORY_PARALLELIZATION=OFF \
        -DUSE_LAPACK=OFF \
        -S .
    cmake --build build --parallel $(nproc)
    cmake --install build
    cd ..
fi

# blaze tensor
if [ ! -d blaze_tensor ]; then
    git clone --depth 1 https://github.com/STEllAR-GROUP/blaze_tensor.git && cd blaze_tensor

    mkdir build
    cmake -B build $ELASTICA_BASE_CMAKE_FLAGS \
        -Dblaze_DIR="$ELASTICA_INSTALL_PREFIX/share/blaze/cmake/" \
        -S .
    cmake --build build --parallel $(nproc)
    cmake --install build
    cd ..
fi

# sleef
if [ ! -d sleef ]; then
    git clone --depth 1 https://github.com/shibatch/sleef && cd sleef

    mkdir build
    cmake -B build $ELASTICA_BASE_CMAKE_FLAGS \
        -S .
    cmake --build build --parallel $(nproc)
    cmake --install build
    cd ..
fi

# brigand
if [ ! -d brigand ]; then
    git clone --depth 1 https://github.com/edouarda/brigand.git && cd brigand

    mkdir build
    cmake -B build $ELASTICA_BASE_CMAKE_FLAGS \
        -S .
    cmake --build build --parallel $(nproc)
    cmake --install build
    cd ..
fi

cd $OLD_CWD