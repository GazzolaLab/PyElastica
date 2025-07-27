
set -e -x

OLD_CWD=$(pwd)

export ELASTICA_DEP_PREFIX="$OLD_CWD/../third-party"
export ELASTICA_INSTALL_PREFIX="$OLD_CWD/../third-party-installed"

mkdir -p $ELASTICA_DEP_PREFIX
cd $ELASTICA_DEP_PREFIX


# With this we
# 1) Force install prefix to $ELASTICA_INSTALL_PREFIX
# 2) use lib directory within $ELASTICA_INSTALL_PREFIX (and not lib64)
# 3) make release binaries
# 4) build shared libraries
# 5) not have @rpath in the linked dylibs (needed on macs only)
# 6) tell cmake to search in $ELASTICA_INSTALL_PREFIX for sub dependencies
export ELASTICA_BASE_CMAKE_FLAGS="-DCMAKE_INSTALL_PREFIX=$ELASTICA_INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$ELASTICA_INSTALL_PREFIX"

    # -DCMAKE_INSTALL_LIBDIR:PATH=lib \
    # -DCMAKE_INSTALL_NAME_DIR=$ELASTICA_INSTALL_PREFIX/lib \
    # -DBUILD_SHARED_LIBS=true \
mkdir -p $ELASTICA_INSTALL_PREFIX

# set pkg_config_path so that system can find dependencies
export PKG_CONFIG_PATH="$ELASTICA_INSTALL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# sleef
if [ ! -d sleef ]; then
    git clone https://github.com/shibatch/sleef --branch 3.8 --single-branch  # AVX removed in 3.9
    cd sleef

    mkdir build
    cmake -B build $ELASTICA_BASE_CMAKE_FLAGS \
        -DBUILD_SHARED_LIBS=ON \
        -DSLEEF_SHOW_CONFIG=ON \
        -S .
    cmake --build build --parallel $(nproc) -j --clean-first
    cmake --install build
    cd ..
fi


# blaze
if [ ! -d blaze ]; then
    git clone https://bitbucket.org/blaze-lib/blaze.git && cd blaze

    mkdir build
    cmake -B build $ELASTICA_BASE_CMAKE_FLAGS \
        -DCMAKE_CXX_FLAGS="-march=native -mavx2" \
        -DBLAZE_SHARED_MEMORY_PARALLELIZATION=OFF \
        -DUSE_LAPACK=OFF \
        -S .
        # -DCMAKE_CXX_FLAGS="-DBLAZE_BLAS_MODE=0 \
        #     -DBLAZE_DEFAULT_STORAGE_ORDER=blaze::rowMajor \
        #     -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 \
        #     -DBLAZE_MPI_PARALLEL_MODE=0 \
        #     -DBLAZE_USE_PADDING=1 \
        #     -DBLAZE_USE_STREAMING=1 \
        #     -DBLAZE_USE_DEFAULT_INITIALIZATON=1 \
        #     -DBLAZE_USE_SLEEF=1 \
        #     -DBLAZE_USE_STRONG_INLINE=1 \
        #     -DBLAZE_USE_ALWAYS_INLINE=1 \
        #     -DBLAZE_USE_VECTORIZATION=1"

    cmake --build build --parallel $(nproc)
    cmake --install build
    cd ..
fi

# blaze tensor
if [ ! -d blaze_tensor ]; then
    git clone https://github.com/STEllAR-GROUP/blaze_tensor.git && cd blaze_tensor

    sed -i.bak 's/pos_/position()/g' blaze_tensor/math/expressions/DMatRavelExpr.h
    sed -i.bak 's/pos_/position()/g' blaze_tensor/math/expressions/DTensRavelExpr.h

    mkdir build
    cmake -B build $ELASTICA_BASE_CMAKE_FLAGS \
        -Dblaze_DIR="$ELASTICA_INSTALL_PREFIX/share/blaze/cmake/" \
        -S .
    cmake --build build --parallel $(nproc)
    cmake --install build
    cd ..
fi

# brigand
if [ ! -d brigand ]; then
    git clone https://github.com/edouarda/brigand.git && cd brigand

    mkdir build
    cmake -B build $ELASTICA_BASE_CMAKE_FLAGS \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -S .
    cmake --build build --parallel $(nproc)
    cmake --install build
    cd ..
fi

cd $OLD_CWD