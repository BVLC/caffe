#!/bin/bash

function usage
{
    script_name=$0
    echo "Usage:"
    echo "  $script_name [--multinode] [--compiler icc/gcc] [--rebuild] [--boost_root boost_install_dir]"
    echo ""
    echo "  Parameters:"
    echo "    multinode:  specify it to build caffe for multinode. build for single node"
    echo "                by default."
    echo "    compiler:   specify compiler to build intel caffe. default compiler is icc."
    echo "    rebuild:    make clean/remove build directory before building caffe if the "
    echo "                option is specified. not to make clean by default."
    echo "    boost_root: specify directory for boost root (installation directory). if "
    echo "                it's not specified (by default), script will download boost in "
    echo "                directory of caffe source and build it."
}

function check_dependency
{
    dep=$1
    which $dep >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Warning: cannot find $dep"
        return 1
    fi
    return 0
}

function build_caffe_gcc
{
    is_multinode_=$1

    cp Makefile.config.example Makefile.config

    if [ $is_multinode_ -eq 1 ]; then
        echo "USE_MLSL := 1" >> Makefile.config
        echo "CAFFE_PER_LAYER_TIMINGS := 1" >> Makefile.config

        mlslvars_sh=`find external/mlsl/ -name mlslvars.sh`
        if [ -f $mlslvars_sh ]; then
            source $mlslvars_sh
        fi
    fi

    if [ $is_rebuild -eq 1 ]; then
        make clean
    fi

    make -j 8
}

function download_build_boost
{
    # download boost
    pushd ${root_dir}

    boost_lib=boost_1_64_0
    boost_zip_file=${boost_lib}.tar.bz2
    # clean 
    if [ -f $boost_zip_file ]; then
        rm $boost_zip_file
    fi

    echo "Download boost library..."
    wget -c -t 0 https://dl.bintray.com/boostorg/release/1.64.0/source/${boost_zip_file}
    echo "Unzip..."
    tar -jxf ${boost_zip_file}
    pushd ${boost_lib}

    # build boost
    echo "Build boost library..."
    boost_root=${root_dir}/${boost_lib}/install
    ./bootstrap.sh
    ./b2 install --prefix=$boost_root

    popd
    popd
}

function build_caffe_icc
{
    is_multinode_=$1
    cmake_params="-DCPU_ONLY=1 -DBOOST_ROOT=$boost_root"
    if [ $is_multinode_ -eq 1 ]; then
        cmake_params+=" -DUSE_MLSL=1 -DCAFFE_PER_LAYER_TIMINGS=1"
    fi

    build_dir=$root_dir/build
    if [ $is_rebuild -eq 1 ] && [ -d $build_dir ]; then
        rm -r $build_dir
    fi

    mkdir $build_dir
    cd $build_dir

    echo "Parameters: $cmake_params"
    CC=icc CXX=icpc cmake .. $cmake_params
    CC=icc CXX=icpc make all -j 8
}

function sync_caffe_dir
{
  caffe_dir=`pwd`
  caffe_parent_dir=`dirname $caffe_dir`
  which ansible >/dev/null
  if [ $? -eq 0 ]; then
      ansible ourcluster -m synchronize -a "src=$caffe_dir dest=$caffe_parent_dir"
  else
      echo "Warning: no ansible command for synchronizing caffe directory in nodes"
  fi
}


root_dir=$(cd $(dirname $(dirname $0)); pwd)

boost_root=""
is_rebuild=0
compiler="icc"
is_multinode=0
while [[ $# -ge 1 ]]
do
    key="$1"
    case $key in
        --multinode)
            is_multinode=1
            ;;
        --rebuild)
            is_rebuild=1
            ;;
        --compiler)
            compiler="$2"
            shift
            ;;
        --boost_root)
            boost_root="$2"
            shift
            ;;    
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
    shift
done


# check compiler
cplus_compiler=""
if [ "$compiler" == "icc" ]; then
    cplus_compiler="icpc"
elif [ $compiler == "gcc" ]; then
    cplus_compiler="g++"
else
    echo "Invalid compiler: $compiler. Exit."
    exit 1  
fi

for bin in $compiler $cplus_compiler
do
    check_dependency $bin
    if [ $? -ne 0 ]; then
        exit 1
    fi
done

echo "Build Intel Caffe by $compiler..."
if [ "$compiler" == "icc" ]; then
    if [ "$boost_root" == "" ]; then
        download_build_boost
    fi
    build_caffe_icc $is_multinode
else
    build_caffe_gcc $is_multinode
fi

if [ $is_multinode -eq 1 ]; then
  sync_caffe_dir
fi

