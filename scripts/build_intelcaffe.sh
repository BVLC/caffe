#!/bin/bash

function usage
{
    script_name=$0
    echo "Usage:"
    echo "  $script_name [--multinode] [--compiler icc/gcc] [--rebuild] "
    echo "               [--boost_root boost_install_dir] [--layer_timing]"
    echo "               [--debug] [--build_option]"
    echo ""
    echo "  Parameters:"
    echo "    multinode:    specify it to build caffe for multinode. build for single"
    echo "                  node by default."
    echo "    compiler:     specify compiler to build intel caffe. default is icc."
    echo "    rebuild:      make clean/remove build directory before building caffe if"
    echo "                  the option is specified. not to make clean by default."
    echo "    boost_root:   specify directory for boost root (installation directory)."
    echo "                  if it's not specified (by default), script will download"
    echo "                  boost in directory of caffe source and build it."
    echo "    layer_timing: build caffe for multinode with CAFFE_PER_LAYER_TIMINGS flag."
    echo "                  by default, the flag is NOT included for build."
    echo "    debug:        build caffe with debug flag. by default, the option is off."
    echo "    build_option: build option to disable optimization. by default, the option is blank."
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
    build_option_=$2
    cp Makefile.config.example Makefile.config

    if [ $is_multinode_ -eq 1 ]; then
        echo "USE_MLSL := 1" >> Makefile.config
        echo "ALLOW_LMDB_NOLOCK := 1" >> Makefile.config

        if [ -z $MLSL_ROOT ]; then
            mlslvars_sh=`find external/mlsl/ -name mlslvars.sh`
            if [ -f $mlslvars_sh ]; then
                source $mlslvars_sh
            fi
        fi
    fi

    if [ $is_layer_timing -eq 1 ]; then
        echo "CAFFE_PER_LAYER_TIMINGS := 1" >> Makefile.config
    fi

    if [ $debug -eq 1 ]; then
        echo "DEBUG := 1" >> Makefile.config
    fi

    if [ "$boost_root" != "" ]; then
        echo "BOOST_ROOT := $boost_root" >> Makefile.config
    fi
    for option in $build_option_
    do
        grep "$option\s*:= 0" Makefile.config > /dev/null
        if [ $? -eq 0 ]; then
           sed -i "s/$option\s*:= 0/$option := 1/g" Makefile.config
        fi
    done
    if [ $is_rebuild -eq 1 ]; then
        make clean
    fi
    
    make -j $(nproc)
    make pycaffe
}

function download_build_boost
{
    # download boost
    pushd ${root_dir} >/dev/null

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
    pushd ${boost_lib} >/dev/null

    # build boost
    echo "Build boost library..."
    boost_root=${root_dir}/${boost_lib}/install
    ./bootstrap.sh
    ./b2 install --prefix=$boost_root

    popd >/dev/null
    popd >/dev/null
}

function build_caffe_icc
{
    is_multinode_=$1
    build_option_=$2
    cmake_params="-DCPU_ONLY=1 -DBOOST_ROOT=$boost_root"
    if [ $is_multinode_ -eq 1 ]; then
        cmake_params+=" -DUSE_MLSL=1 -DALLOW_LMDB_NOLOCK=1"
    fi

    if [ $is_layer_timing -eq 1 ]; then
        cmake_params+=" -DCAFFE_PER_LAYER_TIMINGS=1"
    fi

    if [ $debug -eq 1 ]; then
        cmake_params+=" -DDEBUG=1"
    fi
    for option in $build_option_
    do
        cmake_params+=" -D$option=1 "
    done
    build_dir=$root_dir/build
    if [ $is_rebuild -eq 1 ] && [ -d $build_dir ]; then
        rm -r $build_dir
    fi

    mkdir $build_dir
    cd $build_dir

    echo "Parameters: $cmake_params"
    CC=icc CXX=icpc cmake .. $cmake_params
    CC=icc CXX=icpc make all -j $(nproc) 
}

function sync_caffe_dir
{
    echo "Synchronize caffe binary between nodes..."
    caffe_dir=`pwd`
    caffe_parent_dir=`dirname $caffe_dir`
    which ansible >/dev/null
    if [ $? -eq 0 ]; then
        set -x
        ansible ourcluster -m synchronize -a "src=$caffe_dir dest=$caffe_parent_dir"
        set +x
    else
        echo "Warning: no ansible command for synchronizing caffe directory in nodes"
    fi
}


root_dir=$(cd $(dirname $(dirname $0)); pwd)

debug=0
is_layer_timing=0
boost_root=""
is_rebuild=0
compiler="icc"
is_multinode=0
build_option=""
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
        --build_option)
            build_option="$2"
            shift
            ;;
        --layer_timing)
            is_layer_timing=1
            ;;
        --help)
            usage
            exit 0
            ;;
        --debug)
            debug=1
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

# Fix the compilation failure if icc environment is set.
# During building caffe, MKL library will be downloaded,
# and the environment variable will be set.
unset MKLROOT
unset CPATH

echo "Build Intel Caffe by $compiler..."
if [ "$compiler" == "icc" ]; then
    if [ "$boost_root" == "" ]; then
        download_build_boost
    fi

    if [[ "$boost_root" != /* ]]; then
        boost_root=$(cd $boost_root; pwd)
    fi

    build_caffe_icc $is_multinode "$build_option"
else
    build_caffe_gcc $is_multinode "$build_option"
fi

if [ $is_multinode -eq 1 ]; then
    sync_caffe_dir
fi

