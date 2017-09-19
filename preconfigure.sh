# author Marios Christodoulou
# installation instructions from:
#	* http://caffe.berkeleyvision.org/install_apt.html
#	* http://caffe.berkeleyvision.org/installation.html#compilation
# bugfix summary found at https://github.com/BVLC/caffe/issues/2347

# check distribution and version
IFS="="
id=
release=
while read -r name value; do
	if [ "$name" == "DISTRIB_ID" ]; then
		id=$value
	elif [ "$name" == "DISTRIB_RELEASE" ]; then
		release=$value
	fi
done < /etc/lsb-release

if ! ([ "$id" == "Ubuntu" ] && [ "$release" == "16.04" ]); then
	echo "Ubuntu 16.04 supported and tested!"
	exit 1
fi

# Instalation
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev

sudo apt-get install -y --no-install-recommends libboost-all-dev

# If you believe the instructions, this is not necessary at Ubuntu 16.04
# But it is!
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev

# Makefile bugfixes
find . -type f -exec sed -i -e 's^"hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;

cd /usr/lib/x86_64-linux-gnu/
sudo ln -s libhdf5_serial.so.10.1.0 libhdf5.so
sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so

cd -

[[ -f Makefile.config ]] || cp Makefile.config.example Makefile.config

sed -i.bak '/^INCLUDE_DIRS/ s/$/ \/usr\/include\/hdf5\/serial\//' Makefile.config
