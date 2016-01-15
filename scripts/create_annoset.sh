#!/usr/bin/env sh
# Create the annotation $DB_TYPE inputs
# Example:

if [ $# -gt 11 ] || [ $# -lt 3 ]
then
  echo "Usage: $0 ROOTDIR dataset_name subset [anno_type=detection] [mapfile="empty.txt"] [DB=lmdb] [shuffle=0] [resize=0] [check_size=1] [gray=0] [encoded=0]"
  exit
fi

ROOTDIR=$1
dataset_name=$2
subset=$3
anno_type=detection
if [ $# -ge 4 ]
then
  anno_type=$4
fi
mapfile="empty.txt"
if [ $# -ge 5 ]
then
  mapfile=$5
fi
DB_TYPE=lmdb
if [ $# -ge 6 ]
then
  DB_TYPE=$6
fi
shuffle=0
if [ $# -ge 7 ]
then
  shuffle=$7
fi
resize=0
if [ $# -ge 8 ]
then
  resize=$8
fi
check_size=1
if [ $# -ge 9 ]
then
  check_size=$9
fi
gray=0
if [ $# -ge 10 ]
then
  gray=${10}
fi
encoded=0
if [ $# -ge 11 ]
then
  encoded=${11}
fi

EXAMPLE=$ROOTDIR/$dataset_name/$DB_TYPE
DATA=data/$dataset_name
TOOLS=build/tools

DATA_ROOT=$ROOTDIR/$dataset_name/

if [ ! -d $EXAMPLE ]
then
  mkdir -p $EXAMPLE
fi

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=$resize
if [ $RESIZE -ne 0 ]
then
  RESIZE_HEIGHT=448
  RESIZE_WIDTH=448
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
  echo "Set the DATA_ROOT variable in $0 to the path" \
       "where the image/annotation data is stored."
  exit 1
fi

echo "Creating $dataset_name $subset $DB_TYPE..."

if [ -d $EXAMPLE/"$dataset_name"_"$subset"_$DB_TYPE ]
then
  rm -r $EXAMPLE/"$dataset_name"_"$subset"_$DB_TYPE
fi

GLOG_logtostderr=1 $TOOLS/convert_annoset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=$shuffle \
    --check_size=$check_size \
    --gray=$gray \
    --anno_type=$anno_type \
    --label_map_file=$mapfile \
    --strict_check=1 \
    --backend=$DB_TYPE \
    --encoded=$encoded \
    $DATA_ROOT \
    $DATA/$subset.txt \
    $EXAMPLE/"$dataset_name"_"$subset"_$DB_TYPE

if [ ! -d examples/$dataset_name ]
then
  mkdir examples/$dataset_name
fi
rm -f examples/$dataset_name/"$dataset_name"_"$subset"_$DB_TYPE
ln -s $EXAMPLE/"$dataset_name"_"$subset"_$DB_TYPE examples/$dataset_name/

echo "Done."
