cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=false
data_root_dir="$HOME/data/ILSVRC"
dataset_name="ILSVRC2016"
mapfile="$root_dir/data/$dataset_name/labelmap_ilsvrc_det.prototxt"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if $redo
then
  extra_cmd="$extra_cmd --redo"
fi

for dataset in test
do
  python $root_dir/scripts/create_annoset.py --anno-type="classification" --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$dataset".txt" $data_root_dir/$db/DET/$dataset_name"_"$dataset"_"$db examples/$dataset_name 2>&1 | tee $root_dir/data/$dataset_name/$dataset.log
done

for dataset in val2 trainval1
do
  python $root_dir/scripts/create_annoset.py --anno-type="detection" --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$dataset".txt" $data_root_dir/$db/DET/$dataset_name"_"$dataset"_"$db examples/$dataset_name 2>&1 | tee $root_dir/data/$dataset_name/$dataset.log
done
