#!/bin/bash

root_dir=$HOME/data/VOCdevkit/
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in VOC2007 VOC2012
  do
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    rand_file=$bash_dir/$dataset.random
    if [ $dataset == "test" ]
    then
      cp $dataset_file $rand_file
    else
      cat $dataset_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    fi

    img_file=$bash_dir/$dataset"_img.txt"
    cp $rand_file $img_file
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file

    label_file=$bash_dir/$dataset"_label.txt"
    cp $rand_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file

    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file
    rm -f $rand_file
  done
done
