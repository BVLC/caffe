#!/usr/bin/env sh
#
# This script is used to extract the ILSVRC training and validation set of
# images from the available .tar files that can be downloaded from
# http://www.image-net.org/index (after registration).

# Create the training and validation data from the ILSVRC2012 training and
# validation tar archives.

# Specify the correct PATH to the folder that contains the ILSVRC .tar files; 
# this is an example!
DATAPATH=~/path/to/imagenet/ILSVRC2012/archived/files

# Prameters that, probably, do not need to change
ValidationSetName=ILSVRC2012_img_val.tar
ValidationFolderName=val
ValidationSet=$DATAPATH/$ValidationSetName
ValidationFolder=$DATAPATH/$ValidationFolderName
echo "$ValidationSet";
echo "$ValidationFolder";

TrainingSetName=ILSVRC2012_img_train.tar
TrainingFolderName=train
TrainingSet=$DATAPATH/$TrainingSetName
TrainingFolder=$DATAPATH/$TrainingFolderName
echo "$TrainingSet";
echo "$TrainingFolder";


echo "***********************************************************************";
echo "* This script extracts the training and validation set from the .tar  *";
echo "* archives that are available from the ImageNet website. Keep in mind *";
echo "* that it will take time to extract all the images. Be patient!       *";
echo "***********************************************************************";

# Create a folder for the validation set and extract images
echo "Create a folder for the validation set."
rm -rf $ValidationFolder
mkdir $ValidationFolder

echo "Extracting validation set...";

tar -xf $ValidationSet --directory $ValidationFolder

echo "Validation set extracted successfully!";


# Create a folder for the training set and extract the image archives
echo "Create a folder for the training set.";
rm -rf $TrainingFolder
mkdir $TrainingFolder

echo "Extracting training set... (this process will take time)";

tar -xf $TrainingSet --directory $TrainingFolder

# For every (image) archive that was into the training set .tar file,
# create a folder (for that synset) and extract images
for synset_tar in $TrainingFolder/*.tar;
do

  # remove the path from the synset name
  synset_tar_name=$(basename "$synset_tar")
  
  # keep the extension (.tar) of the synset name
  extension="${synset_tar_name##*.}"
  
  # remove the extension (.tar) of the synset name
  folder_name="${synset_tar_name%.*}"
  
  # Create the synset folder and extract synset's images
  echo -ne "Extract $synset_tar_name into the $TrainingFolder/$folder_name";
  echo " folder...";

  rm -rf $TrainingFolder/$folder_name
  mkdir $TrainingFolder/$folder_name
  
  tar -xf $TrainingFolder/$synset_tar_name \
      --directory $TrainingFolder/$folder_name

  # delete the extracted synset to save space
  rm -rf $TrainingFolder/$synset_tar_name
done

echo "Training set extracted successfully!"
