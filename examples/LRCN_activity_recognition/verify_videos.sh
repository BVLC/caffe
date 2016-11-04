#!/bin/bash
# 
# Example of invoking script:
# ./verify_videos.sh ./frames list_of_video_images_to_test

FRAMES_DIR="./frames/"

CATEGORIES=`ls $FRAMES_DIR | cut -d "_" -f 1-2 | uniq`

RunVideosClassification() 
{
COUNTER=0
SUCCESS_COUNTER=0
# Get Label out of class name and remove white characters
LABEL=`echo $1 | cut -d "_" -f 2 | sed 's: ::g'`

for movie_name in `ls  $FRAMES_DIR | grep $1`; do
  echo "movie_name: $movie_name"
  RESPONSE=`./classify_video.py $movie_name 2>/dev/null`
  echo "$RESPONSE"
  CLASSIFICATION=`echo $RESPONSE | cut -d ":" -f 2 | cut -d "." -f 1 | sed 's: ::g'`
  
  if [ "$LABEL" == "$CLASSIFICATION" ]; then
    echo "OK (Label: $LABEL, Classification: $CLASSIFICATION)"
    ((SUCCESS_COUNTER+=1))
  else 
    echo "Fail!!! (Label: $LABEL, Classification: $CLASSIFICATION)"
  fi
  ((COUNTER+=1))
done
echo "CLASS($LABEL) Accuracy: 0"`bc <<< "scale=2;$SUCCESS_COUNTER/$COUNTER"`
}

RunUCF101TestSplit01Classification()
{
  COUNTER=0
  SUCCESS_COUNTER=0
  while read -r line || [[ -n "$line" ]]; do
      movie_name=`echo $line | cut -d "/" -f 2 | cut -d " " -f 1`
      echo "Movie name: $movie_name"
      LABEL=`echo $movie_name | cut -d "_" -f 2 | sed 's: ::g'`
#      RESPONSE=`./classify_video.py $movie_name 2>/dev/null`
      RESPONSE=`./classify_video.py $movie_name $FRAMES_DIR 2>/tmp/verify_video_log.$COUNTER | tee -a /tmp/verify_video_log.$COUNTER`
      echo "RESPONSE: $RESPONSE"
      CLASSIFICATION=`echo $RESPONSE | cut -d ":" -f 2 | cut -d "." -f 1 | sed 's: ::g'`
      if [ "$LABEL" == "$CLASSIFICATION" ]; then
        echo "OK (Label: $LABEL, Classification: $CLASSIFICATION)"
        ((SUCCESS_COUNTER+=1))
      else 
        echo "Fail!!! (Label: $LABEL, Classification: $CLASSIFICATION)"
      fi
      ((COUNTER+=1))
  done < ucf101_split1_testVideos.txt
  echo "---> Accuracy: 0"`bc <<< "scale=2;$SUCCESS_COUNTER/$COUNTER"`
}


if [ "$#" -ne 1 ]; then
    echo ""
    echo "    Error: Illegal number of parameters. "
    echo ""
    echo "  Syntax : "
    echo "    ./verify_videos.sh <directory holding frames of UCF-101 videos eg. \"frames/\">"
    echo ""
else

  # Make classification of first standard test split of UCF-101 
  # and present overall accuracy
  RunUCF101TestSplit01Classification

  # Run classification on whole UCF-101
  # and present per category accuracy
  #for movie_categorie in $CATEGORIES; do
  #  RunVideosClassification $movie_categorie
  #done
fi




