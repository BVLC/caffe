echo "usage: download_model_from_gist.sh <gist_id> <dirname>"

GIST=$1
DIRNAME=$2

if [ -d "$DIRNAME/$GIST" ]; then
    echo "$DIRNAME/$GIST already exists! Please make sure you're not overwriting anything important!"
    exit
fi

echo "Downloading Caffe model info to $DIRNAME/$GIST ..."
mkdir -p $DIRNAME/$GIST
wget https://gist.github.com/$GIST/download -O $DIRNAME/$GIST/gist.tar.gz
tar xzf $DIRNAME/$GIST/gist.tar.gz --directory=$DIRNAME/$GIST --strip-components=1
rm $DIRNAME/$GIST/gist.tar.gz
echo "Done"
