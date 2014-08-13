# Caffe Model Zoo

A caffe model is distributed as a directory containing:
- solver/model prototxt(s)
- model binary file, with .caffemodel extension
- readme.md, containing:
  - YAML header:
    - model file URL or (torrent magnet link) and MD5 hash
    - Caffe commit hash use to train this model
    - [optional] github gist id
    - license type or text
  - main body: free-form description/details
- helpful scripts

It is up to the user where to host the model file.
Dropbox or their own server are both fine.

We provide scripts:

- publish_model_as_gist.sh: uploads non-binary files in the model directory as a Github Gist and returns the id. If gist id is already part of the readme, then updates existing gist.
- download_model_from_gist.sh <gist_id>: downloads the non-binary files from a Gist.
- download_model_binary.py: downloads the .caffemodel from the URL specified in readme.

The Gist is a good format for distribution because it can contain multiple files, is versionable, and has in-browser syntax highlighting and markdown rendering.

The existing models distributed with Caffe can stay bundled with Caffe, so I am re-working them all into this format.
All relevant examples will be updated to start with `cd models/model_of_interest && ../scripts/download_model_binary.sh`.

## Tasks

- get the imagenet example to work with the new prototxt location
