#!/bin/bash
git checkout master
git branch -D future
git checkout -b future
## merge PRs
# coord maps, net pointer, crop layer
hub merge https://github.com/BVLC/caffe/pull/1976
# gradient accumulation
hub merge https://github.com/BVLC/caffe/pull/1977
# python net spec
hub merge https://github.com/BVLC/caffe/pull/2086
## commit
cat <<"END" > README.md
This is a pre-release Caffe branch for fully convolutional networks. This includes unmerged PRs and no guarantees.

Everything here is subject to change, including the history of this branch.

Consider PR #2016 for reducing memory usage.

See `future.sh` for details.
END
git add README.md
git add future.sh
git commit -m 'add README + creation script'
