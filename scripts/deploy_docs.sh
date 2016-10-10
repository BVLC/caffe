#!/bin/bash
# 
# All modification made by Intel Corporation: © 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Publish documentation to the gh-pages site.

# The remote for pushing the docs (defaults to origin).
# This is where you will submit the PR to BVLC:gh-pages from.
REMOTE=${1:-origin}

echo "Generating docs and pushing to $REMOTE:gh-pages..."
echo "To build and view docs when not on master, simply do 'jekyll serve -s docs'."
echo

REMOTE_URL=`git config --get remote.${REMOTE}.url`
BRANCH=`git rev-parse --abbrev-ref HEAD`
MSG=`git log --oneline -1`

if [[ $BRANCH = 'master' ]]; then
    # Find the docs dir, no matter where the script is called
    DIR="$( cd "$(dirname "$0")" ; pwd -P )"
    DOCS_SITE_DIR=$DIR/../docs/_site

    # Make sure that docs/_site tracks remote:gh-pages.
    # If not, then we make a new repo and check out just that branch.
    mkdir -p $DOCS_SITE_DIR
    cd $DOCS_SITE_DIR
    SITE_REMOTE_URL=`git config --get remote.${REMOTE}.url`
    SITE_BRANCH=`git rev-parse --abbrev-ref HEAD`

    echo $SITE_REMOTE_URL
    echo $SITE_BRANCH
    echo `pwd`

    if [[ ( $SITE_REMOTE_URL = $REMOTE_URL ) && ( $SITE_BRANCH = 'gh-pages' ) ]]; then
        echo "Confirmed that docs/_site has same remote as main repo, and is on gh-pages."
    else
        echo "Checking out $REMOTE:gh-pages into docs/_site (will take a little time)."
        git init .
        git remote add -t gh-pages -f $REMOTE $REMOTE_URL
        git checkout gh-pages
    fi

    echo "Building the site into docs/_site, and committing the changes."
    jekyll build -s .. -d .
    git add --all .
    git commit -m "$MSG"
    git push $REMOTE gh-pages

    echo "All done!"
    cd ../..
else echo "You must run this deployment script from the 'master' branch."
fi
