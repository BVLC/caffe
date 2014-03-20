#!/usr/bin/env sh
# Publish/ Pull-request documentation to the gh-pages site.

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
