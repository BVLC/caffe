echo "The remote from which you will submit the PR to BVLC:gh-pages must be called 'origin'"
echo "To build and view docs when not on master, simply do 'jekyll serve -s docs'."
echo

ORIGIN=`git config --get remote.origin.url`
BRANCH=`git rev-parse --abbrev-ref HEAD`
MSG=`git log --oneline -1`

if [ $BRANCH='master' ]; then
    # Make sure that docs/_site tracks remote:gh-pages.
    # If not, then we make a new repo and check out just that branch.
    mkdir docs/_site
    cd docs/_site
    SITE_ORIGIN=`git config --get remote.origin.url`
    SITE_BRANCH=`git rev-parse --abbrev-ref HEAD`

    if [ $SITE_ORIGIN=$ORIGIN ] && [ $SITE_BRANCH='gh-pages' ]; then
        echo "Confirmed that docs/_site has same origin as main repo, and is on gh-pages."
    else
        echo "Checking out origin:gh-pages into docs/_site."
        git init
        git remote add -t gh-pages -f origin $ORIGIN
        git co gh-pages
    fi

    echo "Building the site into docs/_site, and committing the changes."
    jekyll build -s .. -d .
    git add --all .
    git commit -m "$MSG"
    git push origin gh-pages

    echo "All done!"
    cd ../..
else echo "You must run this deployment script from the 'master' branch."
fi
