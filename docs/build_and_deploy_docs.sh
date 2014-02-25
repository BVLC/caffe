echo "This script builds docs/ and deploys the site to origin/gh-pages."
echo "It must be run from master branch, or nothing will happen."
echo ""
echo "COMMIT YOUR WORK BEFORE RUNNING THIS."
echo "This will delete *all* uncomitted files."
read -p "Have you committed (y/n)? " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

CUR_BRANCH=`git rev-parse --abbrev-ref HEAD`
MSG=`git log --oneline -1`

if [ $CUR_BRANCH='master' ]; then
    jekyll build -s docs
    git checkout gh-pages

    # Need to make sure that gh-pages is a valid branch!
    CUR_BRANCH=`git rev-parse --abbrev-ref HEAD`
    if [ $CUR_BRANCH='gh-pages' ]; then
        git rm -qr .
        cp -r _site/. .
        rm -r _site
        git add -A
        git commit -m "$MSG"
        git push origin gh-pages
        git checkout master
    fi
fi
