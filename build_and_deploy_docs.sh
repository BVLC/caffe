echo "This script, if run from master branch, will build docs website and push it to origin/gh-pages. Make PR from there."

CUR_BRANCH=`git rev-parse --abbrev-ref HEAD`
MSG=`git log --oneline -1`

if [ $CUR_BRANCH='master' ]; then
    jekyll build -s docs
    git checkout gh-pages
    git rm -qr .
    cp -r _site/. .
    rm -r _site
    git add -A
    git commit -m "$MSG"
    git push origin gh-pages
    git checkout master
fi
