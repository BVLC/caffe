#!/bin/sh

if [ -z "`which github-changes`" ]; then
  echo "First, do: [sudo] npm install -g github-changes"
  exit 1
fi

if [ -d .git/refs/remotes/upstream ]; then
  remote=upstream
else
  remote=origin
fi

# Increment v2.x.y -> v2.x+1.0
npm version minor || exit 1

# Generate changelog from pull requests
github-changes -o request -r request \
  --auth --verbose \
  --file CHANGELOG.md \
  --only-pulls --use-commit-body \
  || exit 1

# This may fail if no changelog updates
# TODO: would this ever actually happen?  handle it better?
git add CHANGELOG.md; git commit -m 'Update changelog'

# Publish the new version to npm
npm publish || exit 1

# Increment v2.x.0 -> v2.x.1
# For rationale, see:
# https://github.com/request/oauth-sign/issues/10#issuecomment-58917018
npm version patch || exit 1

# Push back to the main repo
git push $remote master --tags || exit 1
