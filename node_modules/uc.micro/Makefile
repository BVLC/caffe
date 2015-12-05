NPM_PACKAGE := $(shell node -e 'process.stdout.write(require("./package.json").name)')
NPM_VERSION := $(shell node -e 'process.stdout.write(require("./package.json").version)')

REMOTE_NAME ?= origin
REMOTE_REPO ?= $(shell git config --get remote.${REMOTE_NAME}.url)

CURR_HEAD   := $(firstword $(shell git show-ref --hash HEAD | cut -b -6) master)
GITHUB_PROJ := https://github.com//markdown-it/${NPM_PACKAGE}


refresh:
	rm -rf properties
	rm -rf categories

	mkdir -p properties/Any
	cp node_modules/unicode-7.0.0/properties/Any/regex.js properties/Any/
	mkdir -p categories/Cc
	cp node_modules/unicode-7.0.0/categories/Cc/regex.js categories/Cc/
	mkdir -p categories/Cf
	cp node_modules/unicode-7.0.0/categories/Cf/regex.js categories/Cf/
	mkdir -p categories/Z
	cp node_modules/unicode-7.0.0/categories/Z/regex.js categories/Z/
	mkdir -p categories/P
	cp node_modules/unicode-7.0.0/categories/P/regex.js categories/P/


publish:
	@if test 0 -ne `git status --porcelain | wc -l` ; then \
		echo "Unclean working tree. Commit or stash changes first." >&2 ; \
		exit 128 ; \
		fi
	@if test 0 -ne `git fetch ; git status | grep '^# Your branch' | wc -l` ; then \
		echo "Local/Remote history differs. Please push/pull changes." >&2 ; \
		exit 128 ; \
		fi
	@if test 0 -ne `git tag -l ${NPM_VERSION} | wc -l` ; then \
		echo "Tag ${NPM_VERSION} exists. Update package.json" >&2 ; \
		exit 128 ; \
		fi
	git tag ${NPM_VERSION} && git push origin ${NPM_VERSION}
	npm publish ${GITHUB_PROJ}/tarball/${NPM_VERSION}

.PHONY: publish
#.SILENT: help lint test todo
