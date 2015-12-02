SUFFIX=.ometajs

all: lib/ometajs/grammars/bsjs.js

lib/ometajs/grammars/%.js: lib/ometajs/grammars/%.ometajs
	./bin/ometajs2js -b --root "../../../" -i $< -o $@

test:
	npm test

docs:
	docco lib/ometajs/*.js

.PHONY: all docs
