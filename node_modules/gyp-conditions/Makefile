%.js: %.ometajs
	set -o pipefail ; ometajs2js < $< | uglifyjs -b > $@.tmp
	mv $@.tmp $@

all: index.js
