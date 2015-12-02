%.js: %.ometajs
	ometajs2js < $< | uglifyjs -b > $@.tmp
	mv $@.tmp $@

all: index.js
