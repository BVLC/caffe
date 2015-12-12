
all: install build test

install:
	npm install

build:
	node ./node_modules/.bin/jison lex.y lex.l
	mv lex.js lex-parser.js

test:
	node tests/all-tests.js




clean:

superclean: clean
	-find . -type d -name 'node_modules' -exec rm -rf "{}" \;


