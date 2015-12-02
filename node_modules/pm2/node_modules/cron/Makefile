TESTS = tests/*.js

all: test

test:
	npm install .
	@./node_modules/nodeunit/bin/nodeunit \
		$(TESTS)

.PHONY: test
