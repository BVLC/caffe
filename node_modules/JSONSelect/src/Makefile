BUILD_DIR = build

PREFIX = .
SRC_DIR = ${PREFIX}
DIST_DIR = ${PREFIX}/dist


JS_ENGINE ?= `which node nodejs 2> /dev/null`
COMPILER = ${JS_ENGINE} ${BUILD_DIR}/uglify.js --unsafe
POST_COMPILER = ${JS_ENGINE} ${BUILD_DIR}/post-compile.js

SRC = ${SRC_DIR}/jsonselect.js
DIST = ${DIST_DIR}/jsonselect.js
DIST_MIN = ${DIST_DIR}/jsonselect.min.js

all: hint project min tests
	@@echo "Project build complete."

${DIST_DIR}:
	@@mkdir -p ${DIST_DIR}

project: ${DIST}

${DIST}: ${SRC} | ${DIST_DIR}
	@@echo "Building" ${DIST}
	@@echo ${SRC}
	@@cat ${SRC} > ${DIST};


min: project ${DIST_MIN}

${DIST_MIN}: ${DIST}
	@@if test ! -z ${JS_ENGINE}; then \
		echo "Minifying Project" ${DIST_MIN}; \
		${COMPILER} ${DIST} > ${DIST_MIN}.tmp; \
		${POST_COMPILER} ${DIST_MIN}.tmp > ${DIST_MIN}; \
		rm -f ${DIST_MIN}.tmp; \
	else \
		echo "You must have NodeJS installed in order to minify Project."; \
	fi


hint:
	@@if test ! -z ${JS_ENGINE}; then \
		echo "Hinting Project"; \
		${JS_ENGINE} build/jshint-check.js; \
	else \
		echo "Nodejs is missing"; \
	fi


test/tests/README.md: 
	@@cd .. && git submodule init 
	@@cd .. && git submodule update 


tests: test/tests/README.md
	@@if test ! -z ${JS_ENGINE}; then \
		echo "Testing Project"; \
		${JS_ENGINE} test/run.js; \
	else \
		echo "nodejs is missing"; \
	fi


clean:
	@@echo "Removing Distribution directory:" ${DIST_DIR}
	@@rm -rf ${DIST_DIR}




.PHONY: all project hint min tests
