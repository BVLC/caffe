<<<<<<< HEAD
srcdir = /Users/stevenjames/.phpbrew/build/php-5.5.30
builddir = /Users/stevenjames/Documents/caffe
top_srcdir = /Users/stevenjames/.phpbrew/build/php-5.5.30
top_builddir = /Users/stevenjames/Documents/caffe
EGREP = /usr/local/bin/ggrep -E
SED = /usr/local/bin/gsed
CONFIGURE_COMMAND = '/Users/stevenjames/.phpbrew/build/php-5.5.30/configure'
CONFIGURE_OPTIONS =
PHP_MAJOR_VERSION = 5
PHP_MINOR_VERSION = 5
PHP_RELEASE_VERSION = 30
PHP_EXTRA_VERSION =
AWK = gawk
YACC = exit 0;
RE2C = re2c
RE2C_FLAGS =
SHLIB_SUFFIX_NAME = dylib
SHLIB_DL_SUFFIX_NAME = so
PHP_CLI_OBJS = sapi/cli/php_cli.lo sapi/cli/php_http_parser.lo sapi/cli/php_cli_server.lo sapi/cli/ps_title.lo sapi/cli/php_cli_process_title.lo
PHP_EXECUTABLE = $(top_builddir)/$(SAPI_CLI_PATH)
SAPI_CLI_PATH = sapi/cli/php
BUILD_CLI = $(CC) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) $(EXTRA_LDFLAGS_PROGRAM) $(LDFLAGS) $(NATIVE_RPATHS) $(PHP_GLOBAL_OBJS:.lo=.o) $(PHP_BINARY_OBJS:.lo=.o) $(PHP_CLI_OBJS:.lo=.o) $(PHP_FRAMEWORKS) $(EXTRA_LIBS) $(ZEND_EXTRA_LIBS) -o $(SAPI_CLI_PATH)
PHP_CGI_OBJS = sapi/cgi/cgi_main.lo sapi/cgi/fastcgi.lo
SAPI_CGI_PATH = sapi/cgi/php-cgi
BUILD_CGI = $(CC) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) $(EXTRA_LDFLAGS_PROGRAM) $(LDFLAGS) $(NATIVE_RPATHS) $(PHP_GLOBAL_OBJS:.lo=.o) $(PHP_BINARY_OBJS:.lo=.o) $(PHP_CGI_OBJS:.lo=.o) $(PHP_FRAMEWORKS) $(EXTRA_LIBS) $(ZEND_EXTRA_LIBS) -o $(SAPI_CGI_PATH)
PROG_SENDMAIL = /usr/sbin/sendmail
SQLITE3_SHARED_LIBADD =
DOM_SHARED_LIBADD =
FILTER_SHARED_LIBADD =
ICONV_SHARED_LIBADD =
JSON_SHARED_LIBADD =
shared_objects_opcache = ext/opcache/ZendAccelerator.lo ext/opcache/zend_accelerator_blacklist.lo ext/opcache/zend_accelerator_debug.lo ext/opcache/zend_accelerator_hash.lo ext/opcache/zend_accelerator_module.lo ext/opcache/zend_persist.lo ext/opcache/zend_persist_calc.lo ext/opcache/zend_shared_alloc.lo ext/opcache/zend_accelerator_util_funcs.lo ext/opcache/shared_alloc_shm.lo ext/opcache/shared_alloc_mmap.lo ext/opcache/shared_alloc_posix.lo ext/opcache/Optimizer/zend_optimizer.lo
PDO_SQLITE_SHARED_LIBADD =
SESSION_SHARED_LIBADD =
SIMPLEXML_SHARED_LIBADD =
XML_SHARED_LIBADD =
XMLREADER_SHARED_LIBADD =
XMLWRITER_SHARED_LIBADD =
PHP_INSTALLED_SAPIS = cli cgi
PHP_EXECUTABLE = $(top_builddir)/$(SAPI_CLI_PATH)
PHP_SAPI_OBJS = main/internal_functions.lo
PHP_BINARY_OBJS = main/internal_functions_cli.lo
PHP_GLOBAL_OBJS = ext/date/php_date.lo ext/date/lib/astro.lo ext/date/lib/dow.lo ext/date/lib/parse_date.lo ext/date/lib/parse_tz.lo ext/date/lib/timelib.lo ext/date/lib/tm2unixtime.lo ext/date/lib/unixtime2tm.lo ext/date/lib/parse_iso_intervals.lo ext/date/lib/interval.lo ext/ereg/ereg.lo ext/ereg/regex/regcomp.lo ext/ereg/regex/regexec.lo ext/ereg/regex/regerror.lo ext/ereg/regex/regfree.lo ext/libxml/libxml.lo ext/pcre/pcrelib/pcre_chartables.lo ext/pcre/pcrelib/pcre_ucd.lo ext/pcre/pcrelib/pcre_compile.lo ext/pcre/pcrelib/pcre_config.lo ext/pcre/pcrelib/pcre_exec.lo ext/pcre/pcrelib/pcre_fullinfo.lo ext/pcre/pcrelib/pcre_get.lo ext/pcre/pcrelib/pcre_globals.lo ext/pcre/pcrelib/pcre_maketables.lo ext/pcre/pcrelib/pcre_newline.lo ext/pcre/pcrelib/pcre_ord2utf8.lo ext/pcre/pcrelib/pcre_refcount.lo ext/pcre/pcrelib/pcre_study.lo ext/pcre/pcrelib/pcre_tables.lo ext/pcre/pcrelib/pcre_valid_utf8.lo ext/pcre/pcrelib/pcre_version.lo ext/pcre/pcrelib/pcre_xclass.lo ext/pcre/pcrelib/pcre_jit_compile.lo ext/pcre/php_pcre.lo ext/sqlite3/sqlite3.lo ext/sqlite3/libsqlite/sqlite3.lo ext/ctype/ctype.lo ext/dom/php_dom.lo ext/dom/attr.lo ext/dom/document.lo ext/dom/domerrorhandler.lo ext/dom/domstringlist.lo ext/dom/domexception.lo ext/dom/namelist.lo ext/dom/processinginstruction.lo ext/dom/cdatasection.lo ext/dom/documentfragment.lo ext/dom/domimplementation.lo ext/dom/element.lo ext/dom/node.lo ext/dom/string_extend.lo ext/dom/characterdata.lo ext/dom/documenttype.lo ext/dom/domimplementationlist.lo ext/dom/entity.lo ext/dom/nodelist.lo ext/dom/text.lo ext/dom/comment.lo ext/dom/domconfiguration.lo ext/dom/domimplementationsource.lo ext/dom/entityreference.lo ext/dom/notation.lo ext/dom/xpath.lo ext/dom/dom_iterators.lo ext/dom/typeinfo.lo ext/dom/domerror.lo ext/dom/domlocator.lo ext/dom/namednodemap.lo ext/dom/userdatahandler.lo ext/fileinfo/fileinfo.lo ext/fileinfo/libmagic/apprentice.lo ext/fileinfo/libmagic/apptype.lo ext/fileinfo/libmagic/ascmagic.lo ext/fileinfo/libmagic/cdf.lo ext/fileinfo/libmagic/cdf_time.lo ext/fileinfo/libmagic/compress.lo ext/fileinfo/libmagic/encoding.lo ext/fileinfo/libmagic/fsmagic.lo ext/fileinfo/libmagic/funcs.lo ext/fileinfo/libmagic/is_tar.lo ext/fileinfo/libmagic/magic.lo ext/fileinfo/libmagic/print.lo ext/fileinfo/libmagic/readcdf.lo ext/fileinfo/libmagic/softmagic.lo ext/filter/filter.lo ext/filter/sanitizing_filters.lo ext/filter/logical_filters.lo ext/filter/callback_filter.lo ext/hash/hash.lo ext/hash/hash_md.lo ext/hash/hash_sha.lo ext/hash/hash_ripemd.lo ext/hash/hash_haval.lo ext/hash/hash_tiger.lo ext/hash/hash_gost.lo ext/hash/hash_snefru.lo ext/hash/hash_whirlpool.lo ext/hash/hash_adler32.lo ext/hash/hash_crc32.lo ext/hash/hash_fnv.lo ext/hash/hash_joaat.lo ext/iconv/iconv.lo ext/json/json.lo ext/json/utf8_decode.lo ext/json/JSON_parser.lo ext/pdo/pdo.lo ext/pdo/pdo_dbh.lo ext/pdo/pdo_stmt.lo ext/pdo/pdo_sql_parser.lo ext/pdo/pdo_sqlstate.lo ext/pdo_sqlite/pdo_sqlite.lo ext/pdo_sqlite/sqlite_driver.lo ext/pdo_sqlite/sqlite_statement.lo ext/phar/util.lo ext/phar/tar.lo ext/phar/zip.lo ext/phar/stream.lo ext/phar/func_interceptors.lo ext/phar/dirstream.lo ext/phar/phar.lo ext/phar/phar_object.lo ext/phar/phar_path_check.lo ext/posix/posix.lo ext/reflection/php_reflection.lo ext/session/mod_user_class.lo ext/session/session.lo ext/session/mod_files.lo ext/session/mod_mm.lo ext/session/mod_user.lo ext/simplexml/simplexml.lo ext/simplexml/sxe.lo ext/spl/php_spl.lo ext/spl/spl_functions.lo ext/spl/spl_engine.lo ext/spl/spl_iterators.lo ext/spl/spl_array.lo ext/spl/spl_directory.lo ext/spl/spl_exceptions.lo ext/spl/spl_observer.lo ext/spl/spl_dllist.lo ext/spl/spl_heap.lo ext/spl/spl_fixedarray.lo ext/standard/crypt_freesec.lo ext/standard/crypt_blowfish.lo ext/standard/crypt_sha512.lo ext/standard/crypt_sha256.lo ext/standard/php_crypt_r.lo ext/standard/array.lo ext/standard/base64.lo ext/standard/basic_functions.lo ext/standard/browscap.lo ext/standard/crc32.lo ext/standard/crypt.lo ext/standard/cyr_convert.lo ext/standard/datetime.lo ext/standard/dir.lo ext/standard/dl.lo ext/standard/dns.lo ext/standard/exec.lo ext/standard/file.lo ext/standard/filestat.lo ext/standard/flock_compat.lo ext/standard/formatted_print.lo ext/standard/fsock.lo ext/standard/head.lo ext/standard/html.lo ext/standard/image.lo ext/standard/info.lo ext/standard/iptc.lo ext/standard/lcg.lo ext/standard/link.lo ext/standard/mail.lo ext/standard/math.lo ext/standard/md5.lo ext/standard/metaphone.lo ext/standard/microtime.lo ext/standard/pack.lo ext/standard/pageinfo.lo ext/standard/quot_print.lo ext/standard/rand.lo ext/standard/soundex.lo ext/standard/string.lo ext/standard/scanf.lo ext/standard/syslog.lo ext/standard/type.lo ext/standard/uniqid.lo ext/standard/url.lo ext/standard/var.lo ext/standard/versioning.lo ext/standard/assert.lo ext/standard/strnatcmp.lo ext/standard/levenshtein.lo ext/standard/incomplete_class.lo ext/standard/url_scanner_ex.lo ext/standard/ftp_fopen_wrapper.lo ext/standard/http_fopen_wrapper.lo ext/standard/php_fopen_wrapper.lo ext/standard/credits.lo ext/standard/css.lo ext/standard/var_unserializer.lo ext/standard/ftok.lo ext/standard/sha1.lo ext/standard/user_filters.lo ext/standard/uuencode.lo ext/standard/filters.lo ext/standard/proc_open.lo ext/standard/streamsfuncs.lo ext/standard/http.lo ext/standard/password.lo ext/tokenizer/tokenizer.lo ext/tokenizer/tokenizer_data.lo ext/xml/xml.lo ext/xml/compat.lo ext/xmlreader/php_xmlreader.lo ext/xmlwriter/php_xmlwriter.lo TSRM/TSRM.lo TSRM/tsrm_strtok_r.lo TSRM/tsrm_virtual_cwd.lo main/main.lo main/snprintf.lo main/spprintf.lo main/php_sprintf.lo main/fopen_wrappers.lo main/alloca.lo main/php_scandir.lo main/php_ini.lo main/SAPI.lo main/rfc1867.lo main/php_content_types.lo main/strlcpy.lo main/strlcat.lo main/mergesort.lo main/reentrancy.lo main/php_variables.lo main/php_ticks.lo main/network.lo main/php_open_temporary_file.lo main/output.lo main/getopt.lo main/streams/streams.lo main/streams/cast.lo main/streams/memory.lo main/streams/filter.lo main/streams/plain_wrapper.lo main/streams/userspace.lo main/streams/transports.lo main/streams/xp_socket.lo main/streams/mmap.lo main/streams/glob_wrapper.lo Zend/zend_language_parser.lo Zend/zend_language_scanner.lo Zend/zend_ini_parser.lo Zend/zend_ini_scanner.lo Zend/zend_alloc.lo Zend/zend_compile.lo Zend/zend_constants.lo Zend/zend_dynamic_array.lo Zend/zend_dtrace.lo Zend/zend_execute_API.lo Zend/zend_highlight.lo Zend/zend_llist.lo Zend/zend_vm_opcodes.lo Zend/zend_opcode.lo Zend/zend_operators.lo Zend/zend_ptr_stack.lo Zend/zend_stack.lo Zend/zend_variables.lo Zend/zend.lo Zend/zend_API.lo Zend/zend_extensions.lo Zend/zend_hash.lo Zend/zend_list.lo Zend/zend_indent.lo Zend/zend_builtin_functions.lo Zend/zend_sprintf.lo Zend/zend_ini.lo Zend/zend_qsort.lo Zend/zend_multibyte.lo Zend/zend_ts_hash.lo Zend/zend_stream.lo Zend/zend_iterators.lo Zend/zend_interfaces.lo Zend/zend_exceptions.lo Zend/zend_strtod.lo Zend/zend_gc.lo Zend/zend_closures.lo Zend/zend_float.lo Zend/zend_string.lo Zend/zend_signal.lo Zend/zend_generators.lo Zend/zend_objects.lo Zend/zend_object_handlers.lo Zend/zend_objects_API.lo Zend/zend_default_classes.lo Zend/zend_execute.lo
PHP_BINARIES = cli cgi
PHP_MODULES =
PHP_ZEND_EX = $(phplibdir)/opcache.la
EXT_LIBS =
abs_builddir = /Users/stevenjames/Documents/caffe
abs_srcdir = /Users/stevenjames/.phpbrew/build/php-5.5.30
php_abs_top_builddir = /Users/stevenjames/Documents/caffe
php_abs_top_srcdir = /Users/stevenjames/.phpbrew/build/php-5.5.30
bindir = ${exec_prefix}/bin
sbindir = ${exec_prefix}/sbin
exec_prefix = ${prefix}
program_prefix =
program_suffix =
includedir = ${prefix}/include
libdir = ${exec_prefix}/lib/php
mandir = ${datarootdir}/man
phplibdir = /Users/stevenjames/Documents/caffe/modules
phptempdir = /Users/stevenjames/Documents/caffe/libs
prefix = /usr/local
localstatedir = ${prefix}/var
datadir = ${datarootdir}/php
datarootdir = /usr/local/php
sysconfdir = ${prefix}/etc
EXEEXT =
CC = cc
CFLAGS = $(CFLAGS_CLEAN) -prefer-non-pic -static
CFLAGS_CLEAN = -I/usr/include -g -O2 -fvisibility=hidden
CPP = cc -E
CPPFLAGS = -no-cpp-precomp
CXX =
CXXFLAGS = -prefer-non-pic -static
CXXFLAGS_CLEAN =
DEBUG_CFLAGS =
EXTENSION_DIR = /usr/local/lib/php/extensions/no-debug-non-zts-20121212
EXTRA_LDFLAGS = -L/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/lib
EXTRA_LDFLAGS_PROGRAM = -L/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/lib
EXTRA_LIBS = -lresolv -liconv -liconv -lm -lxml2 -lz -licucore -lm -lxml2 -lz -licucore -lm -lxml2 -lz -licucore -lm -lxml2 -lz -licucore -lm -lxml2 -lz -licucore -lm -lxml2 -lz -licucore -lm
ZEND_EXTRA_LIBS =
INCLUDES = -I/Users/stevenjames/Documents/caffe/ext/date/lib -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/regex -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include/libxml2 -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/sqlite3/libsqlite -I$(top_builddir)/TSRM -I$(top_builddir)/Zend -I$(top_srcdir)/main -I$(top_srcdir)/Zend -I$(top_srcdir)/TSRM -I$(top_builddir)/
EXTRA_INCLUDES =
INCLUDE_PATH = .:/usr/local/lib/php
INSTALL_IT =
LFLAGS =
LIBTOOL = $(SHELL) $(top_builddir)/libtool --silent --preserve-dup-deps
LN_S = ln -s
NATIVE_RPATHS = -Wl,-rpath,/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/lib
PEAR_INSTALLDIR = ${exec_prefix}/lib/php
PHP_BUILD_DATE = 2015-12-05
PHP_LDFLAGS = -L/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/lib
PHP_LIBS =
OVERALL_TARGET =
PHP_RPATHS = -R /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/lib
PHP_SAPI = none
PHP_VERSION = 5.5.30
PHP_VERSION_ID = 50530
SHELL = /bin/sh
SHARED_LIBTOOL = $(LIBTOOL)
WARNING_LEVEL =
PHP_FRAMEWORKS =
PHP_FRAMEWORKPATH =
INSTALL_HEADERS = sapi/cli/cli.h ext/date/php_date.h ext/date/lib/timelib.h ext/date/lib/timelib_structs.h ext/date/lib/timelib_config.h ext/ereg/php_ereg.h ext/ereg/php_regex.h ext/ereg/regex/ ext/libxml/php_libxml.h ext/pcre/php_pcre.h ext/pcre/pcrelib/ ext/sqlite3/libsqlite/sqlite3.h ext/dom/xml_common.h ext/filter/php_filter.h ext/hash/php_hash.h ext/hash/php_hash_md.h ext/hash/php_hash_sha.h ext/hash/php_hash_ripemd.h ext/hash/php_hash_haval.h ext/hash/php_hash_tiger.h ext/hash/php_hash_gost.h ext/hash/php_hash_snefru.h ext/hash/php_hash_whirlpool.h ext/hash/php_hash_adler32.h ext/hash/php_hash_crc32.h ext/hash/php_hash_fnv.h ext/hash/php_hash_joaat.h ext/hash/php_hash_types.h ext/iconv/ ext/json/php_json.h ext/pdo/php_pdo.h ext/pdo/php_pdo_driver.h ext/phar/php_phar.h ext/session/php_session.h ext/session/mod_files.h ext/session/mod_user.h ext/spl/php_spl.h ext/spl/spl_array.h ext/spl/spl_directory.h ext/spl/spl_engine.h ext/spl/spl_exceptions.h ext/spl/spl_functions.h ext/spl/spl_iterators.h ext/spl/spl_observer.h ext/spl/spl_dllist.h ext/spl/spl_heap.h ext/spl/spl_fixedarray.h ext/standard/ ext/xml/ Zend/ TSRM/ include/ main/ main/streams/
ZEND_EXT_TYPE = zend_extension
all_targets = $(OVERALL_TARGET) $(PHP_MODULES) $(PHP_ZEND_EX) $(PHP_BINARIES) pharcmd
install_targets = install-modules install-binaries install-build install-headers install-programs install-pear install-pharcmd
install_binary_targets = install-cli install-cgi
mkinstalldirs = $(top_srcdir)/build/shtool mkdir -p
INSTALL = $(top_srcdir)/build/shtool install -c
INSTALL_DATA = $(INSTALL) -m 644

DEFS = -DPHP_ATOM_INC -I$(top_builddir)/include -I$(top_builddir)/main -I$(top_srcdir)
COMMON_FLAGS = $(DEFS) $(INCLUDES) $(EXTRA_INCLUDES) $(CPPFLAGS) $(PHP_FRAMEWORKPATH)

all: $(all_targets) 
	@echo
	@echo "Build complete."
	@echo "Don't forget to run 'make test'."
	@echo

build-modules: $(PHP_MODULES) $(PHP_ZEND_EX)

build-binaries: $(PHP_BINARIES)

libphp$(PHP_MAJOR_VERSION).la: $(PHP_GLOBAL_OBJS) $(PHP_SAPI_OBJS)
	$(LIBTOOL) --mode=link $(CC) $(CFLAGS) $(EXTRA_CFLAGS) -rpath $(phptempdir) $(EXTRA_LDFLAGS) $(LDFLAGS) $(PHP_RPATHS) $(PHP_GLOBAL_OBJS) $(PHP_SAPI_OBJS) $(EXTRA_LIBS) $(ZEND_EXTRA_LIBS) -o $@
	-@$(LIBTOOL) --silent --mode=install cp $@ $(phptempdir)/$@ >/dev/null 2>&1

libs/libphp$(PHP_MAJOR_VERSION).bundle: $(PHP_GLOBAL_OBJS) $(PHP_SAPI_OBJS)
	$(CC) $(MH_BUNDLE_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) $(LDFLAGS) $(EXTRA_LDFLAGS) $(PHP_GLOBAL_OBJS:.lo=.o) $(PHP_SAPI_OBJS:.lo=.o) $(PHP_FRAMEWORKS) $(EXTRA_LIBS) $(ZEND_EXTRA_LIBS) -o $@ && cp $@ libs/libphp$(PHP_MAJOR_VERSION).so

install: $(all_targets) $(install_targets)

install-sapi: $(OVERALL_TARGET)
	@echo "Installing PHP SAPI module:       $(PHP_SAPI)"
	-@$(mkinstalldirs) $(INSTALL_ROOT)$(bindir)
	-@if test ! -r $(phptempdir)/libphp$(PHP_MAJOR_VERSION).$(SHLIB_DL_SUFFIX_NAME); then \
		for i in 0.0.0 0.0 0; do \
			if test -r $(phptempdir)/libphp$(PHP_MAJOR_VERSION).$(SHLIB_DL_SUFFIX_NAME).$$i; then \
				$(LN_S) $(phptempdir)/libphp$(PHP_MAJOR_VERSION).$(SHLIB_DL_SUFFIX_NAME).$$i $(phptempdir)/libphp$(PHP_MAJOR_VERSION).$(SHLIB_DL_SUFFIX_NAME); \
				break; \
			fi; \
		done; \
	fi
	@$(INSTALL_IT)

install-binaries: build-binaries $(install_binary_targets)

install-modules: build-modules
	@test -d modules && \
	$(mkinstalldirs) $(INSTALL_ROOT)$(EXTENSION_DIR)
	@echo "Installing shared extensions:     $(INSTALL_ROOT)$(EXTENSION_DIR)/"
	@rm -f modules/*.la >/dev/null 2>&1
	@$(INSTALL) modules/* $(INSTALL_ROOT)$(EXTENSION_DIR)

install-headers:
	-@if test "$(INSTALL_HEADERS)"; then \
		for i in `echo $(INSTALL_HEADERS)`; do \
			i=`$(top_srcdir)/build/shtool path -d $$i`; \
			paths="$$paths $(INSTALL_ROOT)$(phpincludedir)/$$i"; \
		done; \
		$(mkinstalldirs) $$paths && \
		echo "Installing header files:          $(INSTALL_ROOT)$(phpincludedir)/" && \
		for i in `echo $(INSTALL_HEADERS)`; do \
			if test "$(PHP_PECL_EXTENSION)"; then \
				src=`echo $$i | $(SED) -e "s#ext/$(PHP_PECL_EXTENSION)/##g"`; \
			else \
				src=$$i; \
			fi; \
			if test -f "$(top_srcdir)/$$src"; then \
				$(INSTALL_DATA) $(top_srcdir)/$$src $(INSTALL_ROOT)$(phpincludedir)/$$i; \
			elif test -f "$(top_builddir)/$$src"; then \
				$(INSTALL_DATA) $(top_builddir)/$$src $(INSTALL_ROOT)$(phpincludedir)/$$i; \
			else \
				(cd $(top_srcdir)/$$src && $(INSTALL_DATA) *.h $(INSTALL_ROOT)$(phpincludedir)/$$i; \
				cd $(top_builddir)/$$src && $(INSTALL_DATA) *.h $(INSTALL_ROOT)$(phpincludedir)/$$i) 2>/dev/null || true; \
			fi \
		done; \
	fi
=======
PROJECT := caffe

<<<<<<< HEAD
CONFIG_FILE := Makefile.config
# Explicitly check for the config file, otherwise make -k will proceed anyway.
ifeq ($(wildcard $(CONFIG_FILE)),)
$(error $(CONFIG_FILE) not found. See $(CONFIG_FILE).example.)
endif
include $(CONFIG_FILE)

BUILD_DIR_LINK := $(BUILD_DIR)
ifeq ($(RELEASE_BUILD_DIR),)
	RELEASE_BUILD_DIR := .$(BUILD_DIR)_release
endif
ifeq ($(DEBUG_BUILD_DIR),)
	DEBUG_BUILD_DIR := .$(BUILD_DIR)_debug
endif
<<<<<<< HEAD
=======
CONFIG_FILE ?= Makefile.config
include $(CONFIG_FILE)

BUILD_DIR_LINK := $(BUILD_DIR)
RELEASE_BUILD_DIR ?= .$(BUILD_DIR)_release
DEBUG_BUILD_DIR ?= .$(BUILD_DIR)_debug
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	BUILD_DIR := $(DEBUG_BUILD_DIR)
	OTHER_BUILD_DIR := $(RELEASE_BUILD_DIR)
else
	BUILD_DIR := $(RELEASE_BUILD_DIR)
	OTHER_BUILD_DIR := $(DEBUG_BUILD_DIR)
endif

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
# All of the directories containing code.
SRC_DIRS := $(shell find * -type d -exec bash -c "find {} -maxdepth 1 \
	\( -name '*.cpp' -o -name '*.proto' \) | grep -q ." \; -print)

<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
# The target shared library name
LIB_BUILD_DIR := $(BUILD_DIR)/lib
STATIC_NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).a
DYNAMIC_NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).so

##############################
# Get all source files
##############################
# CXX_SRCS are the source files excluding the test ones.
CXX_SRCS := $(shell find src/$(PROJECT) ! -name "test_*.cpp" -name "*.cpp")
# CU_SRCS are the cuda source files
CU_SRCS := $(shell find src/$(PROJECT) ! -name "test_*.cu" -name "*.cu")
# TEST_SRCS are the test source files
TEST_MAIN_SRC := src/$(PROJECT)/test/test_caffe_main.cpp
TEST_SRCS := $(shell find src/$(PROJECT) -name "test_*.cpp")
TEST_SRCS := $(filter-out $(TEST_MAIN_SRC), $(TEST_SRCS))
TEST_CU_SRCS := $(shell find src/$(PROJECT) -name "test_*.cu")
GTEST_SRC := src/gtest/gtest-all.cpp
# TOOL_SRCS are the source files for the tool binaries
TOOL_SRCS := $(shell find tools -name "*.cpp")
# EXAMPLE_SRCS are the source files for the example binaries
EXAMPLE_SRCS := $(shell find examples -name "*.cpp")
# BUILD_INCLUDE_DIR contains any generated header files we want to include.
BUILD_INCLUDE_DIR := $(BUILD_DIR)/src
# PROTO_SRCS are the protocol buffer definitions
PROTO_SRC_DIR := src/$(PROJECT)/proto
PROTO_SRCS := $(wildcard $(PROTO_SRC_DIR)/*.proto)
# PROTO_BUILD_DIR will contain the .cc and obj files generated from
# PROTO_SRCS; PROTO_BUILD_INCLUDE_DIR will contain the .h header files
PROTO_BUILD_DIR := $(BUILD_DIR)/$(PROTO_SRC_DIR)
PROTO_BUILD_INCLUDE_DIR := $(BUILD_INCLUDE_DIR)/$(PROJECT)/proto
# NONGEN_CXX_SRCS includes all source/header files except those generated
# automatically (e.g., by proto).
NONGEN_CXX_SRCS := $(shell find \
	src/$(PROJECT) \
	include/$(PROJECT) \
	python/$(PROJECT) \
	matlab/+$(PROJECT)/private \
	examples \
	tools \
	-name "*.cpp" -or -name "*.hpp" -or -name "*.cu" -or -name "*.cuh")
LINT_SCRIPT := scripts/cpp_lint.py
LINT_OUTPUT_DIR := $(BUILD_DIR)/.lint
LINT_EXT := lint.txt
LINT_OUTPUTS := $(addsuffix .$(LINT_EXT), $(addprefix $(LINT_OUTPUT_DIR)/, $(NONGEN_CXX_SRCS)))
EMPTY_LINT_REPORT := $(BUILD_DIR)/.$(LINT_EXT)
NONEMPTY_LINT_REPORT := $(BUILD_DIR)/$(LINT_EXT)
# PY$(PROJECT)_SRC is the python wrapper for $(PROJECT)
PY$(PROJECT)_SRC := python/$(PROJECT)/_$(PROJECT).cpp
PY$(PROJECT)_HXX_SRC := python/$(PROJECT)/_$(PROJECT).hpp
PY$(PROJECT)_SO := python/$(PROJECT)/_$(PROJECT).so
PY$(PROJECT)_HXX := include/$(PROJECT)/python_layer.hpp
# MAT$(PROJECT)_SRC is the mex entrance point of matlab package for $(PROJECT)
MAT$(PROJECT)_SRC := matlab/+$(PROJECT)/private/$(PROJECT)_.cpp
ifneq ($(MATLAB_DIR),)
	MAT_SO_EXT := $(shell $(MATLAB_DIR)/bin/mexext)
endif
MAT$(PROJECT)_SO := matlab/+$(PROJECT)/private/$(PROJECT)_.$(MAT_SO_EXT)

##############################
# Derive generated files
##############################
# The generated files for protocol buffers
PROTO_GEN_HEADER_SRCS := $(addprefix $(PROTO_BUILD_DIR)/, \
		$(notdir ${PROTO_SRCS:.proto=.pb.h}))
PROTO_GEN_HEADER := $(addprefix $(PROTO_BUILD_INCLUDE_DIR)/, \
		$(notdir ${PROTO_SRCS:.proto=.pb.h}))
PROTO_GEN_CC := $(addprefix $(BUILD_DIR)/, ${PROTO_SRCS:.proto=.pb.cc})
PY_PROTO_BUILD_DIR := python/$(PROJECT)/proto
PY_PROTO_INIT := python/$(PROJECT)/proto/__init__.py
PROTO_GEN_PY := $(foreach file,${PROTO_SRCS:.proto=_pb2.py}, \
		$(PY_PROTO_BUILD_DIR)/$(notdir $(file)))
# The objects corresponding to the source files
# These objects will be linked into the final shared library, so we
# exclude the tool, example, and test objects.
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
CU_OBJS := $(addprefix $(BUILD_DIR)/cuda/, ${CU_SRCS:.cu=.o})
PROTO_OBJS := ${PROTO_GEN_CC:.cc=.o}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
OBJ_BUILD_DIR := $(BUILD_DIR)/src/$(PROJECT)
LAYER_BUILD_DIR := $(OBJ_BUILD_DIR)/layers
UTIL_BUILD_DIR := $(OBJ_BUILD_DIR)/util
DEVICE_BUILD_DIR := $(OBJ_BUILD_DIR)/devices
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
OBJS := $(PROTO_OBJS) $(CXX_OBJS) $(CU_OBJS)
# tool, example, and test objects
TOOL_OBJS := $(addprefix $(BUILD_DIR)/, ${TOOL_SRCS:.cpp=.o})
TOOL_BUILD_DIR := $(BUILD_DIR)/tools
TEST_CXX_BUILD_DIR := $(BUILD_DIR)/src/$(PROJECT)/test
TEST_CU_BUILD_DIR := $(BUILD_DIR)/cuda/src/$(PROJECT)/test
TEST_CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${TEST_SRCS:.cpp=.o})
TEST_CU_OBJS := $(addprefix $(BUILD_DIR)/cuda/, ${TEST_CU_SRCS:.cu=.o})
TEST_OBJS := $(TEST_CXX_OBJS) $(TEST_CU_OBJS)
GTEST_OBJ := $(addprefix $(BUILD_DIR)/, ${GTEST_SRC:.cpp=.o})
EXAMPLE_OBJS := $(addprefix $(BUILD_DIR)/, ${EXAMPLE_SRCS:.cpp=.o})
# Output files for automatic dependency generation
DEPS := ${CXX_OBJS:.o=.d} ${CU_OBJS:.o=.d} ${TEST_CXX_OBJS:.o=.d} \
	${TEST_CU_OBJS:.o=.d} $(BUILD_DIR)/${MAT$(PROJECT)_SO:.$(MAT_SO_EXT)=.d}
# tool, example, and test bins
TOOL_BINS := ${TOOL_OBJS:.o=.bin}
EXAMPLE_BINS := ${EXAMPLE_OBJS:.o=.bin}
# symlinks to tool bins without the ".bin" extension
TOOL_BIN_LINKS := ${TOOL_BINS:.bin=}
# Put the test binaries in build/test for convenience.
TEST_BIN_DIR := $(BUILD_DIR)/test
TEST_CU_BINS := $(addsuffix .testbin,$(addprefix $(TEST_BIN_DIR)/, \
		$(foreach obj,$(TEST_CU_OBJS),$(basename $(notdir $(obj))))))
TEST_CXX_BINS := $(addsuffix .testbin,$(addprefix $(TEST_BIN_DIR)/, \
		$(foreach obj,$(TEST_CXX_OBJS),$(basename $(notdir $(obj))))))
TEST_BINS := $(TEST_CXX_BINS) $(TEST_CU_BINS)
<<<<<<< HEAD
<<<<<<< HEAD
# TEST_ALL_BIN is the test binary that links caffe dynamically.
=======
# TEST_ALL_BIN is the test binary that links caffe statically.
>>>>>>> origin/BVLC/parallel
=======
# TEST_ALL_BIN is the test binary that links caffe dynamically.
>>>>>>> caffe
TEST_ALL_BIN := $(TEST_BIN_DIR)/test_all.testbin
# TEST_ALL_DYNINK_BIN is the test binary that links caffe as a dynamic library.
TEST_ALL_DYNLINK_BIN := $(TEST_BIN_DIR)/test_all_dynamic_link.testbin

##############################
# Derive compiler warning dump locations
##############################
WARNS_EXT := warnings.txt
CXX_WARNS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o.$(WARNS_EXT)})
CU_WARNS := $(addprefix $(BUILD_DIR)/cuda/, ${CU_SRCS:.cu=.o.$(WARNS_EXT)})
TOOL_WARNS := $(addprefix $(BUILD_DIR)/, ${TOOL_SRCS:.cpp=.o.$(WARNS_EXT)})
EXAMPLE_WARNS := $(addprefix $(BUILD_DIR)/, ${EXAMPLE_SRCS:.cpp=.o.$(WARNS_EXT)})
TEST_WARNS := $(addprefix $(BUILD_DIR)/, ${TEST_SRCS:.cpp=.o.$(WARNS_EXT)})
TEST_CU_WARNS := $(addprefix $(BUILD_DIR)/cuda/, ${TEST_CU_SRCS:.cu=.o.$(WARNS_EXT)})
ALL_CXX_WARNS := $(CXX_WARNS) $(TOOL_WARNS) $(EXAMPLE_WARNS) $(TEST_WARNS)
ALL_CU_WARNS := $(CU_WARNS) $(TEST_CU_WARNS)
ALL_WARNS := $(ALL_CXX_WARNS) $(ALL_CU_WARNS)

EMPTY_WARN_REPORT := $(BUILD_DIR)/.$(WARNS_EXT)
NONEMPTY_WARN_REPORT := $(BUILD_DIR)/$(WARNS_EXT)

##############################
# Derive include and lib directories
##############################
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include

CUDA_LIB_DIR :=
# add <cuda>/lib64 only if it exists
ifneq ("$(wildcard $(CUDA_DIR)/lib64)","")
	CUDA_LIB_DIR += $(CUDA_DIR)/lib64
endif
CUDA_LIB_DIR += $(CUDA_DIR)/lib

INCLUDE_DIRS += $(BUILD_INCLUDE_DIR) ./src ./include
ifneq ($(CPU_ONLY), 1)
	INCLUDE_DIRS += $(CUDA_INCLUDE_DIR)
	LIBRARY_DIRS += $(CUDA_LIB_DIR)
	LIBRARIES := cudart cublas curand
endif
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe

LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5

# handle IO dependencies
USE_LEVELDB ?= 1
USE_LMDB ?= 1
USE_OPENCV ?= 1

ifeq ($(USE_LEVELDB), 1)
	LIBRARIES += leveldb snappy
endif
ifeq ($(USE_LMDB), 1)
	LIBRARIES += lmdb
endif
ifeq ($(USE_OPENCV), 1)
	LIBRARIES += opencv_core opencv_highgui opencv_imgproc 

	ifeq ($(OPENCV_VERSION), 3)
		LIBRARIES += opencv_imgcodecs
	endif
		
endif
<<<<<<< HEAD
=======
LIBRARIES += glog gflags protobuf leveldb snappy \
	lmdb boost_system hdf5_hl hdf5 m \
	opencv_core opencv_highgui opencv_imgproc
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
PYTHON_LIBRARIES := boost_python python2.7
WARNINGS := -Wall -Wno-sign-compare

##############################
# Set build directories
##############################

DISTRIBUTE_DIR ?= distribute
DISTRIBUTE_SUBDIRS := $(DISTRIBUTE_DIR)/bin $(DISTRIBUTE_DIR)/lib
DIST_ALIASES := dist
ifneq ($(strip $(DISTRIBUTE_DIR)),distribute)
		DIST_ALIASES += distribute
endif

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
ALL_BUILD_DIRS := $(sort \
		$(BUILD_DIR) $(LIB_BUILD_DIR) $(OBJ_BUILD_DIR) \
		$(LAYER_BUILD_DIR) $(UTIL_BUILD_DIR) $(DEVICE_BUILD_DIR) \
		$(TOOL_BUILD_DIR) \
		$(TEST_BUILD_DIR) $(TEST_BIN_DIR) $(GTEST_BUILD_DIR) \
		$(EXAMPLE_BUILD_DIRS) \
		$(LINT_OUTPUT_DIR) \
		$(PROTO_BUILD_DIR) $(PROTO_BUILD_INCLUDE_DIR) $(PY_PROTO_BUILD_DIR) \
		$(DISTRIBUTE_SUBDIRS))
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
ALL_BUILD_DIRS := $(sort $(BUILD_DIR) $(addprefix $(BUILD_DIR)/, $(SRC_DIRS)) \
	$(addprefix $(BUILD_DIR)/cuda/, $(SRC_DIRS)) \
	$(LIB_BUILD_DIR) $(TEST_BIN_DIR) $(PY_PROTO_BUILD_DIR) $(LINT_OUTPUT_DIR) \
	$(DISTRIBUTE_SUBDIRS) $(PROTO_BUILD_INCLUDE_DIR))

##############################
# Set directory for Doxygen-generated documentation
##############################
DOXYGEN_CONFIG_FILE ?= ./.Doxyfile
# should be the same as OUTPUT_DIRECTORY in the .Doxyfile
DOXYGEN_OUTPUT_DIR ?= ./doxygen
DOXYGEN_COMMAND ?= doxygen
# All the files that might have Doxygen documentation.
DOXYGEN_SOURCES := $(shell find \
	src/$(PROJECT) \
	include/$(PROJECT) \
	python/ \
	matlab/ \
	examples \
	tools \
	-name "*.cpp" -or -name "*.hpp" -or -name "*.cu" -or -name "*.cuh" -or \
        -name "*.py" -or -name "*.m")
DOXYGEN_SOURCES += $(DOXYGEN_CONFIG_FILE)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master

##############################
# Set directory for Doxygen-generated documentation
##############################
DOXYGEN_CONFIG_FILE ?= ./.Doxyfile
# should be the same as OUTPUT_DIRECTORY in the .Doxyfile
DOXYGEN_OUTPUT_DIR ?= ./doxygen
DOXYGEN_COMMAND ?= doxygen
# All the files that might have Doxygen documentation.
DOXYGEN_SOURCES := $(shell find \
	src/$(PROJECT) \
	include/$(PROJECT) \
	python/ \
	matlab/ \
	examples \
	tools \
	-name "*.cpp" -or -name "*.hpp" -or -name "*.cu" -or -name "*.cuh" -or \
        -name "*.py" -or -name "*.m")
DOXYGEN_SOURCES += $(DOXYGEN_CONFIG_FILE)

=======
>>>>>>> caffe

##############################
# Configure build
##############################

# Determine platform
UNAME := $(shell uname -s)
ifeq ($(UNAME), Linux)
	LINUX := 1
else ifeq ($(UNAME), Darwin)
	OSX := 1
endif

# Linux
ifeq ($(LINUX), 1)
	CXX ?= /usr/bin/g++
	GCCVERSION := $(shell $(CXX) -dumpversion | cut -f1,2 -d.)
	# older versions of gcc are too dumb to build boost with -Wuninitalized
	ifeq ($(shell echo | awk '{exit $(GCCVERSION) < 4.6;}'), 1)
		WARNINGS += -Wno-uninitialized
	endif
	# boost::thread is reasonably called boost_thread (compare OS X)
	# We will also explicitly add stdc++ to the link target.
	LIBRARIES += boost_thread stdc++
endif

# OS X:
# clang++ instead of g++
# libstdc++ for NVCC compatibility on OS X >= 10.9 with CUDA < 7.0
ifeq ($(OSX), 1)
	CXX := /usr/bin/clang++
	ifneq ($(CPU_ONLY), 1)
		CUDA_VERSION := $(shell $(CUDA_DIR)/bin/nvcc -V | grep -o 'release \d' | grep -o '\d')
		ifeq ($(shell echo | awk '{exit $(CUDA_VERSION) < 7.0;}'), 1)
			CXXFLAGS += -stdlib=libstdc++
			LINKFLAGS += -stdlib=libstdc++
		endif
		# clang throws this warning for cuda headers
		WARNINGS += -Wno-unneeded-internal-declaration
	endif
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
	# gtest needs to use its own tuple to not conflict with clang
	COMMON_FLAGS += -DGTEST_USE_OWN_TR1_TUPLE=1
	# boost::thread is called boost_thread-mt to mark multithreading on OS X
	LIBRARIES += boost_thread-mt
	# we need to explicitly ask for the rpath to be obeyed
	DYNAMIC_FLAGS := -install_name @rpath/libcaffe.so
	ORIGIN := @loader_path
else
	ORIGIN := \$$ORIGIN
<<<<<<< HEAD
=======
	# boost::thread is called boost_thread-mt to mark multithreading on OS X
	LIBRARIES += boost_thread-mt
        NVCCFLAGS += -DOSX
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
endif

# Custom compiler
ifdef CUSTOM_CXX
	CXX := $(CUSTOM_CXX)
endif

# Static linking
ifneq (,$(findstring clang++,$(CXX)))
	STATIC_LINK_COMMAND := -Wl,-force_load $(STATIC_NAME)
else ifneq (,$(findstring g++,$(CXX)))
	STATIC_LINK_COMMAND := -Wl,--whole-archive $(STATIC_NAME) -Wl,--no-whole-archive
else
<<<<<<< HEAD
<<<<<<< HEAD
  # The following line must not be indented with a tab, since we are not inside a target
  $(error Cannot static link with the $(CXX) compiler)
=======
	$(error Cannot static link with the $(CXX) compiler.)
>>>>>>> origin/BVLC/parallel
=======
  # The following line must not be indented with a tab, since we are not inside a target
  $(error Cannot static link with the $(CXX) compiler)
>>>>>>> caffe
endif

# Debugging
ifeq ($(DEBUG), 1)
<<<<<<< HEAD
	COMMON_FLAGS += -DDEBUG -g -O0
<<<<<<< HEAD
=======
	COMMON_FLAGS += -DDEBUG -g -O0 -DBOOST_NOINLINE='__attribute__ ((noinline))'
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
	NVCCFLAGS += -G
else
	COMMON_FLAGS += -DNDEBUG -O2
endif

# cuDNN acceleration configuration.
ifeq ($(USE_CUDNN), 1)
	LIBRARIES += cudnn
	COMMON_FLAGS += -DUSE_CUDNN
endif

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
# configure IO libraries
ifeq ($(USE_OPENCV), 1)
	COMMON_FLAGS += -DUSE_OPENCV
endif
ifeq ($(USE_LEVELDB), 1)
	COMMON_FLAGS += -DUSE_LEVELDB
endif
ifeq ($(USE_LMDB), 1)
	COMMON_FLAGS += -DUSE_LMDB
ifeq ($(ALLOW_LMDB_NOLOCK), 1)
	COMMON_FLAGS += -DALLOW_LMDB_NOLOCK
endif
endif

<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
# CPU-only configuration
ifeq ($(CPU_ONLY), 1)
	OBJS := $(PROTO_OBJS) $(CXX_OBJS)
	TEST_OBJS := $(TEST_CXX_OBJS)
	TEST_BINS := $(TEST_CXX_BINS)
	ALL_WARNS := $(ALL_CXX_WARNS)
	TEST_FILTER := --gtest_filter="-*GPU*"
	COMMON_FLAGS += -DCPU_ONLY
endif

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
# Python layer support
ifeq ($(WITH_PYTHON_LAYER), 1)
	COMMON_FLAGS += -DWITH_PYTHON_LAYER
	LIBRARIES += $(PYTHON_LIBRARIES)
<<<<<<< HEAD
=======
ifeq ($(RDMA), 1)
	COMMON_FLAGS += -DRDMA
	LIBRARIES += ibverbs ibumad
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
endif

# BLAS configuration (default = ATLAS)
BLAS ?= atlas
ifeq ($(BLAS), mkl)
	# MKL
	LIBRARIES += mkl_rt
	COMMON_FLAGS += -DUSE_MKL
	MKL_DIR ?= /opt/intel/mkl
	BLAS_INCLUDE ?= $(MKL_DIR)/include
	BLAS_LIB ?= $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64
else ifeq ($(BLAS), open)
	# OpenBLAS
	LIBRARIES += openblas
else
	# ATLAS
	ifeq ($(LINUX), 1)
		ifeq ($(BLAS), atlas)
			# Linux simply has cblas and atlas
			LIBRARIES += cblas atlas
		endif
	else ifeq ($(OSX), 1)
		# OS X packages atlas as the vecLib framework
		LIBRARIES += cblas
		# 10.10 has accelerate while 10.9 has veclib
		XCODE_CLT_VER := $(shell pkgutil --pkg-info=com.apple.pkg.CLTools_Executables | grep 'version' | sed 's/[^0-9]*\([0-9]\).*/\1/')
		XCODE_CLT_GEQ_6 := $(shell [ $(XCODE_CLT_VER) -gt 5 ] && echo 1)
		ifeq ($(XCODE_CLT_GEQ_6), 1)
			BLAS_INCLUDE ?= /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/
			LDFLAGS += -framework Accelerate
		else
			BLAS_INCLUDE ?= /System/Library/Frameworks/vecLib.framework/Versions/Current/Headers/
			LDFLAGS += -framework vecLib
		endif
	endif
endif
INCLUDE_DIRS += $(BLAS_INCLUDE)
LIBRARY_DIRS += $(BLAS_LIB)

LIBRARY_DIRS += $(LIB_BUILD_DIR)

<<<<<<< HEAD
<<<<<<< HEAD
# Automatic dependency generation (nvcc is handled separately)
CXXFLAGS += -MMD -MP

=======
>>>>>>> origin/BVLC/parallel
=======
# Automatic dependency generation (nvcc is handled separately)
CXXFLAGS += -MMD -MP

>>>>>>> caffe
# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)
NVCCFLAGS += -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
# mex may invoke an older gcc that is too liberal with -Wuninitalized
MATLAB_CXXFLAGS := $(CXXFLAGS) -Wno-uninitialized
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)

USE_PKG_CONFIG ?= 0
ifeq ($(USE_PKG_CONFIG), 1)
	PKG_CONFIG := $(shell pkg-config opencv --libs)
else
	PKG_CONFIG :=
endif
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(PKG_CONFIG) \
		$(foreach library,$(LIBRARIES),-l$(library))
PYTHON_LDFLAGS := $(LDFLAGS) $(foreach library,$(PYTHON_LIBRARIES),-l$(library))

# 'superclean' target recursively* deletes all files ending with an extension
# in $(SUPERCLEAN_EXTS) below.  This may be useful if you've built older
# versions of Caffe that do not place all generated files in a location known
# to the 'clean' target.
#
# 'supercleanlist' will list the files to be deleted by make superclean.
#
# * Recursive with the exception that symbolic links are never followed, per the
# default behavior of 'find'.
SUPERCLEAN_EXTS := .so .a .o .bin .testbin .pb.cc .pb.h _pb2.py .cuo

# Set the sub-targets of the 'everything' target.
EVERYTHING_TARGETS := all py$(PROJECT) test warn lint
# Only build matcaffe as part of "everything" if MATLAB_DIR is specified.
ifneq ($(MATLAB_DIR),)
	EVERYTHING_TARGETS += mat$(PROJECT)
endif

##############################
# Define build targets
##############################
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
.PHONY: all lib test clean docs linecount lint lintclean tools examples $(DIST_ALIASES) \
	py mat py$(PROJECT) mat$(PROJECT) proto runtest \
	superclean supercleanlist supercleanfiles warn everything

all: lib tools examples
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

lib: $(STATIC_NAME) $(DYNAMIC_NAME)

=======

lib: $(STATIC_NAME) $(DYNAMIC_NAME)

>>>>>>> BVLC/master
=======

lib: $(STATIC_NAME) $(DYNAMIC_NAME)

>>>>>>> master
=======

lib: $(STATIC_NAME) $(DYNAMIC_NAME)

>>>>>>> caffe
=======

lib: $(STATIC_NAME) $(DYNAMIC_NAME)

>>>>>>> master
=======

lib: $(STATIC_NAME) $(DYNAMIC_NAME)

>>>>>>> master
=======
=======
.PHONY: all test clean docs linecount lint lintclean tools examples $(DIST_ALIASES) \
	py mat py$(PROJECT) mat$(PROJECT) proto runtest \
	superclean supercleanlist supercleanfiles warn everything

all: $(STATIC_NAME) $(DYNAMIC_NAME) tools examples
>>>>>>> origin/BVLC/parallel

lib: $(STATIC_NAME) $(DYNAMIC_NAME)

>>>>>>> BVLC/master
=======

lib: $(STATIC_NAME) $(DYNAMIC_NAME)

>>>>>>> master
=======

lib: $(STATIC_NAME) $(DYNAMIC_NAME)

>>>>>>> master
=======

lib: $(STATIC_NAME) $(DYNAMIC_NAME)

>>>>>>> caffe
everything: $(EVERYTHING_TARGETS)

linecount:
	cloc --read-lang-def=$(PROJECT).cloc \
		src/$(PROJECT) include/$(PROJECT) tools examples \
		python matlab

lint: $(EMPTY_LINT_REPORT)

lintclean:
	@ $(RM) -r $(LINT_OUTPUT_DIR) $(EMPTY_LINT_REPORT) $(NONEMPTY_LINT_REPORT)

docs: $(DOXYGEN_OUTPUT_DIR)
	@ cd ./docs ; ln -sfn ../$(DOXYGEN_OUTPUT_DIR)/html doxygen

$(DOXYGEN_OUTPUT_DIR): $(DOXYGEN_CONFIG_FILE) $(DOXYGEN_SOURCES)
	$(DOXYGEN_COMMAND) $(DOXYGEN_CONFIG_FILE)

$(EMPTY_LINT_REPORT): $(LINT_OUTPUTS) | $(BUILD_DIR)
	@ cat $(LINT_OUTPUTS) > $@
	@ if [ -s "$@" ]; then \
		cat $@; \
		mv $@ $(NONEMPTY_LINT_REPORT); \
		echo "Found one or more lint errors."; \
		exit 1; \
	  fi; \
	  $(RM) $(NONEMPTY_LINT_REPORT); \
	  echo "No lint errors!";

$(LINT_OUTPUTS): $(LINT_OUTPUT_DIR)/%.lint.txt : % $(LINT_SCRIPT) | $(LINT_OUTPUT_DIR)
	@ mkdir -p $(dir $@)
	@ python $(LINT_SCRIPT) $< 2>&1 \
		| grep -v "^Done processing " \
		| grep -v "^Total errors found: 0" \
		> $@ \
		|| true

test: $(TEST_ALL_BIN) $(TEST_ALL_DYNLINK_BIN) $(TEST_BINS)

tools: $(TOOL_BINS) $(TOOL_BIN_LINKS)

examples: $(EXAMPLE_BINS)

py$(PROJECT): py

py: $(PY$(PROJECT)_SO) $(PROTO_GEN_PY)

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
$(PY$(PROJECT)_SO): $(PY$(PROJECT)_SRC) $(PY$(PROJECT)_HXX) | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@ $<
	$(Q)$(CXX) -shared -o $@ $(PY$(PROJECT)_SRC) \
		-o $@ $(LINKFLAGS) -l$(PROJECT) $(PYTHON_LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../../build/lib
<<<<<<< HEAD
=======
$(PY$(PROJECT)_SO): $(STATIC_NAME) $(PY$(PROJECT)_SRC) $(PY$(PROJECT)_HXX_SRC)
	$(CXX) -shared -o $@ $(PY$(PROJECT)_SRC) \
		$(STATIC_LINK_COMMAND) $(LINKFLAGS) $(PYTHON_LDFLAGS)
	@ echo
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe

mat$(PROJECT): mat

mat: $(MAT$(PROJECT)_SO)
>>>>>>> pod-caffe-pod.hpp-merge

PHP_TEST_SETTINGS = -d 'open_basedir=' -d 'output_buffering=0' -d 'memory_limit=-1'
PHP_TEST_SHARED_EXTENSIONS =  ` \
	if test "x$(PHP_MODULES)" != "x"; then \
		for i in $(PHP_MODULES)""; do \
			. $$i; $(top_srcdir)/build/shtool echo -n -- " -d extension=$$dlname"; \
		done; \
	fi; \
	if test "x$(PHP_ZEND_EX)" != "x"; then \
		for i in $(PHP_ZEND_EX)""; do \
			. $$i; $(top_srcdir)/build/shtool echo -n -- " -d $(ZEND_EXT_TYPE)=$(top_builddir)/modules/$$dlname"; \
		done; \
	fi`
PHP_DEPRECATED_DIRECTIVES_REGEX = '^(magic_quotes_(gpc|runtime|sybase)?|(zend_)?extension(_debug)?(_ts)?)[\t\ ]*='

test: all
	@if test ! -z "$(PHP_EXECUTABLE)" && test -x "$(PHP_EXECUTABLE)"; then \
		INI_FILE=`$(PHP_EXECUTABLE) -d 'display_errors=stderr' -r 'echo php_ini_loaded_file();' 2> /dev/null`; \
		if test "$$INI_FILE"; then \
			$(EGREP) -h -v $(PHP_DEPRECATED_DIRECTIVES_REGEX) "$$INI_FILE" > $(top_builddir)/tmp-php.ini; \
		else \
			echo > $(top_builddir)/tmp-php.ini; \
		fi; \
		INI_SCANNED_PATH=`$(PHP_EXECUTABLE) -d 'display_errors=stderr' -r '$$a = explode(",\n", trim(php_ini_scanned_files())); echo $$a[0];' 2> /dev/null`; \
		if test "$$INI_SCANNED_PATH"; then \
			INI_SCANNED_PATH=`$(top_srcdir)/build/shtool path -d $$INI_SCANNED_PATH`; \
			$(EGREP) -h -v $(PHP_DEPRECATED_DIRECTIVES_REGEX) "$$INI_SCANNED_PATH"/*.ini >> $(top_builddir)/tmp-php.ini; \
		fi; \
		TEST_PHP_EXECUTABLE=$(PHP_EXECUTABLE) \
		TEST_PHP_SRCDIR=$(top_srcdir) \
		CC="$(CC)" \
			$(PHP_EXECUTABLE) -n -c $(top_builddir)/tmp-php.ini $(PHP_TEST_SETTINGS) $(top_srcdir)/run-tests.php -n -c $(top_builddir)/tmp-php.ini -d extension_dir=$(top_builddir)/modules/ $(PHP_TEST_SHARED_EXTENSIONS) $(TESTS); \
		TEST_RESULT_EXIT_CODE=$$?; \
		rm $(top_builddir)/tmp-php.ini; \
		exit $$TEST_RESULT_EXIT_CODE; \
	else \
		echo "ERROR: Cannot run tests without CLI sapi."; \
	fi
<<<<<<< HEAD
=======
	@ echo MEX $<
	$(Q)$(MATLAB_DIR)/bin/mex $(MAT$(PROJECT)_SRC) \
			CXX="$(CXX)" \
			CXXFLAGS="\$$CXXFLAGS $(MATLAB_CXXFLAGS)" \
			CXXLIBS="\$$CXXLIBS $(STATIC_LINK_COMMAND) $(LDFLAGS)" -output $@
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
	@ if [ -f "$(PROJECT)_.d" ]; then \
		mv -f $(PROJECT)_.d $(BUILD_DIR)/${MAT$(PROJECT)_SO:.$(MAT_SO_EXT)=.d}; \
	fi

runtest: $(TEST_ALL_BIN)
	$(TOOL_BUILD_DIR)/caffe
	$(TEST_ALL_BIN) $(TEST_GPUID) --gtest_shuffle $(TEST_FILTER)
=======
	@ echo

runtest: $(TEST_ALL_BIN) $(TEST_ALL_DYNLINK_BIN)
	$(TEST_ALL_BIN) $(TEST_GPUID) --gtest_shuffle $(TEST_FILTER) && \
	$(TEST_ALL_DYNLINK_BIN) $(TEST_GPUID) --gtest_shuffle $(TEST_FILTER)
>>>>>>> origin/BVLC/parallel

pytest: py
	cd python; python -m unittest discover -s caffe/test

mattest: mat
	cd matlab; $(MATLAB_DIR)/bin/matlab -nodisplay -r 'caffe.run_tests(), exit()'

pytest: py
	cd python; python -m unittest discover -s caffe/test

mattest: mat
	cd matlab; $(MATLAB_DIR)/bin/matlab -nodisplay -r 'caffe.run_tests(), exit()'

warn: $(EMPTY_WARN_REPORT)

$(EMPTY_WARN_REPORT): $(ALL_WARNS) | $(BUILD_DIR)
	@ cat $(ALL_WARNS) > $@
	@ if [ -s "$@" ]; then \
		cat $@; \
		mv $@ $(NONEMPTY_WARN_REPORT); \
		echo "Compiler produced one or more warnings."; \
		exit 1; \
	  fi; \
	  $(RM) $(NONEMPTY_WARN_REPORT); \
	  echo "No compiler warnings!";

$(ALL_WARNS): %.o.$(WARNS_EXT) : %.o

$(BUILD_DIR_LINK): $(BUILD_DIR)/.linked

# Create a target ".linked" in this BUILD_DIR to tell Make that the "build" link
# is currently correct, then delete the one in the OTHER_BUILD_DIR in case it
# exists and $(DEBUG) is toggled later.
$(BUILD_DIR)/.linked:
	@ mkdir -p $(BUILD_DIR)
	@ $(RM) $(OTHER_BUILD_DIR)/.linked
	@ $(RM) -r $(BUILD_DIR_LINK)
	@ ln -s $(BUILD_DIR) $(BUILD_DIR_LINK)
	@ touch $@

$(ALL_BUILD_DIRS): | $(BUILD_DIR_LINK)
	@ mkdir -p $@

$(DYNAMIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
<<<<<<< HEAD
<<<<<<< HEAD
	@ echo LD -o $@
	$(Q)$(CXX) -shared -o $@ $(OBJS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo AR -o $@
	$(Q)ar rcs $@ $(OBJS)
=======
	$(CXX) -shared -o $@ $(OBJS) $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	ar rcs $@ $(OBJS)
	@ echo
>>>>>>> origin/BVLC/parallel

=======

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo AR -o $@
	$(Q)ar rcs $@ $(OBJS)

<<<<<<< HEAD
>>>>>>> master
=======

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo AR -o $@
	$(Q)ar rcs $@ $(OBJS)

>>>>>>> caffe
=======
=======
$(TEST_ALL_BIN): $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) $(STATIC_NAME) \
		| $(TEST_BIN_DIR)
	$(CXX) $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) $(STATIC_LINK_COMMAND) \
		-o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(TEST_ALL_DYNLINK_BIN): $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) $(DYNAMIC_NAME) \
		| $(TEST_BIN_DIR)
	$(CXX) $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(LIB_BUILD_DIR)
	@ echo

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_BUILD_DIR)/%.cuo $(GTEST_OBJ) $(STATIC_NAME) \
		| $(TEST_BIN_DIR)
	$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) $(STATIC_LINK_COMMAND) \
		-o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_BUILD_DIR)/%.o $(GTEST_OBJ) $(STATIC_NAME) \
		| $(TEST_BIN_DIR)
	$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) $(STATIC_LINK_COMMAND) \
		-o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo
>>>>>>> origin/BVLC/parallel

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo AR -o $@
	$(Q)ar rcs $@ $(OBJS)

<<<<<<< HEAD
>>>>>>> master
=======

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo AR -o $@
	$(Q)ar rcs $@ $(OBJS)
=======
$(TOOL_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_LINK_COMMAND) -o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(EXAMPLE_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_LINK_COMMAND) -o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo
>>>>>>> origin/BVLC/parallel

>>>>>>> master
=======

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo AR -o $@
	$(Q)ar rcs $@ $(OBJS)

>>>>>>> BVLC/master
=======

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo AR -o $@
	$(Q)ar rcs $@ $(OBJS)

>>>>>>> master
=======

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo AR -o $@
	$(Q)ar rcs $@ $(OBJS)

>>>>>>> master
=======
	@ echo LD -o $@
	$(Q)$(CXX) -shared -o $@ $(OBJS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS)

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo AR -o $@
	$(Q)ar rcs $@ $(OBJS)

>>>>>>> caffe
$(BUILD_DIR)/%.o: %.cpp | $(ALL_BUILD_DIRS)
	@ echo CXX $<
	$(Q)$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

$(DEVICE_BUILD_DIR)/%.o: src/$(PROJECT)/devices/%.cpp $(HXX_SRCS) \
		| $(DEVICE_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ echo
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe

$(PROTO_BUILD_DIR)/%.pb.o: $(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_GEN_HEADER) \
		| $(PROTO_BUILD_DIR)
	@ echo CXX $<
	$(Q)$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)

$(BUILD_DIR)/cuda/%.o: %.cu | $(ALL_BUILD_DIRS)
	@ echo NVCC $<
	$(Q)$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(Q)$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)

$(TEST_ALL_BIN): $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) \
		| $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo CXX/LD -o $@ $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

=======
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)

<<<<<<< HEAD
$(BUILD_DIR)/cuda/%.o: %.cu | $(ALL_BUILD_DIRS)
	@ echo NVCC $<
	$(Q)$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(Q)$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 2> $@.$(WARNS_EXT) \
=======
$(OBJ_BUILD_DIR)/%.cuo: src/$(PROJECT)/%.cu $(HXX_SRCS) \
		| $(LAYER_BUILD_DIR)
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(LAYER_BUILD_DIR)/%.cuo: src/$(PROJECT)/layers/%.cu $(HXX_SRCS) \
		| $(LAYER_BUILD_DIR)
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 2> $@.$(WARNS_EXT) \
>>>>>>> origin/BVLC/parallel
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)

$(TEST_ALL_BIN): $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) \
		| $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo CXX/LD -o $@ $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

>>>>>>> BVLC/master
$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CU_BUILD_DIR)/%.o \
=======

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CU_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CXX_BUILD_DIR)/%.o \
>>>>>>> BVLC/master
=======

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CU_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CXX_BUILD_DIR)/%.o \
>>>>>>> master
=======

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CU_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CXX_BUILD_DIR)/%.o \
>>>>>>> caffe
=======

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CU_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CXX_BUILD_DIR)/%.o \
>>>>>>> master
=======

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CU_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CXX_BUILD_DIR)/%.o \
>>>>>>> master
=======

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CU_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CXX_BUILD_DIR)/%.o \
>>>>>>> BVLC/master
=======

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CU_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CXX_BUILD_DIR)/%.o \
>>>>>>> master
=======

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CU_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CXX_BUILD_DIR)/%.o \
>>>>>>> master
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
$(DEVICE_BUILD_DIR)/%.cuo: src/$(PROJECT)/devices/%.cu $(HXX_SRCS) \
		| $(DEVICE_BUILD_DIR)
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ echo

$(UTIL_BUILD_DIR)/%.cuo: src/$(PROJECT)/util/%.cu | $(UTIL_BUILD_DIR)
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo
=======
=======

>>>>>>> BVLC/master
$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CXX_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib

=======

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib

>>>>>>> BVLC/master
=======

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib

>>>>>>> master
=======

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib

>>>>>>> caffe
=======

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib

>>>>>>> master
=======

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib

>>>>>>> master
=======

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib

>>>>>>> BVLC/master
=======

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib

>>>>>>> master
=======

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib

>>>>>>> master
=======

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CU_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_CXX_BUILD_DIR)/%.o \
	$(GTEST_OBJ) | $(DYNAMIC_NAME) $(TEST_BIN_DIR)
	@ echo LD $<
	$(Q)$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) \
		-o $@ $(LINKFLAGS) $(LDFLAGS) -l$(PROJECT) -Wl,-rpath,$(ORIGIN)/../lib

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib

>>>>>>> caffe
$(EXAMPLE_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../../lib

proto: $(PROTO_GEN_CC) $(PROTO_GEN_HEADER)

$(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_BUILD_DIR)/%.pb.h : \
		$(PROTO_SRC_DIR)/%.proto | $(PROTO_BUILD_DIR)
<<<<<<< HEAD
<<<<<<< HEAD
	@ echo PROTOC $<
	$(Q)protoc --proto_path=$(PROTO_SRC_DIR) --cpp_out=$(PROTO_BUILD_DIR) $<

$(PY_PROTO_BUILD_DIR)/%_pb2.py : $(PROTO_SRC_DIR)/%.proto \
		$(PY_PROTO_INIT) | $(PY_PROTO_BUILD_DIR)
	@ echo PROTOC \(python\) $<
	$(Q)protoc --proto_path=$(PROTO_SRC_DIR) --python_out=$(PY_PROTO_BUILD_DIR) $<
=======
	protoc --proto_path=$(PROTO_SRC_DIR) --cpp_out=$(PROTO_BUILD_DIR) $<
	@ echo

$(PY_PROTO_BUILD_DIR)/%_pb2.py : $(PROTO_SRC_DIR)/%.proto \
		$(PY_PROTO_INIT) | $(PY_PROTO_BUILD_DIR)
	protoc --proto_path=$(PROTO_SRC_DIR) --python_out=$(PY_PROTO_BUILD_DIR) $<
	@ echo
>>>>>>> origin/BVLC/parallel
=======
	@ echo PROTOC $<
	$(Q)protoc --proto_path=$(PROTO_SRC_DIR) --cpp_out=$(PROTO_BUILD_DIR) $<

$(PY_PROTO_BUILD_DIR)/%_pb2.py : $(PROTO_SRC_DIR)/%.proto \
		$(PY_PROTO_INIT) | $(PY_PROTO_BUILD_DIR)
	@ echo PROTOC \(python\) $<
	$(Q)protoc --proto_path=$(PROTO_SRC_DIR) --python_out=$(PY_PROTO_BUILD_DIR) $<
>>>>>>> caffe

$(PY_PROTO_INIT): | $(PY_PROTO_BUILD_DIR)
	touch $(PY_PROTO_INIT)
>>>>>>> pod-caffe-pod.hpp-merge

clean:
	find . -name \*.gcno -o -name \*.gcda | xargs rm -f
	find . -name \*.lo -o -name \*.o | xargs rm -f
	find . -name \*.la -o -name \*.a | xargs rm -f 
	find . -name \*.so | xargs rm -f
	find . -name .libs -a -type d|xargs rm -rf
	rm -f libphp$(PHP_MAJOR_VERSION).la $(SAPI_CLI_PATH) $(SAPI_CGI_PATH) $(SAPI_MILTER_PATH) $(SAPI_LITESPEED_PATH) $(SAPI_FPM_PATH) $(OVERALL_TARGET) modules/* libs/*

distclean: clean
	rm -f Makefile config.cache config.log config.status Makefile.objects Makefile.fragments libtool main/php_config.h main/internal_functions_cli.c main/internal_functions.c stamp-h sapi/apache/libphp$(PHP_MAJOR_VERSION).module sapi/apache_hooks/libphp$(PHP_MAJOR_VERSION).module buildmk.stamp Zend/zend_dtrace_gen.h Zend/zend_dtrace_gen.h.bak Zend/zend_config.h TSRM/tsrm_config.h
	rm -f php5.spec main/build-defs.h scripts/phpize
	rm -f ext/date/lib/timelib_config.h ext/mbstring/oniguruma/config.h ext/mbstring/libmbfl/config.h ext/mysqlnd/php_mysqlnd_config.h
	rm -f scripts/man1/phpize.1 scripts/php-config scripts/man1/php-config.1 sapi/cli/php.1 sapi/cgi/php-cgi.1 ext/phar/phar.1 ext/phar/phar.phar.1
	rm -f sapi/fpm/php-fpm.conf sapi/fpm/init.d.php-fpm sapi/fpm/php-fpm.service sapi/fpm/php-fpm.8 sapi/fpm/status.html
	rm -f ext/iconv/php_have_bsd_iconv.h ext/iconv/php_have_glibc_iconv.h ext/iconv/php_have_ibm_iconv.h ext/iconv/php_have_iconv.h ext/iconv/php_have_libiconv.h ext/iconv/php_iconv_aliased_libiconv.h ext/iconv/php_iconv_supports_errno.h ext/iconv/php_php_iconv_h_path.h ext/iconv/php_php_iconv_impl.h
	rm -f ext/phar/phar.phar ext/phar/phar.php
	if test "$(srcdir)" != "$(builddir)"; then \
	  rm -f ext/phar/phar/phar.inc; \
	fi
	$(EGREP) define'.*include/php' $(top_srcdir)/configure | $(SED) 's/.*>//'|xargs rm -f

.PHONY: all clean install distclean test
.NOEXPORT:
cli: $(SAPI_CLI_PATH)

$(SAPI_CLI_PATH): $(PHP_GLOBAL_OBJS) $(PHP_BINARY_OBJS) $(PHP_CLI_OBJS)
	$(BUILD_CLI)

install-cli: $(SAPI_CLI_PATH)
	@echo "Installing PHP CLI binary:        $(INSTALL_ROOT)$(bindir)/"
	@$(mkinstalldirs) $(INSTALL_ROOT)$(bindir)
	@$(INSTALL) -m 0755 $(SAPI_CLI_PATH) $(INSTALL_ROOT)$(bindir)/$(program_prefix)php$(program_suffix)$(EXEEXT)
	@echo "Installing PHP CLI man page:      $(INSTALL_ROOT)$(mandir)/man1/"
	@$(mkinstalldirs) $(INSTALL_ROOT)$(mandir)/man1
	@$(INSTALL_DATA) sapi/cli/php.1 $(INSTALL_ROOT)$(mandir)/man1/$(program_prefix)php$(program_suffix).1

cgi: $(SAPI_CGI_PATH)

$(SAPI_CGI_PATH): $(PHP_GLOBAL_OBJS) $(PHP_BINARY_OBJS) $(PHP_CGI_OBJS)
	$(BUILD_CGI)

install-cgi: $(SAPI_CGI_PATH)
	@echo "Installing PHP CGI binary:        $(INSTALL_ROOT)$(bindir)/"
	@$(mkinstalldirs) $(INSTALL_ROOT)$(bindir)
	@$(INSTALL) -m 0755 $(SAPI_CGI_PATH) $(INSTALL_ROOT)$(bindir)/$(program_prefix)php-cgi$(program_suffix)$(EXEEXT)
	@echo "Installing PHP CGI man page:      $(INSTALL_ROOT)$(mandir)/man1/"
	@$(mkinstalldirs) $(INSTALL_ROOT)$(mandir)/man1
	@$(INSTALL_DATA) sapi/cgi/php-cgi.1 $(INSTALL_ROOT)$(mandir)/man1/$(program_prefix)php-cgi$(program_suffix).1


ext/fileinfo/libmagic/apprentice.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/data_file.c
phpincludedir=$(prefix)/include/php

PDO_HEADER_FILES= \
	php_pdo.h \
	php_pdo_driver.h


/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo_sql_parser.c: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo_sql_parser.re
	(cd $(top_srcdir); $(RE2C) --no-generation-date -o ext/pdo/pdo_sql_parser.c ext/pdo/pdo_sql_parser.re)

install-pdo-headers:
	@echo "Installing PDO headers:          $(INSTALL_ROOT)$(phpincludedir)/ext/pdo/"
	@$(mkinstalldirs) $(INSTALL_ROOT)$(phpincludedir)/ext/pdo
	@for f in $(PDO_HEADER_FILES); do \
		if test -f "$(top_srcdir)/$$f"; then \
			$(INSTALL_DATA) $(top_srcdir)/$$f $(INSTALL_ROOT)$(phpincludedir)/ext/pdo; \
		elif test -f "$(top_builddir)/$$f"; then \
			$(INSTALL_DATA) $(top_builddir)/$$f $(INSTALL_ROOT)$(phpincludedir)/ext/pdo; \
		elif test -f "$(top_srcdir)/ext/pdo/$$f"; then \
			$(INSTALL_DATA) $(top_srcdir)/ext/pdo/$$f $(INSTALL_ROOT)$(phpincludedir)/ext/pdo; \
		elif test -f "$(top_builddir)/ext/pdo/$$f"; then \
			$(INSTALL_DATA) $(top_builddir)/ext/pdo/$$f $(INSTALL_ROOT)$(phpincludedir)/ext/pdo; \
		else \
			echo "hmmm"; \
		fi \
	done;

# mini hack
install: $(all_targets) $(install_targets) install-pdo-headers

/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar_path_check.c: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar_path_check.re
	@(cd $(top_srcdir); $(RE2C) --no-generation-date -b -o ext/phar/phar_path_check.c ext/phar/phar_path_check.re)

pharcmd: ext/phar/phar.php ext/phar/phar.phar

PHP_PHARCMD_SETTINGS = -d 'open_basedir=' -d 'output_buffering=0' -d 'memory_limit=-1' -d phar.readonly=0 -d 'safe_mode=0'
PHP_PHARCMD_EXECUTABLE = ` \
	if test -x "$(top_builddir)/$(SAPI_CLI_PATH)"; then \
		$(top_srcdir)/build/shtool echo -n -- "$(top_builddir)/$(SAPI_CLI_PATH) -n"; \
		if test "x$(PHP_MODULES)" != "x"; then \
		$(top_srcdir)/build/shtool echo -n -- " -d extension_dir=$(top_builddir)/modules"; \
		for i in bz2 zlib phar; do \
			if test -f "$(top_builddir)/modules/$$i.la"; then \
				. $(top_builddir)/modules/$$i.la; $(top_srcdir)/build/shtool echo -n -- " -d extension=$$dlname"; \
			fi; \
		done; \
		fi; \
	else \
		$(top_srcdir)/build/shtool echo -n -- "$(PHP_EXECUTABLE)"; \
	fi;`
PHP_PHARCMD_BANG = `$(top_srcdir)/build/shtool echo -n -- "$(INSTALL_ROOT)$(bindir)/$(program_prefix)php$(program_suffix)$(EXEEXT)";`

ext/phar/phar/phar.inc: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar/phar.inc
	-@test -d ext/phar/phar || mkdir ext/phar/phar
	-@test -f ext/phar/phar/phar.inc || cp /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar/phar.inc ext/phar/phar/phar.inc

ext/phar/phar.php: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/build_precommand.php /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar/*.inc /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar/*.php $(SAPI_CLI_PATH)
	-@echo "Generating phar.php"
	@$(PHP_PHARCMD_EXECUTABLE) $(PHP_PHARCMD_SETTINGS) /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/build_precommand.php > ext/phar/phar.php

ext/phar/phar.phar: ext/phar/phar.php ext/phar/phar/phar.inc /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar/*.inc /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar/*.php $(SAPI_CLI_PATH)
	-@echo "Generating phar.phar"
	-@rm -f ext/phar/phar.phar
	-@rm -f /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar.phar
	@$(PHP_PHARCMD_EXECUTABLE) $(PHP_PHARCMD_SETTINGS) ext/phar/phar.php pack -f ext/phar/phar.phar -a pharcommand -c auto -x \\.svn -p 0 -s /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar/phar.php -h sha1 -b "$(PHP_PHARCMD_BANG)"  /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar/
	-@chmod +x ext/phar/phar.phar

install-pharcmd: pharcmd
	-@$(mkinstalldirs) $(INSTALL_ROOT)$(bindir)
	$(INSTALL) ext/phar/phar.phar $(INSTALL_ROOT)$(bindir)
	-@rm -f $(INSTALL_ROOT)$(bindir)/phar
	$(LN_S) -f phar.phar $(INSTALL_ROOT)$(bindir)/phar
	@$(mkinstalldirs) $(INSTALL_ROOT)$(mandir)/man1
	@$(INSTALL_DATA) ext/phar/phar.1 $(INSTALL_ROOT)$(mandir)/man1/phar.1
	@$(INSTALL_DATA) ext/phar/phar.phar.1 $(INSTALL_ROOT)$(mandir)/man1/phar.phar.1


/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/var_unserializer.c: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/var_unserializer.re
	@(cd $(top_srcdir); $(RE2C) --no-generation-date -b -o ext/standard/var_unserializer.c ext/standard/var_unserializer.re)

/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/url_scanner_ex.c: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/url_scanner_ex.re
	@(cd $(top_srcdir); $(RE2C) --no-generation-date -b -o ext/standard/url_scanner_ex.c	ext/standard/url_scanner_ex.re)

ext/standard/info.lo: ext/standard/../../main/build-defs.h

ext/standard/basic_functions.lo: $(top_srcdir)/Zend/zend_language_parser.h
$(top_srcdir)/Zend/zend_language_parser.c:
$(top_srcdir)/Zend/zend_language_scanner.c:
ext/tokenizer/tokenizer.lo: $(top_srcdir)/Zend/zend_language_parser.c $(top_srcdir)/Zend/zend_language_scanner.c
# -*- makefile -*-

peardir=$(PEAR_INSTALLDIR)

# Skip all php.ini files altogether
PEAR_INSTALL_FLAGS = -n -dshort_open_tag=0 -dopen_basedir= -derror_reporting=1803 -dmemory_limit=-1 -ddetect_unicode=0

WGET = `which wget 2>/dev/null`
FETCH = `which fetch 2>/dev/null`
PEAR_PREFIX = -dp a${program_prefix}
PEAR_SUFFIX = -ds a$(program_suffix)

install-pear-installer: $(SAPI_CLI_PATH)
	@$(top_builddir)/sapi/cli/php $(PEAR_INSTALL_FLAGS) pear/install-pear-nozlib.phar -d "$(peardir)" -b "$(bindir)" ${PEAR_PREFIX} ${PEAR_SUFFIX}

install-pear:
	@echo "Installing PEAR environment:      $(INSTALL_ROOT)$(peardir)/"
	@if test ! -f pear/install-pear-nozlib.phar; then \
		if test -f /Users/stevenjames/.phpbrew/build/php-5.5.30/pear/install-pear-nozlib.phar; then \
			cp /Users/stevenjames/.phpbrew/build/php-5.5.30/pear/install-pear-nozlib.phar pear/install-pear-nozlib.phar; \
		else \
			if test ! -z "$(WGET)" && test -x "$(WGET)"; then \
				"$(WGET)" http://pear.php.net/install-pear-nozlib.phar -nd -P pear/; \
			elif test ! -z "$(FETCH)" && test -x "$(FETCH)"; then \
				"$(FETCH)" -o pear/ http://pear.php.net/install-pear-nozlib.phar; \
			else \
				$(top_builddir)/sapi/cli/php -n /Users/stevenjames/.phpbrew/build/php-5.5.30/pear/fetch.php http://pear.php.net/install-pear-nozlib.phar pear/install-pear-nozlib.phar; \
			fi \
		fi \
	fi
	@if test -f pear/install-pear-nozlib.phar && $(mkinstalldirs) $(INSTALL_ROOT)$(peardir); then \
		$(MAKE) -s install-pear-installer; \
	else \
		cat /Users/stevenjames/.phpbrew/build/php-5.5.30/pear/install-pear.txt; \
	fi

<<<<<<< HEAD

#
# Build environment install
#

phpincludedir = $(includedir)/php
phpbuilddir = $(libdir)/build

BUILD_FILES = \
	scripts/phpize.m4 \
	build/mkdep.awk \
	build/scan_makefile_in.awk \
	build/libtool.m4 \
	Makefile.global \
	acinclude.m4 \
	ltmain.sh \
	run-tests.php

BUILD_FILES_EXEC = \
	build/shtool \
	config.guess \
	config.sub

bin_SCRIPTS = phpize php-config
man_PAGES = phpize php-config

install-build:
	@echo "Installing build environment:     $(INSTALL_ROOT)$(phpbuilddir)/"
	@$(mkinstalldirs) $(INSTALL_ROOT)$(phpbuilddir) $(INSTALL_ROOT)$(bindir) && \
	(cd $(top_srcdir) && \
	$(INSTALL) $(BUILD_FILES_EXEC) $(INSTALL_ROOT)$(phpbuilddir) && \
	$(INSTALL_DATA) $(BUILD_FILES) $(INSTALL_ROOT)$(phpbuilddir))

install-programs: scripts/phpize scripts/php-config
	@echo "Installing helper programs:       $(INSTALL_ROOT)$(bindir)/"
	@$(mkinstalldirs) $(INSTALL_ROOT)$(bindir)
	@for prog in $(bin_SCRIPTS); do \
		echo "  program: $(program_prefix)$${prog}$(program_suffix)"; \
		$(INSTALL) -m 755 scripts/$${prog} $(INSTALL_ROOT)$(bindir)/$(program_prefix)$${prog}$(program_suffix); \
	done
	@echo "Installing man pages:             $(INSTALL_ROOT)$(mandir)/man1/"
	@$(mkinstalldirs) $(INSTALL_ROOT)$(mandir)/man1
	@for page in $(man_PAGES); do \
		echo "  page: $(program_prefix)$${page}$(program_suffix).1"; \
		$(INSTALL_DATA) scripts/man1/$${page}.1 $(INSTALL_ROOT)$(mandir)/man1/$(program_prefix)$${page}$(program_suffix).1; \
	done

scripts/phpize: /Users/stevenjames/.phpbrew/build/php-5.5.30/scripts/phpize.in $(top_builddir)/config.status
	(CONFIG_FILES=$@ CONFIG_HEADERS= $(top_builddir)/config.status)

scripts/php-config: /Users/stevenjames/.phpbrew/build/php-5.5.30/scripts/php-config.in $(top_builddir)/config.status
	(CONFIG_FILES=$@ CONFIG_HEADERS= $(top_builddir)/config.status)

#
# Zend
#

Zend/zend_language_scanner.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_parser.h
Zend/zend_ini_scanner.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_parser.h

/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_scanner.c: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_scanner.l
	@(cd $(top_srcdir); $(RE2C) $(RE2C_FLAGS) --no-generation-date --case-inverted -cbdFt Zend/zend_language_scanner_defs.h -oZend/zend_language_scanner.c Zend/zend_language_scanner.l)

/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_parser.h: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_parser.c
/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_parser.c: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_parser.y
	@$(YACC) -p zend -v -d /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_parser.y -o $@

/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_parser.h: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_parser.c
/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_parser.c: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_parser.y
	@$(YACC) -p ini_ -v -d /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_parser.y -o $@

/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_scanner.c: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_scanner.l
	@(cd $(top_srcdir); $(RE2C) $(RE2C_FLAGS) --no-generation-date --case-inverted -cbdFt Zend/zend_ini_scanner_defs.h -oZend/zend_ini_scanner.c Zend/zend_ini_scanner.l)

Zend/zend_indent.lo Zend/zend_highlight.lo Zend/zend_compile.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_parser.h
Zend/zend_execute.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_vm_execute.h /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_vm_opcodes.h
sapi/cli/php_cli.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/php_cli.c
	$(LIBTOOL) --mode=compile $(CC)  -Isapi/cli/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/php_cli.c -o sapi/cli/php_cli.lo 
sapi/cli/php_http_parser.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/php_http_parser.c
	$(LIBTOOL) --mode=compile $(CC)  -Isapi/cli/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/php_http_parser.c -o sapi/cli/php_http_parser.lo 
sapi/cli/php_cli_server.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/php_cli_server.c
	$(LIBTOOL) --mode=compile $(CC)  -Isapi/cli/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/php_cli_server.c -o sapi/cli/php_cli_server.lo 
sapi/cli/ps_title.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/ps_title.c
	$(LIBTOOL) --mode=compile $(CC)  -Isapi/cli/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/ps_title.c -o sapi/cli/ps_title.lo 
sapi/cli/php_cli_process_title.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/php_cli_process_title.c
	$(LIBTOOL) --mode=compile $(CC)  -Isapi/cli/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cli/php_cli_process_title.c -o sapi/cli/php_cli_process_title.lo 
sapi/cgi/cgi_main.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cgi/cgi_main.c
	$(LIBTOOL) --mode=compile $(CC)  -Isapi/cgi/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cgi/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cgi/cgi_main.c -o sapi/cgi/cgi_main.lo 
sapi/cgi/fastcgi.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cgi/fastcgi.c
	$(LIBTOOL) --mode=compile $(CC)  -Isapi/cgi/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cgi/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/sapi/cgi/fastcgi.c -o sapi/cgi/fastcgi.lo 
ext/date/php_date.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/php_date.c
	$(LIBTOOL) --mode=compile $(CC) -Iext/date/lib -Iext/date/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/php_date.c -o ext/date/php_date.lo 
ext/date/lib/astro.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/astro.c
	$(LIBTOOL) --mode=compile $(CC) -Iext/date/lib -Iext/date/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/astro.c -o ext/date/lib/astro.lo 
ext/date/lib/dow.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/dow.c
	$(LIBTOOL) --mode=compile $(CC) -Iext/date/lib -Iext/date/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/dow.c -o ext/date/lib/dow.lo 
ext/date/lib/parse_date.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/parse_date.c
	$(LIBTOOL) --mode=compile $(CC) -Iext/date/lib -Iext/date/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/parse_date.c -o ext/date/lib/parse_date.lo 
ext/date/lib/parse_tz.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/parse_tz.c
	$(LIBTOOL) --mode=compile $(CC) -Iext/date/lib -Iext/date/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/parse_tz.c -o ext/date/lib/parse_tz.lo 
ext/date/lib/timelib.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/timelib.c
	$(LIBTOOL) --mode=compile $(CC) -Iext/date/lib -Iext/date/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/timelib.c -o ext/date/lib/timelib.lo 
ext/date/lib/tm2unixtime.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/tm2unixtime.c
	$(LIBTOOL) --mode=compile $(CC) -Iext/date/lib -Iext/date/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/tm2unixtime.c -o ext/date/lib/tm2unixtime.lo 
ext/date/lib/unixtime2tm.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/unixtime2tm.c
	$(LIBTOOL) --mode=compile $(CC) -Iext/date/lib -Iext/date/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/unixtime2tm.c -o ext/date/lib/unixtime2tm.lo 
ext/date/lib/parse_iso_intervals.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/parse_iso_intervals.c
	$(LIBTOOL) --mode=compile $(CC) -Iext/date/lib -Iext/date/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/parse_iso_intervals.c -o ext/date/lib/parse_iso_intervals.lo 
ext/date/lib/interval.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/interval.c
	$(LIBTOOL) --mode=compile $(CC) -Iext/date/lib -Iext/date/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/date/lib/interval.c -o ext/date/lib/interval.lo 
ext/ereg/ereg.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/ereg.c
	$(LIBTOOL) --mode=compile $(CC) -Dregexec=php_regexec -Dregerror=php_regerror -Dregfree=php_regfree -Dregcomp=php_regcomp -Iext/ereg/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/ereg.c -o ext/ereg/ereg.lo 
ext/ereg/regex/regcomp.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/regex/regcomp.c
	$(LIBTOOL) --mode=compile $(CC) -Dregexec=php_regexec -Dregerror=php_regerror -Dregfree=php_regfree -Dregcomp=php_regcomp -Iext/ereg/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/regex/regcomp.c -o ext/ereg/regex/regcomp.lo 
ext/ereg/regex/regexec.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/regex/regexec.c
	$(LIBTOOL) --mode=compile $(CC) -Dregexec=php_regexec -Dregerror=php_regerror -Dregfree=php_regfree -Dregcomp=php_regcomp -Iext/ereg/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/regex/regexec.c -o ext/ereg/regex/regexec.lo 
ext/ereg/regex/regerror.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/regex/regerror.c
	$(LIBTOOL) --mode=compile $(CC) -Dregexec=php_regexec -Dregerror=php_regerror -Dregfree=php_regfree -Dregcomp=php_regcomp -Iext/ereg/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/regex/regerror.c -o ext/ereg/regex/regerror.lo 
ext/ereg/regex/regfree.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/regex/regfree.c
	$(LIBTOOL) --mode=compile $(CC) -Dregexec=php_regexec -Dregerror=php_regerror -Dregfree=php_regfree -Dregcomp=php_regcomp -Iext/ereg/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ereg/regex/regfree.c -o ext/ereg/regex/regfree.lo 
ext/libxml/libxml.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/libxml/libxml.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/libxml/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/libxml/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/libxml/libxml.c -o ext/libxml/libxml.lo 
ext/pcre/pcrelib/pcre_chartables.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_chartables.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_chartables.c -o ext/pcre/pcrelib/pcre_chartables.lo 
ext/pcre/pcrelib/pcre_ucd.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_ucd.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_ucd.c -o ext/pcre/pcrelib/pcre_ucd.lo 
ext/pcre/pcrelib/pcre_compile.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_compile.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_compile.c -o ext/pcre/pcrelib/pcre_compile.lo 
ext/pcre/pcrelib/pcre_config.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_config.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_config.c -o ext/pcre/pcrelib/pcre_config.lo 
ext/pcre/pcrelib/pcre_exec.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_exec.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_exec.c -o ext/pcre/pcrelib/pcre_exec.lo 
ext/pcre/pcrelib/pcre_fullinfo.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_fullinfo.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_fullinfo.c -o ext/pcre/pcrelib/pcre_fullinfo.lo 
ext/pcre/pcrelib/pcre_get.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_get.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_get.c -o ext/pcre/pcrelib/pcre_get.lo 
ext/pcre/pcrelib/pcre_globals.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_globals.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_globals.c -o ext/pcre/pcrelib/pcre_globals.lo 
ext/pcre/pcrelib/pcre_maketables.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_maketables.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_maketables.c -o ext/pcre/pcrelib/pcre_maketables.lo 
ext/pcre/pcrelib/pcre_newline.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_newline.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_newline.c -o ext/pcre/pcrelib/pcre_newline.lo 
ext/pcre/pcrelib/pcre_ord2utf8.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_ord2utf8.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_ord2utf8.c -o ext/pcre/pcrelib/pcre_ord2utf8.lo 
ext/pcre/pcrelib/pcre_refcount.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_refcount.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_refcount.c -o ext/pcre/pcrelib/pcre_refcount.lo 
ext/pcre/pcrelib/pcre_study.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_study.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_study.c -o ext/pcre/pcrelib/pcre_study.lo 
ext/pcre/pcrelib/pcre_tables.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_tables.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_tables.c -o ext/pcre/pcrelib/pcre_tables.lo 
ext/pcre/pcrelib/pcre_valid_utf8.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_valid_utf8.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_valid_utf8.c -o ext/pcre/pcrelib/pcre_valid_utf8.lo 
ext/pcre/pcrelib/pcre_version.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_version.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_version.c -o ext/pcre/pcrelib/pcre_version.lo 
ext/pcre/pcrelib/pcre_xclass.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_xclass.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_xclass.c -o ext/pcre/pcrelib/pcre_xclass.lo 
ext/pcre/pcrelib/pcre_jit_compile.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_jit_compile.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib/pcre_jit_compile.c -o ext/pcre/pcrelib/pcre_jit_compile.lo 
ext/pcre/php_pcre.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/php_pcre.c
	$(LIBTOOL) --mode=compile $(CC) -DHAVE_CONFIG_H -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/pcrelib -Iext/pcre/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pcre/php_pcre.c -o ext/pcre/php_pcre.lo 
ext/sqlite3/sqlite3.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/sqlite3/sqlite3.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/sqlite3/libsqlite -DSQLITE_ENABLE_FTS3=1 -DSQLITE_CORE=1 -DSQLITE_ENABLE_COLUMN_METADATA=1 -DSQLITE_THREADSAFE=0  -Iext/sqlite3/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/sqlite3/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/sqlite3/sqlite3.c -o ext/sqlite3/sqlite3.lo 
ext/sqlite3/libsqlite/sqlite3.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/sqlite3/libsqlite/sqlite3.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/sqlite3/libsqlite -DSQLITE_ENABLE_FTS3=1 -DSQLITE_CORE=1 -DSQLITE_ENABLE_COLUMN_METADATA=1 -DSQLITE_THREADSAFE=0  -Iext/sqlite3/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/sqlite3/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/sqlite3/libsqlite/sqlite3.c -o ext/sqlite3/libsqlite/sqlite3.lo 
ext/ctype/ctype.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ctype/ctype.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/ctype/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ctype/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/ctype/ctype.c -o ext/ctype/ctype.lo 
ext/dom/php_dom.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/php_dom.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/php_dom.c -o ext/dom/php_dom.lo 
ext/dom/attr.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/attr.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/attr.c -o ext/dom/attr.lo 
ext/dom/document.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/document.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/document.c -o ext/dom/document.lo 
ext/dom/domerrorhandler.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domerrorhandler.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domerrorhandler.c -o ext/dom/domerrorhandler.lo 
ext/dom/domstringlist.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domstringlist.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domstringlist.c -o ext/dom/domstringlist.lo 
ext/dom/domexception.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domexception.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domexception.c -o ext/dom/domexception.lo 
ext/dom/namelist.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/namelist.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/namelist.c -o ext/dom/namelist.lo 
ext/dom/processinginstruction.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/processinginstruction.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/processinginstruction.c -o ext/dom/processinginstruction.lo 
ext/dom/cdatasection.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/cdatasection.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/cdatasection.c -o ext/dom/cdatasection.lo 
ext/dom/documentfragment.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/documentfragment.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/documentfragment.c -o ext/dom/documentfragment.lo 
ext/dom/domimplementation.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domimplementation.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domimplementation.c -o ext/dom/domimplementation.lo 
ext/dom/element.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/element.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/element.c -o ext/dom/element.lo 
ext/dom/node.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/node.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/node.c -o ext/dom/node.lo 
ext/dom/string_extend.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/string_extend.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/string_extend.c -o ext/dom/string_extend.lo 
ext/dom/characterdata.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/characterdata.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/characterdata.c -o ext/dom/characterdata.lo 
ext/dom/documenttype.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/documenttype.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/documenttype.c -o ext/dom/documenttype.lo 
ext/dom/domimplementationlist.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domimplementationlist.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domimplementationlist.c -o ext/dom/domimplementationlist.lo 
ext/dom/entity.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/entity.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/entity.c -o ext/dom/entity.lo 
ext/dom/nodelist.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/nodelist.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/nodelist.c -o ext/dom/nodelist.lo 
ext/dom/text.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/text.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/text.c -o ext/dom/text.lo 
ext/dom/comment.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/comment.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/comment.c -o ext/dom/comment.lo 
ext/dom/domconfiguration.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domconfiguration.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domconfiguration.c -o ext/dom/domconfiguration.lo 
ext/dom/domimplementationsource.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domimplementationsource.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domimplementationsource.c -o ext/dom/domimplementationsource.lo 
ext/dom/entityreference.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/entityreference.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/entityreference.c -o ext/dom/entityreference.lo 
ext/dom/notation.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/notation.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/notation.c -o ext/dom/notation.lo 
ext/dom/xpath.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/xpath.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/xpath.c -o ext/dom/xpath.lo 
ext/dom/dom_iterators.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/dom_iterators.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/dom_iterators.c -o ext/dom/dom_iterators.lo 
ext/dom/typeinfo.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/typeinfo.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/typeinfo.c -o ext/dom/typeinfo.lo 
ext/dom/domerror.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domerror.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domerror.c -o ext/dom/domerror.lo 
ext/dom/domlocator.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domlocator.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/domlocator.c -o ext/dom/domlocator.lo 
ext/dom/namednodemap.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/namednodemap.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/namednodemap.c -o ext/dom/namednodemap.lo 
ext/dom/userdatahandler.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/userdatahandler.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/dom/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/dom/userdatahandler.c -o ext/dom/userdatahandler.lo 
ext/fileinfo/fileinfo.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/fileinfo.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/fileinfo.c -o ext/fileinfo/fileinfo.lo 
ext/fileinfo/libmagic/apprentice.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/apprentice.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/apprentice.c -o ext/fileinfo/libmagic/apprentice.lo 
ext/fileinfo/libmagic/apptype.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/apptype.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/apptype.c -o ext/fileinfo/libmagic/apptype.lo 
ext/fileinfo/libmagic/ascmagic.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/ascmagic.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/ascmagic.c -o ext/fileinfo/libmagic/ascmagic.lo 
ext/fileinfo/libmagic/cdf.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/cdf.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/cdf.c -o ext/fileinfo/libmagic/cdf.lo 
ext/fileinfo/libmagic/cdf_time.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/cdf_time.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/cdf_time.c -o ext/fileinfo/libmagic/cdf_time.lo 
ext/fileinfo/libmagic/compress.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/compress.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/compress.c -o ext/fileinfo/libmagic/compress.lo 
ext/fileinfo/libmagic/encoding.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/encoding.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/encoding.c -o ext/fileinfo/libmagic/encoding.lo 
ext/fileinfo/libmagic/fsmagic.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/fsmagic.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/fsmagic.c -o ext/fileinfo/libmagic/fsmagic.lo 
ext/fileinfo/libmagic/funcs.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/funcs.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/funcs.c -o ext/fileinfo/libmagic/funcs.lo 
ext/fileinfo/libmagic/is_tar.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/is_tar.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/is_tar.c -o ext/fileinfo/libmagic/is_tar.lo 
ext/fileinfo/libmagic/magic.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/magic.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/magic.c -o ext/fileinfo/libmagic/magic.lo 
ext/fileinfo/libmagic/print.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/print.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/print.c -o ext/fileinfo/libmagic/print.lo 
ext/fileinfo/libmagic/readcdf.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/readcdf.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/readcdf.c -o ext/fileinfo/libmagic/readcdf.lo 
ext/fileinfo/libmagic/softmagic.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/softmagic.c
	$(LIBTOOL) --mode=compile $(CC) -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic -Iext/fileinfo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/fileinfo/libmagic/softmagic.c -o ext/fileinfo/libmagic/softmagic.lo 
ext/filter/filter.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/filter.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/filter/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/filter.c -o ext/filter/filter.lo 
ext/filter/sanitizing_filters.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/sanitizing_filters.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/filter/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/sanitizing_filters.c -o ext/filter/sanitizing_filters.lo 
ext/filter/logical_filters.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/logical_filters.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/filter/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/logical_filters.c -o ext/filter/logical_filters.lo 
ext/filter/callback_filter.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/callback_filter.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/filter/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/filter/callback_filter.c -o ext/filter/callback_filter.lo 
ext/hash/hash.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash.c -o ext/hash/hash.lo 
ext/hash/hash_md.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_md.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_md.c -o ext/hash/hash_md.lo 
ext/hash/hash_sha.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_sha.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_sha.c -o ext/hash/hash_sha.lo 
ext/hash/hash_ripemd.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_ripemd.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_ripemd.c -o ext/hash/hash_ripemd.lo 
ext/hash/hash_haval.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_haval.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_haval.c -o ext/hash/hash_haval.lo 
ext/hash/hash_tiger.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_tiger.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_tiger.c -o ext/hash/hash_tiger.lo 
ext/hash/hash_gost.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_gost.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_gost.c -o ext/hash/hash_gost.lo 
ext/hash/hash_snefru.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_snefru.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_snefru.c -o ext/hash/hash_snefru.lo 
ext/hash/hash_whirlpool.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_whirlpool.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_whirlpool.c -o ext/hash/hash_whirlpool.lo 
ext/hash/hash_adler32.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_adler32.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_adler32.c -o ext/hash/hash_adler32.lo 
ext/hash/hash_crc32.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_crc32.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_crc32.c -o ext/hash/hash_crc32.lo 
ext/hash/hash_fnv.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_fnv.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_fnv.c -o ext/hash/hash_fnv.lo 
ext/hash/hash_joaat.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_joaat.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/hash/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/hash/hash_joaat.c -o ext/hash/hash_joaat.lo 
ext/iconv/iconv.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/iconv/iconv.c
	$(LIBTOOL) --mode=compile $(CC) -I"/usr/include" -Iext/iconv/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/iconv/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/iconv/iconv.c -o ext/iconv/iconv.lo 
ext/json/json.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/json/json.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/json/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/json/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/json/json.c -o ext/json/json.lo 
ext/json/utf8_decode.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/json/utf8_decode.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/json/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/json/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/json/utf8_decode.c -o ext/json/utf8_decode.lo 
ext/json/JSON_parser.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/json/JSON_parser.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/json/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/json/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/json/JSON_parser.c -o ext/json/JSON_parser.lo 
ext/opcache/ZendAccelerator.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ZendAccelerator.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ZendAccelerator.c -o ext/opcache/ZendAccelerator.lo 
ext/opcache/zend_accelerator_blacklist.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_accelerator_blacklist.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_accelerator_blacklist.c -o ext/opcache/zend_accelerator_blacklist.lo 
ext/opcache/zend_accelerator_debug.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_accelerator_debug.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_accelerator_debug.c -o ext/opcache/zend_accelerator_debug.lo 
ext/opcache/zend_accelerator_hash.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_accelerator_hash.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_accelerator_hash.c -o ext/opcache/zend_accelerator_hash.lo 
ext/opcache/zend_accelerator_module.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_accelerator_module.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_accelerator_module.c -o ext/opcache/zend_accelerator_module.lo 
ext/opcache/zend_persist.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_persist.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_persist.c -o ext/opcache/zend_persist.lo 
ext/opcache/zend_persist_calc.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_persist_calc.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_persist_calc.c -o ext/opcache/zend_persist_calc.lo 
ext/opcache/zend_shared_alloc.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_shared_alloc.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_shared_alloc.c -o ext/opcache/zend_shared_alloc.lo 
ext/opcache/zend_accelerator_util_funcs.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_accelerator_util_funcs.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/zend_accelerator_util_funcs.c -o ext/opcache/zend_accelerator_util_funcs.lo 
ext/opcache/shared_alloc_shm.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/shared_alloc_shm.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/shared_alloc_shm.c -o ext/opcache/shared_alloc_shm.lo 
ext/opcache/shared_alloc_mmap.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/shared_alloc_mmap.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/shared_alloc_mmap.c -o ext/opcache/shared_alloc_mmap.lo 
ext/opcache/shared_alloc_posix.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/shared_alloc_posix.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/shared_alloc_posix.c -o ext/opcache/shared_alloc_posix.lo 
ext/opcache/Optimizer/zend_optimizer.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/Optimizer/zend_optimizer.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/opcache/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/opcache/Optimizer/zend_optimizer.c -o ext/opcache/Optimizer/zend_optimizer.lo 
$(phplibdir)/opcache.la: ext/opcache/opcache.la
	$(LIBTOOL) --mode=install cp ext/opcache/opcache.la $(phplibdir)

ext/opcache/opcache.la: $(shared_objects_opcache) $(OPCACHE_SHARED_DEPENDENCIES)
	$(LIBTOOL) --mode=link $(CC) $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) $(LDFLAGS) -o $@ -export-dynamic -avoid-version -prefer-pic -module -rpath $(phplibdir) $(EXTRA_LDFLAGS) $(shared_objects_opcache) $(OPCACHE_SHARED_LIBADD)

ext/pdo/pdo.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/pdo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo.c -o ext/pdo/pdo.lo 
ext/pdo/pdo_dbh.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo_dbh.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/pdo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo_dbh.c -o ext/pdo/pdo_dbh.lo 
ext/pdo/pdo_stmt.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo_stmt.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/pdo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo_stmt.c -o ext/pdo/pdo_stmt.lo 
ext/pdo/pdo_sql_parser.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo_sql_parser.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/pdo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo_sql_parser.c -o ext/pdo/pdo_sql_parser.lo 
ext/pdo/pdo_sqlstate.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo_sqlstate.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/pdo/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo/pdo_sqlstate.c -o ext/pdo/pdo_sqlstate.lo 
ext/pdo_sqlite/pdo_sqlite.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo_sqlite/pdo_sqlite.c
	$(LIBTOOL) --mode=compile $(CC) -DPDO_SQLITE_BUNDLED=1 -DSQLITE_ENABLE_FTS3=1 -DSQLITE_CORE=1 -DSQLITE_ENABLE_COLUMN_METADATA=1 -DSQLITE_THREADSAFE=0 -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext -Iext/pdo_sqlite/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo_sqlite/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo_sqlite/pdo_sqlite.c -o ext/pdo_sqlite/pdo_sqlite.lo 
ext/pdo_sqlite/sqlite_driver.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo_sqlite/sqlite_driver.c
	$(LIBTOOL) --mode=compile $(CC) -DPDO_SQLITE_BUNDLED=1 -DSQLITE_ENABLE_FTS3=1 -DSQLITE_CORE=1 -DSQLITE_ENABLE_COLUMN_METADATA=1 -DSQLITE_THREADSAFE=0 -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext -Iext/pdo_sqlite/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo_sqlite/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo_sqlite/sqlite_driver.c -o ext/pdo_sqlite/sqlite_driver.lo 
ext/pdo_sqlite/sqlite_statement.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo_sqlite/sqlite_statement.c
	$(LIBTOOL) --mode=compile $(CC) -DPDO_SQLITE_BUNDLED=1 -DSQLITE_ENABLE_FTS3=1 -DSQLITE_CORE=1 -DSQLITE_ENABLE_COLUMN_METADATA=1 -DSQLITE_THREADSAFE=0 -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext -Iext/pdo_sqlite/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo_sqlite/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/pdo_sqlite/sqlite_statement.c -o ext/pdo_sqlite/sqlite_statement.lo 
ext/phar/util.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/util.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/phar/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/util.c -o ext/phar/util.lo 
ext/phar/tar.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/tar.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/phar/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/tar.c -o ext/phar/tar.lo 
ext/phar/zip.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/zip.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/phar/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/zip.c -o ext/phar/zip.lo 
ext/phar/stream.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/stream.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/phar/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/stream.c -o ext/phar/stream.lo 
ext/phar/func_interceptors.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/func_interceptors.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/phar/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/func_interceptors.c -o ext/phar/func_interceptors.lo 
ext/phar/dirstream.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/dirstream.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/phar/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/dirstream.c -o ext/phar/dirstream.lo 
ext/phar/phar.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/phar/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar.c -o ext/phar/phar.lo 
ext/phar/phar_object.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar_object.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/phar/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar_object.c -o ext/phar/phar_object.lo 
ext/phar/phar_path_check.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar_path_check.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/phar/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/phar/phar_path_check.c -o ext/phar/phar_path_check.lo 
ext/posix/posix.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/posix/posix.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/posix/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/posix/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/posix/posix.c -o ext/posix/posix.lo 
ext/reflection/php_reflection.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/reflection/php_reflection.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/reflection/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/reflection/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/reflection/php_reflection.c -o ext/reflection/php_reflection.lo 
ext/session/mod_user_class.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/mod_user_class.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/session/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/mod_user_class.c -o ext/session/mod_user_class.lo 
ext/session/session.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/session.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/session/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/session.c -o ext/session/session.lo 
ext/session/mod_files.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/mod_files.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/session/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/mod_files.c -o ext/session/mod_files.lo 
ext/session/mod_mm.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/mod_mm.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/session/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/mod_mm.c -o ext/session/mod_mm.lo 
ext/session/mod_user.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/mod_user.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/session/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/session/mod_user.c -o ext/session/mod_user.lo 
ext/simplexml/simplexml.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/simplexml/simplexml.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/simplexml/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/simplexml/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/simplexml/simplexml.c -o ext/simplexml/simplexml.lo 
ext/simplexml/sxe.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/simplexml/sxe.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/simplexml/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/simplexml/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/simplexml/sxe.c -o ext/simplexml/sxe.lo 
ext/spl/php_spl.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/php_spl.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/php_spl.c -o ext/spl/php_spl.lo 
ext/spl/spl_functions.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_functions.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_functions.c -o ext/spl/spl_functions.lo 
ext/spl/spl_engine.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_engine.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_engine.c -o ext/spl/spl_engine.lo 
ext/spl/spl_iterators.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_iterators.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_iterators.c -o ext/spl/spl_iterators.lo 
ext/spl/spl_array.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_array.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_array.c -o ext/spl/spl_array.lo 
ext/spl/spl_directory.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_directory.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_directory.c -o ext/spl/spl_directory.lo 
ext/spl/spl_exceptions.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_exceptions.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_exceptions.c -o ext/spl/spl_exceptions.lo 
ext/spl/spl_observer.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_observer.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_observer.c -o ext/spl/spl_observer.lo 
ext/spl/spl_dllist.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_dllist.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_dllist.c -o ext/spl/spl_dllist.lo 
ext/spl/spl_heap.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_heap.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_heap.c -o ext/spl/spl_heap.lo 
ext/spl/spl_fixedarray.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_fixedarray.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/spl/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/spl/spl_fixedarray.c -o ext/spl/spl_fixedarray.lo 
ext/standard/crypt_freesec.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crypt_freesec.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crypt_freesec.c -o ext/standard/crypt_freesec.lo 
ext/standard/crypt_blowfish.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crypt_blowfish.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crypt_blowfish.c -o ext/standard/crypt_blowfish.lo 
ext/standard/crypt_sha512.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crypt_sha512.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crypt_sha512.c -o ext/standard/crypt_sha512.lo 
ext/standard/crypt_sha256.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crypt_sha256.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crypt_sha256.c -o ext/standard/crypt_sha256.lo 
ext/standard/php_crypt_r.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/php_crypt_r.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/php_crypt_r.c -o ext/standard/php_crypt_r.lo 
ext/standard/array.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/array.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/array.c -o ext/standard/array.lo 
ext/standard/base64.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/base64.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/base64.c -o ext/standard/base64.lo 
ext/standard/basic_functions.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/basic_functions.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/basic_functions.c -o ext/standard/basic_functions.lo 
ext/standard/browscap.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/browscap.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/browscap.c -o ext/standard/browscap.lo 
ext/standard/crc32.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crc32.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crc32.c -o ext/standard/crc32.lo 
ext/standard/crypt.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crypt.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/crypt.c -o ext/standard/crypt.lo 
ext/standard/cyr_convert.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/cyr_convert.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/cyr_convert.c -o ext/standard/cyr_convert.lo 
ext/standard/datetime.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/datetime.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/datetime.c -o ext/standard/datetime.lo 
ext/standard/dir.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/dir.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/dir.c -o ext/standard/dir.lo 
ext/standard/dl.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/dl.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/dl.c -o ext/standard/dl.lo 
ext/standard/dns.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/dns.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/dns.c -o ext/standard/dns.lo 
ext/standard/exec.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/exec.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/exec.c -o ext/standard/exec.lo 
ext/standard/file.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/file.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/file.c -o ext/standard/file.lo 
ext/standard/filestat.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/filestat.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/filestat.c -o ext/standard/filestat.lo 
ext/standard/flock_compat.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/flock_compat.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/flock_compat.c -o ext/standard/flock_compat.lo 
ext/standard/formatted_print.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/formatted_print.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/formatted_print.c -o ext/standard/formatted_print.lo 
ext/standard/fsock.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/fsock.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/fsock.c -o ext/standard/fsock.lo 
ext/standard/head.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/head.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/head.c -o ext/standard/head.lo 
ext/standard/html.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/html.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/html.c -o ext/standard/html.lo 
ext/standard/image.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/image.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/image.c -o ext/standard/image.lo 
ext/standard/info.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/info.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/info.c -o ext/standard/info.lo 
ext/standard/iptc.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/iptc.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/iptc.c -o ext/standard/iptc.lo 
ext/standard/lcg.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/lcg.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/lcg.c -o ext/standard/lcg.lo 
ext/standard/link.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/link.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/link.c -o ext/standard/link.lo 
ext/standard/mail.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/mail.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/mail.c -o ext/standard/mail.lo 
ext/standard/math.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/math.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/math.c -o ext/standard/math.lo 
ext/standard/md5.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/md5.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/md5.c -o ext/standard/md5.lo 
ext/standard/metaphone.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/metaphone.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/metaphone.c -o ext/standard/metaphone.lo 
ext/standard/microtime.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/microtime.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/microtime.c -o ext/standard/microtime.lo 
ext/standard/pack.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/pack.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/pack.c -o ext/standard/pack.lo 
ext/standard/pageinfo.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/pageinfo.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/pageinfo.c -o ext/standard/pageinfo.lo 
ext/standard/quot_print.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/quot_print.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/quot_print.c -o ext/standard/quot_print.lo 
ext/standard/rand.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/rand.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/rand.c -o ext/standard/rand.lo 
ext/standard/soundex.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/soundex.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/soundex.c -o ext/standard/soundex.lo 
ext/standard/string.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/string.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/string.c -o ext/standard/string.lo 
ext/standard/scanf.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/scanf.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/scanf.c -o ext/standard/scanf.lo 
ext/standard/syslog.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/syslog.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/syslog.c -o ext/standard/syslog.lo 
ext/standard/type.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/type.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/type.c -o ext/standard/type.lo 
ext/standard/uniqid.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/uniqid.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/uniqid.c -o ext/standard/uniqid.lo 
ext/standard/url.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/url.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/url.c -o ext/standard/url.lo 
ext/standard/var.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/var.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/var.c -o ext/standard/var.lo 
ext/standard/versioning.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/versioning.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/versioning.c -o ext/standard/versioning.lo 
ext/standard/assert.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/assert.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/assert.c -o ext/standard/assert.lo 
ext/standard/strnatcmp.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/strnatcmp.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/strnatcmp.c -o ext/standard/strnatcmp.lo 
ext/standard/levenshtein.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/levenshtein.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/levenshtein.c -o ext/standard/levenshtein.lo 
ext/standard/incomplete_class.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/incomplete_class.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/incomplete_class.c -o ext/standard/incomplete_class.lo 
ext/standard/url_scanner_ex.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/url_scanner_ex.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/url_scanner_ex.c -o ext/standard/url_scanner_ex.lo 
ext/standard/ftp_fopen_wrapper.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ftp_fopen_wrapper.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ftp_fopen_wrapper.c -o ext/standard/ftp_fopen_wrapper.lo 
ext/standard/http_fopen_wrapper.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/http_fopen_wrapper.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/http_fopen_wrapper.c -o ext/standard/http_fopen_wrapper.lo 
ext/standard/php_fopen_wrapper.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/php_fopen_wrapper.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/php_fopen_wrapper.c -o ext/standard/php_fopen_wrapper.lo 
ext/standard/credits.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/credits.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/credits.c -o ext/standard/credits.lo 
ext/standard/css.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/css.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/css.c -o ext/standard/css.lo 
ext/standard/var_unserializer.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/var_unserializer.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/var_unserializer.c -o ext/standard/var_unserializer.lo 
ext/standard/ftok.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ftok.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ftok.c -o ext/standard/ftok.lo 
ext/standard/sha1.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/sha1.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/sha1.c -o ext/standard/sha1.lo 
ext/standard/user_filters.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/user_filters.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/user_filters.c -o ext/standard/user_filters.lo 
ext/standard/uuencode.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/uuencode.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/uuencode.c -o ext/standard/uuencode.lo 
ext/standard/filters.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/filters.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/filters.c -o ext/standard/filters.lo 
ext/standard/proc_open.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/proc_open.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/proc_open.c -o ext/standard/proc_open.lo 
ext/standard/streamsfuncs.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/streamsfuncs.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/streamsfuncs.c -o ext/standard/streamsfuncs.lo 
ext/standard/http.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/http.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/http.c -o ext/standard/http.lo 
ext/standard/password.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/password.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/standard/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/standard/password.c -o ext/standard/password.lo 
ext/tokenizer/tokenizer.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/tokenizer/tokenizer.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/tokenizer/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/tokenizer/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/tokenizer/tokenizer.c -o ext/tokenizer/tokenizer.lo 
ext/tokenizer/tokenizer_data.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/tokenizer/tokenizer_data.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/tokenizer/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/tokenizer/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/tokenizer/tokenizer_data.c -o ext/tokenizer/tokenizer_data.lo 
ext/xml/xml.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xml/xml.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/xml/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xml/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xml/xml.c -o ext/xml/xml.lo 
ext/xml/compat.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xml/compat.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/xml/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xml/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xml/compat.c -o ext/xml/compat.lo 
ext/xmlreader/php_xmlreader.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xmlreader/php_xmlreader.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/xmlreader/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xmlreader/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xmlreader/php_xmlreader.c -o ext/xmlreader/php_xmlreader.lo 
ext/xmlwriter/php_xmlwriter.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xmlwriter/php_xmlwriter.c
	$(LIBTOOL) --mode=compile $(CC)  -Iext/xmlwriter/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xmlwriter/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/ext/xmlwriter/php_xmlwriter.c -o ext/xmlwriter/php_xmlwriter.lo 
TSRM/TSRM.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/TSRM/TSRM.c
	$(LIBTOOL) --mode=compile $(CC)  -ITSRM/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/TSRM/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/TSRM/TSRM.c -o TSRM/TSRM.lo 
TSRM/tsrm_strtok_r.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/TSRM/tsrm_strtok_r.c
	$(LIBTOOL) --mode=compile $(CC)  -ITSRM/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/TSRM/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/TSRM/tsrm_strtok_r.c -o TSRM/tsrm_strtok_r.lo 
TSRM/tsrm_virtual_cwd.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/TSRM/tsrm_virtual_cwd.c
	$(LIBTOOL) --mode=compile $(CC)  -ITSRM/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/TSRM/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/TSRM/tsrm_virtual_cwd.c -o TSRM/tsrm_virtual_cwd.lo 
main/main.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/main.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/main.c -o main/main.lo 
main/snprintf.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/snprintf.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/snprintf.c -o main/snprintf.lo 
main/spprintf.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/spprintf.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/spprintf.c -o main/spprintf.lo 
main/php_sprintf.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_sprintf.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_sprintf.c -o main/php_sprintf.lo 
main/fopen_wrappers.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/fopen_wrappers.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/fopen_wrappers.c -o main/fopen_wrappers.lo 
main/alloca.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/alloca.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/alloca.c -o main/alloca.lo 
main/php_scandir.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_scandir.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_scandir.c -o main/php_scandir.lo 
main/php_ini.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_ini.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_ini.c -o main/php_ini.lo 
main/SAPI.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/SAPI.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/SAPI.c -o main/SAPI.lo 
main/rfc1867.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/rfc1867.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/rfc1867.c -o main/rfc1867.lo 
main/php_content_types.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_content_types.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_content_types.c -o main/php_content_types.lo 
main/strlcpy.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/strlcpy.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/strlcpy.c -o main/strlcpy.lo 
main/strlcat.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/strlcat.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/strlcat.c -o main/strlcat.lo 
main/mergesort.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/mergesort.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/mergesort.c -o main/mergesort.lo 
main/reentrancy.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/reentrancy.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/reentrancy.c -o main/reentrancy.lo 
main/php_variables.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_variables.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_variables.c -o main/php_variables.lo 
main/php_ticks.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_ticks.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_ticks.c -o main/php_ticks.lo 
main/network.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/network.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/network.c -o main/network.lo 
main/php_open_temporary_file.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_open_temporary_file.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/php_open_temporary_file.c -o main/php_open_temporary_file.lo 
main/output.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/output.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/output.c -o main/output.lo 
main/getopt.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/getopt.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/getopt.c -o main/getopt.lo 
main/streams/streams.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/streams.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/streams/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/streams.c -o main/streams/streams.lo 
main/streams/cast.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/cast.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/streams/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/cast.c -o main/streams/cast.lo 
main/streams/memory.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/memory.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/streams/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/memory.c -o main/streams/memory.lo 
main/streams/filter.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/filter.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/streams/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/filter.c -o main/streams/filter.lo 
main/streams/plain_wrapper.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/plain_wrapper.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/streams/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/plain_wrapper.c -o main/streams/plain_wrapper.lo 
main/streams/userspace.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/userspace.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/streams/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/userspace.c -o main/streams/userspace.lo 
main/streams/transports.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/transports.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/streams/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/transports.c -o main/streams/transports.lo 
main/streams/xp_socket.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/xp_socket.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/streams/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/xp_socket.c -o main/streams/xp_socket.lo 
main/streams/mmap.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/mmap.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/streams/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/mmap.c -o main/streams/mmap.lo 
main/streams/glob_wrapper.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/glob_wrapper.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/streams/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/main/streams/glob_wrapper.c -o main/streams/glob_wrapper.lo 
main/internal_functions.lo: main/internal_functions.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c main/internal_functions.c -o main/internal_functions.lo 
main/internal_functions_cli.lo: main/internal_functions_cli.c
	$(LIBTOOL) --mode=compile $(CC)  -Imain/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/main/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c main/internal_functions_cli.c -o main/internal_functions_cli.lo 
Zend/zend_language_parser.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_parser.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_parser.c -o Zend/zend_language_parser.lo 
Zend/zend_language_scanner.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_scanner.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_language_scanner.c -o Zend/zend_language_scanner.lo 
Zend/zend_ini_parser.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_parser.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_parser.c -o Zend/zend_ini_parser.lo 
Zend/zend_ini_scanner.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_scanner.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini_scanner.c -o Zend/zend_ini_scanner.lo 
Zend/zend_alloc.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_alloc.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_alloc.c -o Zend/zend_alloc.lo 
Zend/zend_compile.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_compile.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_compile.c -o Zend/zend_compile.lo 
Zend/zend_constants.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_constants.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_constants.c -o Zend/zend_constants.lo 
Zend/zend_dynamic_array.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_dynamic_array.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_dynamic_array.c -o Zend/zend_dynamic_array.lo 
Zend/zend_dtrace.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_dtrace.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_dtrace.c -o Zend/zend_dtrace.lo 
Zend/zend_execute_API.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_execute_API.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_execute_API.c -o Zend/zend_execute_API.lo 
Zend/zend_highlight.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_highlight.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_highlight.c -o Zend/zend_highlight.lo 
Zend/zend_llist.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_llist.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_llist.c -o Zend/zend_llist.lo 
Zend/zend_vm_opcodes.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_vm_opcodes.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_vm_opcodes.c -o Zend/zend_vm_opcodes.lo 
Zend/zend_opcode.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_opcode.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_opcode.c -o Zend/zend_opcode.lo 
Zend/zend_operators.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_operators.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_operators.c -o Zend/zend_operators.lo 
Zend/zend_ptr_stack.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ptr_stack.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ptr_stack.c -o Zend/zend_ptr_stack.lo 
Zend/zend_stack.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_stack.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_stack.c -o Zend/zend_stack.lo 
Zend/zend_variables.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_variables.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_variables.c -o Zend/zend_variables.lo 
Zend/zend.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend.c -o Zend/zend.lo 
Zend/zend_API.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_API.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_API.c -o Zend/zend_API.lo 
Zend/zend_extensions.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_extensions.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_extensions.c -o Zend/zend_extensions.lo 
Zend/zend_hash.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_hash.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_hash.c -o Zend/zend_hash.lo 
Zend/zend_list.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_list.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_list.c -o Zend/zend_list.lo 
Zend/zend_indent.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_indent.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_indent.c -o Zend/zend_indent.lo 
Zend/zend_builtin_functions.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_builtin_functions.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_builtin_functions.c -o Zend/zend_builtin_functions.lo 
Zend/zend_sprintf.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_sprintf.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_sprintf.c -o Zend/zend_sprintf.lo 
Zend/zend_ini.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ini.c -o Zend/zend_ini.lo 
Zend/zend_qsort.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_qsort.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_qsort.c -o Zend/zend_qsort.lo 
Zend/zend_multibyte.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_multibyte.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_multibyte.c -o Zend/zend_multibyte.lo 
Zend/zend_ts_hash.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ts_hash.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_ts_hash.c -o Zend/zend_ts_hash.lo 
Zend/zend_stream.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_stream.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_stream.c -o Zend/zend_stream.lo 
Zend/zend_iterators.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_iterators.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_iterators.c -o Zend/zend_iterators.lo 
Zend/zend_interfaces.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_interfaces.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_interfaces.c -o Zend/zend_interfaces.lo 
Zend/zend_exceptions.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_exceptions.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_exceptions.c -o Zend/zend_exceptions.lo 
Zend/zend_strtod.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_strtod.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_strtod.c -o Zend/zend_strtod.lo 
Zend/zend_gc.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_gc.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_gc.c -o Zend/zend_gc.lo 
Zend/zend_closures.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_closures.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_closures.c -o Zend/zend_closures.lo 
Zend/zend_float.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_float.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_float.c -o Zend/zend_float.lo 
Zend/zend_string.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_string.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_string.c -o Zend/zend_string.lo 
Zend/zend_signal.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_signal.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_signal.c -o Zend/zend_signal.lo 
Zend/zend_generators.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_generators.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_generators.c -o Zend/zend_generators.lo 
Zend/zend_objects.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_objects.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_objects.c -o Zend/zend_objects.lo 
Zend/zend_object_handlers.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_object_handlers.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_object_handlers.c -o Zend/zend_object_handlers.lo 
Zend/zend_objects_API.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_objects_API.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_objects_API.c -o Zend/zend_objects_API.lo 
Zend/zend_default_classes.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_default_classes.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_default_classes.c -o Zend/zend_default_classes.lo 
Zend/zend_execute.lo: /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_execute.c
	$(LIBTOOL) --mode=compile $(CC)  -IZend/ -I/Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/ $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS) -c /Users/stevenjames/.phpbrew/build/php-5.5.30/Zend/zend_execute.c -o Zend/zend_execute.lo 
=======
$(DIST_ALIASES): $(DISTRIBUTE_DIR)

$(DISTRIBUTE_DIR): all py | $(DISTRIBUTE_SUBDIRS)
	# add include
	cp -r include $(DISTRIBUTE_DIR)/
	mkdir -p $(DISTRIBUTE_DIR)/include/caffe/proto
	cp $(PROTO_GEN_HEADER_SRCS) $(DISTRIBUTE_DIR)/include/caffe/proto
	# add tool and example binaries
	cp $(TOOL_BINS) $(DISTRIBUTE_DIR)/bin
	cp $(EXAMPLE_BINS) $(DISTRIBUTE_DIR)/bin
	# add libraries
	cp $(STATIC_NAME) $(DISTRIBUTE_DIR)/lib
<<<<<<< HEAD
<<<<<<< HEAD
	install -m 644 $(DYNAMIC_NAME) $(DISTRIBUTE_DIR)/lib
=======
	cp $(DYNAMIC_NAME) $(DISTRIBUTE_DIR)/lib
>>>>>>> origin/BVLC/parallel
=======
	install -m 644 $(DYNAMIC_NAME) $(DISTRIBUTE_DIR)/lib
>>>>>>> caffe
	# add python - it's not the standard way, indeed...
	cp -r python $(DISTRIBUTE_DIR)/python

-include $(DEPS)
>>>>>>> pod-caffe-pod.hpp-merge
