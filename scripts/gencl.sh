#!/bin/sh
# Run the C preprocessor on an OpenCL kernel to generate a C string array
# suitable for clCreateProgramWithSource.  This allows us to create
# standalone OpenCL programs that do not depend on paths to the source
# tree (the programs will still run the OpenCL run-time compiler to
# compile the kernel, but the kernel is a string within the program, with
# no external include dependencies)
# Mark Moraes, D. E. Shaw Research

# indenting the cpp output makes errors from the OpenCL runtime compiler
# much more understandable.  User can override with whatever they want.
# The classic BSD indent (yes, the one that lived in /usr/ucb/indent once)
# defaults to -br, but recent GNU indent versions do not.  Both appear to
# accept -br, fortunately... (BSD indent does not accept -kr or -linux, alas)

PATH=$PATH:/usr/bin
export PATH
if type indent > /dev/null 2>&1; then
	: ${GENCL_INDENT="indent -br"}
else
	: ${GENCL_INDENT=cat}
fi

# We rely on gsub in awk, which exists in everything except classic
# old V7 awk (Solaris!).  If we can find gawk or nawk, we prefer those.
# http://www.shelldorado.com/articles/awkcompat.html
for f in gawk nawk awk; do
	if type "$f" > /dev/null 2>&1; then
		: ${GENCL_AWK="$f"}
		break
	fi
done
case "${GENCL_AWK}" in
'')	echo "$0: could not find awk!">&2; exit 1;;
esac
usage="Usage: $0 inputoclfilename outputfilename"
case $# in
2)	;;
*)	echo "$usage" >&2; exit 1;;
esac
case "$1" in
''|-*)	echo "Invalid or empty inputoclfilename: $1
$usage" >&2; exit 1;;
esac
set -e
${CC-cc} -xc -E -P -nostdinc -D__OPENCL_VERSION__=1 $CPPFLAGS "$1" | 
	${GENCL_INDENT} | 
	${GENCL_AWK} 'BEGIN {}
	{gsub("\\", "\\\\", $0); gsub("\"", "\\\"", $0); print $0}
	END {}' > "$2"
