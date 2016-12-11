set SOLUTION_DIR=%~1%
set PROTO_DIR=%~2%

set INCLUDE_PROTO_DIR=%SOLUTION_DIR%..\include\caffe\proto
SET SRC_PROTO_DIR=%SOLUTION_DIR%..\src\caffe\proto
set PROTO_TEMP_DIR=%SRC_PROTO_DIR%\temp

echo ProtoCompile.cmd : Create proto temp directory "%PROTO_TEMP_DIR%"
mkdir "%PROTO_TEMP_DIR%"

echo ProtoCompile.cmd : Generating "%PROTO_TEMP_DIR%\caffe.pb.h" and "%PROTO_TEMP_DIR%\caffe.pb.cc"
"%PROTO_DIR%protoc" --proto_path="%SRC_PROTO_DIR%" --cpp_out="%PROTO_TEMP_DIR%" "%SRC_PROTO_DIR%\caffe.proto"

echo ProtoCompile.cmd : Create proto include directory
mkdir "%INCLUDE_PROTO_DIR%"

echo ProtoCompile.cmd : Compare newly compiled caffe.pb.h with existing one
fc /b "%PROTO_TEMP_DIR%\caffe.pb.h" "%INCLUDE_PROTO_DIR%\caffe.pb.h" > NUL

if errorlevel 1 (
    echo ProtoCompile.cmd : Move newly generated caffe.pb.h to "%INCLUDE_PROTO_DIR%\caffe.pb.h"
    echo ProtoCompile.cmd : and caffe.pb.cc to "%SRC_PROTO_DIR%\caffe.pb.cc"
    move /y "%PROTO_TEMP_DIR%\caffe.pb.h" "%INCLUDE_PROTO_DIR%\caffe.pb.h"
    move /y "%PROTO_TEMP_DIR%\caffe.pb.cc" "%SRC_PROTO_DIR%\caffe.pb.cc"
)

rmdir /S /Q "%PROTO_TEMP_DIR%"