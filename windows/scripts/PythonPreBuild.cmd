set SOLUTION_DIR=%~1%
set PROTO_COMPILER_DIR=%~2%
set OUTPUT_DIR=%~3%

echo PythonPreBuild.cmd : Create output directories for python scripts.

if not exist "%OUTPUT_DIR%\pycaffe" mkdir "%OUTPUT_DIR%\pycaffe"
if not exist "%OUTPUT_DIR%\pycaffe\caffe" mkdir "%OUTPUT_DIR%\pycaffe\caffe"
if not exist "%OUTPUT_DIR%\pycaffe\caffe\proto" mkdir "%OUTPUT_DIR%\pycaffe\caffe\proto"

echo PythonPreBuild.cmd : Create dummy __init__.py file
rem. > "%OUTPUT_DIR%\pycaffe\caffe\proto\__init__.py"

echo PythonPreBuild.cmd : Generating src\caffe\proto\caffe.pb.h with python bindings
"%PROTO_COMPILER_DIR%\protoc" "%SOLUTION_DIR%\..\src\caffe\proto\caffe.proto" --proto_path="%SOLUTION_DIR%\..\src\caffe\proto" --python_out="%OUTPUT_DIR%\pycaffe\caffe\proto"