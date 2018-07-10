import sys
import os
import subprocess
import argparse
import fnmatch

parser = argparse.ArgumentParser(description='Load CUDA NVRTC required headers.')
parser.add_argument('--output_dir', action="store", dest="output_dir")
parser.add_argument('--header_files', nargs='*', action="store", dest="header_files")
parser.add_argument('--compiler', action="store", dest="compiler")
parser.add_argument('--standard_include_names', nargs='*', action="store", dest="standard_include_names")
parser.add_argument('--header_exclude_names', nargs='*', action="store", dest="header_exclude_names")
parser.add_argument('--msvc', action="store", dest="msvc")
args = parser.parse_args()

# FIXME: This only works for CLANG and GCC, add option to cover MSVC
compiler_answer = subprocess.check_output("echo | " + args.compiler + " -Wp,-v -x c++ - -fsyntax-only 2>&1", shell = True)
compiler_answer_lines = compiler_answer.splitlines()
header_search_paths = []
for line in compiler_answer_lines:
    if (line[0] == " "):
        line = line[1:]
        header_search_paths.append(line)

nvrtc_header_file_names = []
nvrtc_header_files = []
for header_file in args.header_files:
    head, tail = os.path.split(header_file)
    nvrtc_header_file_names.append(tail)
    nvrtc_header_files.append(header_file)

for standard_include_name in args.standard_include_names:
    head, tail = os.path.split(standard_include_name)
    found = False
    for search_path in header_search_paths:
        for root, dirnames, filenames in os.walk(search_path):
            for filename in fnmatch.filter(filenames, tail):
                if (not found):
                    nvrtc_header_file_names.append(standard_include_name)
                    nvrtc_header_files.append(os.path.join(root, filename))
                    found = True

def scan_includes(line):
    if '#include' in line:
        first_quote = line.find('"')
        second_quote = line.find('"', first_quote + 1)
        first_bracket = line.find('<')
        second_bracket = line.find('>', first_bracket + 1)
        
        cut_start = max(first_quote, first_bracket) + 1
        cut_end = max(second_quote, second_bracket)

        standard_include_name = line[cut_start : cut_end]
        
        if len(standard_include_name) == 0:
            return
        
        head, tail = os.path.split(standard_include_name)
        if not tail in nvrtc_header_file_names and not tail in args.header_exclude_names:
            for search_path in header_search_paths:
                for root, dirnames, filenames in os.walk(search_path):
                    for filename in fnmatch.filter(filenames, tail):
                        nvrtc_header_file_names.append(standard_include_name)
                        nvrtc_header_files.append(os.path.join(root, filename))
                        return
    


header = open(args.output_dir + '/cuda_nvrtc_headers.hpp', 'w')

header.write('// Automatically generated file, DO NOT EDIT!\n')
header.write('#include "caffe/common.hpp"\n')
header.write('namespace caffe { \n')
header.write('map<string, string> get_cuda_nvrtc_headers() {\n')
header.write('map<string, string> headers;\n')

n = 0
N = len(nvrtc_header_file_names)
while(n < N):
    nvrtc_header = open(nvrtc_header_files[n], 'r')
    header.write('{\n')
    header.write('stringstream ss;\n')
    for line in nvrtc_header:
        scan_includes(line)
        N = len(nvrtc_header_file_names)
        line = line.replace('\n', '')
        line = line.replace('\\', '\\\\')
        line = line.replace('"', '\\"');
        header.write('ss << "' + line + '" << std::endl;\n')
    header.write('headers["' + nvrtc_header_file_names[n] + '"] = ss.str();\n')
    header.write('}\n')
    n += 1

header.write('return headers;\n')
header.write('}\n')
header.write('}  // namespace caffe\n')
