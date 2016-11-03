#!/usr/bin/env python
# 
# All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
Takes as arguments:
1. the path to a JSON file (such as an IPython notebook).
2. the path to output file

If 'metadata' dict in the JSON file contains 'include_in_docs': true,
then copies the file to output file, appending the 'metadata' property
as YAML front-matter, adding the field 'category' with value 'notebook'.
"""
import os
import sys
import json

filename = sys.argv[1]
output_filename = sys.argv[2]
content = json.load(open(filename))

if 'include_in_docs' in content['metadata'] and content['metadata']['include_in_docs']:
    yaml_frontmatter = ['---']
    for key, val in content['metadata'].iteritems():
        if key == 'example_name':
            key = 'title'
            if val == '':
                val = os.path.basename(filename)
        yaml_frontmatter.append('{}: {}'.format(key, val))
    yaml_frontmatter += ['category: notebook']
    yaml_frontmatter += ['original_path: ' + filename]

    with open(output_filename, 'w') as fo:
        fo.write('\n'.join(yaml_frontmatter + ['---']) + '\n')
        fo.write(open(filename).read())
