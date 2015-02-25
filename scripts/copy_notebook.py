#!/usr/bin/env python
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
