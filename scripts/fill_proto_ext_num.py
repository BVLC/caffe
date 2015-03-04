#!/usr/bin/env python

import re
import hashlib

REPLACE_REGEXP = re.compile(r'\${FIELD_NUM:(?P<start>.*?),(?P<end>.*?),(?P<key>.*?)}')
PROTOBUF_MAX_VALUE = 2 ** 29 - 1 # = 536,870,911

# Returns an integer in the range [start, end] (inclusive) based on the sha1
# hash of the input key.
def compute_hash(key, start, end):
    if start > end:
        raise Exception('Invalid range: start = %d > end = %d' % (start, end))
    output_range = end - start + 1
    hashed_key = int(hashlib.sha1(key).hexdigest(), 16)
    modded_key = hashed_key % output_range + start
    return modded_key

# Replaces wildcards of the form ${FIELD_NUM:start,end,key} with an integer
# in [start, end] (inclusive) based on the hash of key.
def replace_var_field_nums(text):
    while True:
        match = REPLACE_REGEXP.search(text)
        if match is None: return text
        start = int(match.group('start'))
        end = match.group('end')
        if end == 'max': end = PROTOBUF_MAX_VALUE
        end = int(end)
        key = match.group('key')
        hashed_key = compute_hash(key, start, end)
        text = text[:match.start()] + str(hashed_key) + text[match.end():]

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        raise Exception('Usage: %s <input filename> <output filename>' % sys.argv[0])
    _, input_filename, output_filename = sys.argv
    with open(input_filename, 'r') as input_file:
        text = input_file.read()
    text = replace_var_field_nums(text)
    with open(output_filename, 'w') as output_file:
        output_file.write(text)
