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
import datetime
import os
import sys

def extract_datetime_from_line(line, year):
    # Expected format: I0210 13:39:22.381027 25210 solver.cpp:204] Iteration 100, lr = 0.00992565
    raw_line = line
    line = line.strip().split()
    try:
        month = int(line[0][1:3])
        day = int(line[0][3:])
        timestamp = line[1]
        pos = timestamp.rfind('.')
        ts = [int(x) for x in timestamp[:pos].split(':')]
        hour = ts[0]
        minute = ts[1]
        second = ts[2]
        microsecond = int(timestamp[pos + 1:])
        dt = datetime.datetime(year, month, day, hour, minute, second, microsecond)
    except ValueError:
        print "Unexpected error when parsing line:"
        print raw_line
        sys.exit(1)
    return dt


def get_log_created_year(input_file):
    """Get year from log file system timestamp
    """

    log_created_time = os.path.getctime(input_file)
    log_created_year = datetime.datetime.fromtimestamp(log_created_time).year
    return log_created_year


def get_start_time(line_iterable, year):
    """Find start time from group of lines
    """

    start_datetime = None
    for line in line_iterable:
        line = line.strip()
        if line.find('Solving') != -1:
            start_datetime = extract_datetime_from_line(line, year)
            break
    return start_datetime


def extract_seconds(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    log_created_year = get_log_created_year(input_file)
    start_datetime = get_start_time(lines, log_created_year)
    assert start_datetime, 'Start time not found'

    out = open(output_file, 'w')
    for line in lines:
        line = line.strip()
        if line.find('Iteration') != -1:
            dt = extract_datetime_from_line(line, log_created_year)
            elapsed_seconds = (dt - start_datetime).total_seconds()
            out.write('%f\n' % elapsed_seconds)
    out.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ./extract_seconds input_file output_file')
        exit(1)
    extract_seconds(sys.argv[1], sys.argv[2])
