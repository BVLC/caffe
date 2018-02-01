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

from extract_seconds import extract_datetime_from_line, get_log_created_year

class time_DFA:
    def __init__(self,log_path):
        self.filepath = log_path
        self.test_time = []
        self.snapshot_time = []
        self.total_test_time = 0.0
        self.total_snapshot_time = 0.0
        self.total_time = 0.0
        self.log_year = get_log_created_year(log_path)

    def parse_time(self):
        f = open(self.filepath,'r')
        line = ''
        while True:
            self.prev_line = line
            line = f.readline()

            if not line:
                if self.total_time == 0 and begin_time and self.prev_line:
                    quit_time = extract_datetime_from_line(self.prev_line, self.log_year)
                    self.total_time = (quit_time - begin_time).total_seconds()
                break

            if  '] Solving' in line:
                begin_time = extract_datetime_from_line(line, self.log_year)

            if '] Optimization Done.' in line:
                quit_time = extract_datetime_from_line(line, self.log_year)
                self.total_time = (quit_time-begin_time).total_seconds()
                break

            if '] Iteration' in line and 'lr' in line:
                self.train_state(f)

            elif 'Testing net ' in line:
                start_time = extract_datetime_from_line(line, self.log_year)
                iter_num = line.split(' ')[-4]
                end_time = self.test_state(f)
                self.test_time.append({'config':iter_num,'time':(end_time-start_time).total_seconds()})
                self.total_test_time += (end_time-start_time).total_seconds()

            elif '] Snapshotting to binary proto' in line:
                start_time = extract_datetime_from_line(self.prev_line, self.log_year)
                end_time = self.snapshot_state(f)
                if end_time:
                    self.total_snapshot_time += (end_time - start_time).total_seconds()

            elif '] Snapshot begin' in line:
                start_time = extract_datetime_from_line(line, self.log_year)
                end_time = self.accurate_snapshot_state(f)
                if end_time:
                    self.total_snapshot_time += (end_time - start_time).total_seconds()

    def test_state(self, f):
        line = f.readline()
        while line:
            if 'Test net output' not in line and 'Test net output' in self.prev_line:
                return extract_datetime_from_line(self.prev_line, self.log_year)
            else:
                self.prev_line = line
                line = f.readline()
        return extract_datetime_from_line(self.prev_line, self.log_year)

    def train_state(self,f):
        pass

    def snapshot_state(self,f):
        line = f.readline()
        while line:
            if '] Snapshotting solver state' in line:
                return extract_datetime_from_line(line, self.log_year)
            else:
                self.prev_line = line
                line = f.readline()
        return extract_datetime_from_line(self.prev_line, self.log_year)

    def accurate_snapshot_state(self,f):
        line = f.readline()
        while line:
            if '] Snapshot end' in line:
                return extract_datetime_from_line(line, self.log_year)
            else:
                self.prev_line = line
                line = f.readline()
        return extract_datetime_from_line(self.prev_line, self.log_year)
