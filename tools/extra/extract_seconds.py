#!/usr/bin/env python
import datetime
import os
import sys

def extract_datetime_from_line(line, year):
    # Expected format: I0210 13:39:22.381027 25210 solver.cpp:204] Iteration 100, lr = 0.00992565
    line = line.strip().split()
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
    return dt

def extract_seconds(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    log_created_time = os.path.getctime(input_file)
    log_created_year = datetime.datetime.fromtimestamp(log_created_time).year
    start_time_found = False
    out = open(output_file, 'w')
    for line in lines:
        line = line.strip()
        if not start_time_found and line.find('Solving') != -1:
            start_time_found = True
            start_datetime = extract_datetime_from_line(line, log_created_year)
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
