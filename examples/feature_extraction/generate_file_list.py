#!/usr/bin/env python
import os
import sys

def help():
    print 'Usage: ./generate_file_list.py file_dir file_list.txt'
    exit(1)

def main():
    if len(sys.argv) < 3:
        help()
    file_dir = sys.argv[1]
    file_list_txt = sys.argv[2]
    if not os.path.exists(file_dir):
        print 'Error: file dir does not exist ', file_dir
        exit(1)
    file_dir = os.path.abspath(file_dir) + '/'
    with open(file_list_txt, 'w') as output:
        for root, dirs, files in os.walk(file_dir):
            for name in files:
                file_path = file_path.replace(os.path.join(root, name), '')
                output.write(file_path + '\n')                

if __name__ == '__main__':
    main()
