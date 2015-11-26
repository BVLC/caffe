###############################################################################################################
#
# Intel Confidential
# This software is for Intel internal use only. Do not distribute.
# 
# Benchmarkresults.py: Generates csv and xlxs files from the benchmark log
# Author: Deepak Narasimha Murthy
# Usage: 1. From Caffe_root -> $ python ./tools/benchmark_tools/collect_benchmark.py 
#               All the net models will be executed
#        2. From Caffe_root -> $ python ./tools/benchmark_tools/collect_benchmark.py [iteration number]
#               All the net models will be executed and the output will have iteration number along with it     
#        3. From Caffe_root -> $ python ./tools/benchmark_tools/collect_benchmark.py [benchmark name] [iteration number]
#               Specified net models will be executed and the output will have iteration number along with it
#        4. From Caffe_root -> $ python ./tools/benchmark_tools/collect_benchmark.py [benchmark name]
#               Specified net model will be executed]
#
# 
# Input: Benchmark name should be 'alexnet' or 'caffenet' or 'lenet'
#        version = number of iteration as mentioned in the .prototxt file or as per user requirement
# output: benchmark_result_"benchmarkname""iteration".csv
#	in the benchmark_tools folder
#########################################################################################################################

import re
import string
import subprocess
import sys  
import os
from os import system


def extract_contents(contents):
	"""
	Time taken for all the layers in forward Pass, backward pass along with 
	total time for the benchmark is extracted from the contents of benchmark
	logfile 

	Arguments:
		contents : String - The raw log file from benchmark which contains 
		data from initialization, layer details, creation of different layers, 
		Computations required for different layers, Memory requirement for the
		data and performance of the benchmark
	"""

	benchmark_starts = contents.find('Average time per layer: ')
	benchmark_ends = contents.find('*** Benchmark ends ***')

	return contents[benchmark_starts + 24 : benchmark_ends]

def csv_update(line_layer,direction,type1,csvsheet):
	"""
	Function is used to update each row of the csv sheet with layer name,
	type of pass and time taken by each layer

	Arguments:
		line_layer 	: string 	- Contains data regarding for the given layer
		direction 	: string 	- Gives information about type of pass: Forward,
					or Backward pass
				
		type1		: Integer	- Information about the type(Based on Nmber of columns 
															to be updated)
		worksheet   : file    	- The excel file which has to be updated					
	"""

	direction_start = line_layer.find(direction)
	time_end = line_layer.find('ms.')
	data = []

	if type1 == 1:
		length = len(direction)	
		first_column = line_layer[: direction_start]
		second_column = line_layer[direction_start : direction_start + length]
		third_colum = line_layer[direction_start + length + 2 : time_end]
		csvsheet.write(first_column + ', ' + second_column + ', ' + third_colum + '\n')	
			
	elif type1 == 2:
		length = len(direction)
		first_column = line_layer[direction_start : direction_start + length]
		second_column = line_layer[direction_start + length + 2 : time_end]
		csvsheet.write(first_column + ', ' + second_column + '\n')		

def extract_data(csvsheet,benchmark_name): 
	"""
	Extracts the data which is required for filling each cell in the csvsheet
	whose data is temporarily stored in "workfile.txt"

	Arguments:
		csvsheet: file 	- CSV file which is required for updating the data 
		benchmark_name  : string 	- Gives infomration about the benchmark
	"""

	#bench_reqd = open('workfile.txt', 'r')
	workfile = "workfile_" + benchmark_name + ".txt"
	bench_reqd = open(workfile, 'r')
	for line_layer in bench_reqd:	
		layer_starts = line_layer.find('] ')
		line_layer = line_layer[layer_starts + 2: ]
		#print line_layer

		if "forward" in line_layer:
			direction = 'forward'
			type1 = 1
			csv_update(line_layer,direction,type1,csvsheet)			
		
		elif "backward"	in line_layer:
			direction = 'backward'
			type1 = 1
			csv_update(line_layer,direction,type1,csvsheet)	

		elif "Forward pass" in line_layer:
			direction = 'Forward pass'
			type1 = 2
			csv_update(line_layer,direction,type1,csvsheet)	

		elif "Backward pass" in line_layer:
			direction = 'Backward pass'
			type1 = 2
			csv_update(line_layer,direction,type1,csvsheet)

		elif "Total Time" in line_layer:
			direction = 'Total Time'
			type1 = 2
			csv_update(line_layer,direction,type1,csvsheet)	
			
		if "Total Time" in line_layer:
			break

	bench_reqd.close()

def create_csv(benchmark_name,class1,count,csvsheet,iteration):
	"""
	Creates CSV file which is also the output file.

	Arguments: 	
		benchmark_name  : string 	- Gives infomration about the benchmark 
		class1 			: Integer	- Informs how many benchmarks are being executed
		count 			: Integer	- Informs if csv sheet has to be closed or not						  
		csvsheet 	 	:csv sheet	- The csv sheet which has to be updated
	"""
	

	logfile = "logfile_" + benchmark_name + "_" + iteration+".txt"
	contents = open(logfile).read()
	workfile = "workfile_" + benchmark_name +".txt"
	bench_file = open(workfile, 'a')
	

	if (class1 == 0):
		csvsheet.write('Layer, Direction, Time_in_milliseconds \n')
		csvsheet.write("Benchmark Name " + benchmark_name +  '\n')
	elif((class1 == 1) or (class1 == 2) or (class1 == 3)):
		csvsheet.write("Benchmark Name " + benchmark_name +  '\n')		 	
	
	benchmark_contents = extract_contents(contents)
	bench_file.write(benchmark_contents)
	bench_file.close()

	test_data = extract_data(csvsheet,benchmark_name)

	if ((count == 0) or (count == 3)):
		csvsheet.close()
	

	remove1 = logfile
	remove2 = workfile
	remove_log = './tools/benchmark_tools/' + logfile
	
	if os.path.isfile(remove_log):
		os.remove(remove_log)

	if os.path.isfile(remove1):
		benchmark_command = 'cp ' + logfile + ' ./tools/benchmark_tools/' + logfile 
		copy_log = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
		os.remove(remove1)
	if os.path.isfile(remove2):
		os.remove(remove2)


def main(argv):

	"""
	Takes arguments from user for benchmark name and 
	Script is executed to get the log file of the given benchmark
	CSV files are created and updated based on logfile,number 
	of benchmarks required

	"""
	path = './tools/benchmark_tools/'
	benchmark_n = 'all'
	iteration = 50
	count = 1
	mode = ' -gpu 0 '
	type_mode = ''

	if ((len(sys.argv) >= 5)):
		print "Error: Please give one benchmark name and number of iterations for running particular benchmark"
		sys.exit(0) 

	if (len(sys.argv) == 4):
		benchmark_n = sys.argv[1]
		iteration = sys.argv[2]
		type_mode = sys.argv[3]


	if (len(sys.argv) == 3):
		temp = str(sys.argv[1])
		type_mode = sys.argv[2]
		if (temp.isdigit()):
			iteration = temp
		else:
			benchmark_n = temp

	if (len(sys.argv) == 2):
		temp = str(sys.argv[1])
		if (temp.isdigit()):
			iteration = temp
		else:
			if (temp == 'gpu'):
				type_mode = temp
			else:
				benchmark_n = temp


	csvname = path + "benchmark_results_" + str(iteration) + ".csv"
	csvsheet = open(csvname, 'w')

	if (type_mode == 'gpu'):
		if benchmark_n == 'caffenet' or benchmark_n == 'all':
			logfile_caffenet = "logfile_caffenet_" + iteration + ".txt"
			benchmark_command = './build/tools/caffe time -model ./models/bvlc_reference_caffenet/train_val.prototxt -iterations ' + str(iteration) + mode + ' > logfile_caffenet_' + iteration +'.txt 2>&1'
			#benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + mode + ' > logfile_caffenet.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			create_csv("caffenet", count, count, csvsheet,iteration)
			count = count + 1

		if benchmark_n == 'alexnet' or benchmark_n == 'all':
			logfile_alexnet = "logfile_alexnet_" + iteration + ".txt"
			benchmark_command = './build/tools/caffe time -model ./models/bvlc_alexnet/train_val.prototxt -iterations ' + str(iteration) + mode +' > logfile_alexnet_' + iteration +'.txt 2>&1'
			#benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + mode + ' > logfile_alexnet.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			create_csv("alexnet", count, count, csvsheet,iteration)
			count = count + 1

		if benchmark_n == 'lenet' or benchmark_n == 'all':
			logfile_lenet = "logfile_lenet_" + iteration + ".txt"
			benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + mode + ' > logfile_lenet_' + iteration +'.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			create_csv("lenet", count, count, csvsheet,iteration)


	else:
		
		if benchmark_n == 'caffenet' or benchmark_n == 'all':
			logfile_caffenet = "logfile_caffenet_" + iteration + ".txt"
			benchmark_command = './build/tools/caffe time -model ./models/bvlc_reference_caffenet/train_val.prototxt -iterations ' + str(iteration) + ' > logfile_caffenet_' + iteration +'.txt 2>&1'
			#benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + ' > logfile_caffenet.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			create_csv("caffenet", count, count, csvsheet,iteration)
			count = count + 1

		if benchmark_n == 'alexnet' or benchmark_n == 'all':
			logfile_alexnet = "logfile_alexnet_" + iteration + ".txt"
			benchmark_command = './build/tools/caffe time -model ./models/bvlc_alexnet/train_val.prototxt -iterations ' + str(iteration) + ' > logfile_alexnet_' + iteration +'.txt 2>&1'
			#benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + ' > logfile_alexnet.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			create_csv("alexnet", count, count, csvsheet,iteration)
			count = count + 1

		if benchmark_n == 'lenet' or benchmark_n == 'all':
			logfile_lenet = "logfile_lenet_" + iteration + ".txt"
			benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + ' > logfile_lenet_' + iteration +'.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			create_csv("lenet", count, count, csvsheet,iteration)

if __name__=="__main__":
	main(sys.argv)
