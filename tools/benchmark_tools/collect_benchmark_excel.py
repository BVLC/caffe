###############################################################################################################
#
# Intel Confidential
# This software is for Intel internal use only. Do not distribute.
# 
# Benchmarkresults.py: Generates csv and xlxs files from the benchmark log
# Author: Deepak Narasimha Murthy
# Usage: 1. From Caffe_root -> $ python ./tools/benchmark_tools/collect_benchmark_excel.py 
#			All the net models will be executed
#		 2. From Caffe_root -> $ python ./tools/benchmark_tools/collect_benchmark_excel.py [iteration number]
#			All the net models will be executed and the output will have iteration number along with it 	
# 		 3. From Caffe_root -> $ python ./tools/benchmark_tools/collect_benchmark_excel.py [benchmark name] [iteration number]
#			Specified net models will be executed and the output will have iteration number along with it
#		 4. From Caffe_root -> $ python ./tools/benchmark_tools/collect_benchmark_excel.py [benchmark name]
#			Specified net models will be executed 
# Input: Benchmark name should be 'alexnet' or 'caffenet' or 'lenet'
#        number of iteration as mentioned in the .prototxt file
# output: benchmark_results_benchmarknameiterationcount.xlxs 
#	in the benchmark_tools folder
#########################################################################################################################

import re
import string
import xlsxwriter
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

	benchmark_starts = contents.find('Average time per layer:')
	benchmark_ends = contents.find('*** Benchmark ends ***')

	return contents[benchmark_starts + 24 : benchmark_ends]

def excel_update(line_layer,direction,row,col,type1,worksheet):
	"""
	Function is used to update each row of the excel sheet with layer name,
	type of pass and time taken by each layer

	Arguments:
		line_layer 	: string 	- Contains data regarding for the given layer
		direction 	: string 	- Gives information about type of pass: Forward,
					or Backward pass
		row			: integer 	- Information about which row of the excel sheet 
						has to be updated			 

		col 		: integer 	- Information about which column of the excel sheet 
						has to be updated				
		type1		: Integer	- Information about the type(Based on Nmber of columns 
															to be updated)
		worksheet   : file    	- The excel file which has to be updated					
	"""

	direction_start = line_layer.find(direction)
	time_end = line_layer.find('ms.')

	if type1 == 1:
		length = len(direction)	
		first_column = line_layer[: direction_start]
		second_column = line_layer[direction_start : direction_start + length]
		third_colum = line_layer[direction_start + length + 2 : time_end]
		worksheet.write(row,col,first_column)
		worksheet.write(row,col + 1,second_column)
		worksheet.write(row,col + 2, third_colum)
	elif type1 == 2:
		length = len(direction)
		first_column = line_layer[direction_start : direction_start + length]
		second_column = line_layer[direction_start + length + 2 : time_end]
		worksheet.write(row,col + 1,first_column)
		worksheet.write(row,col + 2,second_column)

def extract_data(worksheet,row,col,benchmark_name):

	"""
	Extracts data from the log file which was collected after running the benchmark
	and extracted data is sent as argument to excel updation

	Arguments:
		worksheet: excel files	- The file which has to be updated 
		row		 : Integer 		- The row number to be updated in excel	
		col 	 : Integer 		- The column number to be updated in excel	
		benchmark_name: String 	- Benchmark name detail

	"""

	workfile = "workfile_excel_" + benchmark_name + ".txt"
	bench_reqd = open(workfile, 'r')

	for line_layer in bench_reqd:	
		layer_starts = line_layer.find('] ')
		line_layer = line_layer[layer_starts + 2: ]
		#print line_layer

		if "forward" in line_layer:
			direction = 'forward'
			type1 = 1
			excel_update(line_layer,direction,row,col,type1,worksheet)		
			row = row + 1	
		
		elif "backward"	in line_layer:
			direction = 'backward'
			type1 = 1
			excel_update(line_layer,direction,row,col,type1,worksheet)
			row = row + 1	

		elif "Forward pass" in line_layer:
			direction = 'Forward pass'
			type1 = 2
			excel_update(line_layer,direction,row,col,type1,worksheet)
			row = row + 1	

		elif "Backward pass" in line_layer:
			direction = 'Backward pass'
			type1 = 2
			excel_update(line_layer,direction,row,col,type1,worksheet)
			row = row + 1

		elif "Total Time" in line_layer:
			direction = 'Total Time'
			type1 = 2
			excel_update(line_layer,direction,row,col,type1,worksheet)
			row = row + 1	
			
		if "Total Time" in line_layer:
			break


	bench_reqd.close()


def create_excel(benchmark_name,count,worksheet,workbook,iteration):
	"""
	Creats an excel file and number of sheets required for the an excel based on 
	uuser selection.

	Arguments:
		benchmark_name	: String 	- Benchmark name detail
		class1			: Integer 	- Specifies if a single or all the 
		 							  benchmarks are being executed
		count			: integer 	- Specifeis when the workboos has to be closed						  
		worksheet 	 	:excel sheet- The excel sheet which has to be updated
		workbook 		: excel 	- The excel file which has to be updated

	"""
	

	logfile =  "logfile_" + benchmark_name + "_" + str(iteration) +".txt"
	contents = open(logfile).read()
	workfile = "workfile_excel_" + benchmark_name + ".txt"
	bench_file = open(workfile, 'a')

	row = 0
	col = 0
	
	benchmark_contents = extract_contents(contents)
	bench_file.write(benchmark_contents)
	bench_file.close()

	 
	worksheet.write(row,col, 'Benchmark Name')
	worksheet.write(row,col, benchmark_name)
	row = row + 1

	worksheet.write(row,col,'LAYER')
	worksheet.write(row,col + 1,'DIRECTION')
	worksheet.write(row,col + 2, 'TIME in milliseconds')
	row = row + 1

	test_data = extract_data(worksheet,row,col,benchmark_name)

	if ((count == 0) or (count == 3)):
		workbook.close()
	
	remove1 = logfile
	remove2 = workfile

	if os.path.isfile(remove1):
		os.remove(remove1)
	if os.path.isfile(remove2):
		os.remove(remove2)


def main(argv):

	"""
	Takes arguments from user for benchmark name and number of iterations in the benchmark
	Script is executed to get the log file of the given benchmark
	Excel file is created and updated based on logfile

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


	excelname = path + "benchmark_results" + benchmark_n + "_iterations_" + str(iteration) +  '.xlsx'
	workbook = xlsxwriter.Workbook(excelname)
	

	if (type_mode == 'gpu'):
		if benchmark_n == 'caffenet' or benchmark_n == 'all':
			benchmark_n = 'caffenet'
			logfile_caffenet = "logfile_caffenet_" + str(iteration) + ".txt"
			benchmark_command = './build/tools/caffe time -model ./models/bvlc_reference_caffenet/train_val.prototxt -iterations ' + str(iteration) + mode + ' > logfile_caffenet_' + str(iteration) +'.txt 2>&1'
			#benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + mode + ' > logfile_caffenet.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			worksheet = workbook.add_worksheet()
			create_excel(benchmark_n,count,worksheet,workbook,iteration)
			count = count + 1
			benchmark_n = 'all'

		if benchmark_n == 'alexnet' or benchmark_n == 'all':
			benchmark_n = 'alexnet'
			logfile_alexnet = "logfile_alexnet_" + str(iteration) + ".txt"
			benchmark_command = './build/tools/caffe time -model ./models/bvlc_alexnet/train_val.prototxt -iterations ' + str(iteration) + mode +' > logfile_alexnet_' + str(iteration) +'.txt 2>&1'
			#benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + mode + ' > logfile_alexnet.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			worksheet = workbook.add_worksheet()
			create_excel(benchmark_n,count,worksheet,workbook,iteration)
			count = count + 1
			benchmark_n = 'all'

		if benchmark_n == 'lenet' or benchmark_n == 'all':
			benchmark_n = 'lenet'
			logfile_lenet = "logfile_lenet_" + str(iteration) + ".txt"
			benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + mode + ' > logfile_lenet_' + str(iteration) +'.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			worksheet = workbook.add_worksheet()
			create_excel(benchmark_n,count,worksheet,workbook,iteration)
			benchmark_n = 'all'

	else:
		
		if benchmark_n == 'caffenet' or benchmark_n == 'all':
			
			benchmark_n = 'caffenet'
			logfile_caffenet = "logfile_caffenet_" + str(iteration) + ".txt"
			benchmark_command = './build/tools/caffe time -model ./models/bvlc_reference_caffenet/train_val.prototxt -iterations ' + str(iteration) + ' > logfile_caffenet_' + str(iteration) +'.txt 2>&1'
			#benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + ' > logfile_caffenet.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			worksheet = workbook.add_worksheet()
			create_excel(benchmark_n,count,worksheet,workbook,iteration)
			count = count + 1
			benchmark_n = 'all'

		if benchmark_n == 'alexnet' or benchmark_n == 'all':
			
			benchmark_n = 'alexnet'
			logfile_alexnet = "logfile_alexnet_" + str(iteration) + ".txt"
			benchmark_command = './build/tools/caffe time -model ./models/bvlc_alexnet/train_val.prototxt -iterations ' + str(iteration) + ' > logfile_alexnet_' + str(iteration) +'.txt 2>&1'
			#benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + ' > logfile_alexnet.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			worksheet = workbook.add_worksheet()
			create_excel(benchmark_n,count,worksheet,workbook,iteration)
			count = count + 1
			benchmark_n = 'all'

		if benchmark_n == 'lenet' or benchmark_n == 'all':
			
			benchmark_n = 'lenet'
			logfile_lenet = "logfile_lenet_" + str(iteration) + ".txt"
			benchmark_command = './build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations ' + str(iteration) + ' > logfile_lenet_' + str(iteration) +'.txt 2>&1'
			run_bench = subprocess.call(benchmark_command, shell = True, preexec_fn=os.setsid)
			worksheet = workbook.add_worksheet()
			create_excel(benchmark_n,count,worksheet,workbook,iteration)		
			benchmark_n = 'all'	
	#print "not working"		

if __name__=="__main__":
	main(sys.argv)
