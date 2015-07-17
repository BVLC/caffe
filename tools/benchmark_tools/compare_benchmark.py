###############################################################################################################
#
# Intel Confidential
# This software is for Intel internal use only. Do not distribute.
# 
# compare_benchmarks.py: Generates excel file after comparing two benchmark results
# Author: Deepak Narasimha Murthy
# Usage: From Caffe_root -> $ python ./tools/benchmark_tools/compare_benchmarks.py benchmark_previous.xlsx
# 
# Input: Name of the previous benchmark_previous file.
#        
# output: Compared reults along with the results of the benchmarks in excel sheet will be saved
#		   in the benchmark_tools folder and also mailed to all users in the group
#########################################################################################################################

import xlsxwriter
import sys  
import os
import xlrd
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email import Encoders


sender = 'deepak.narasimha.murthy@intel.com'
receivers = ['deepak.narasimha.murthy@intel.com']#,'preeti.bindu@intel.com'] #'jingyi.jin@intel.com']#, 'preeti.bindu@intel.com', 'jeremy.bottleson@intel.com', 'jeff.andrews@intel.com', 'sungye.kim@intel.com']

def excel_update(curr_row,row,curr_cell,cell_value1,cell_value2,compare_worksheet,format_wrap,format_green,format_red,format_shrink):
	"""
	Function is used to update each row of the excel sheet with layer name,
	type of pass and time taken of both the benchmarks and the result after 
	comparison

	Arguments:
		curr_row	: Integer 			- information about which row of the benchmarks
		row			: integer 			- Information about which row of the excel sheet 
								  	  	  has to be updated			 
		curr_cell 	: integer 			- Information about which column of the excel sheet 
									      has to be updated				
		cell_value1 : string/integer 	- Contains data from benchmark version 1
		cell_value2 : string/integer 	- Contains data from benchmark version 2
		compare_worksheet   : file    	- The excel file which has to be updated
		format_wrap	: format type excel	- format for wrapping the text	
		format_green: format type excel	-  format for coloring the cell green
		format_red: format type excel	-  format for coloring the cell red				
	"""


	if (curr_cell == 2 and curr_row > 1):
		result = float(float(cell_value2) - float(cell_value1))
		limit_value = float(float(cell_value1)/float(100))
		if result > limit_value:
			compare_worksheet.write(row,6,result,format_red)
		elif (result < (limit_value * -1 )):
			compare_worksheet.write(row,6,result,format_green)
		else:
			compare_worksheet.write(row,6,result,format_wrap)

	compare_worksheet.write(row,curr_cell,cell_value1,format_wrap)
	compare_worksheet.write(row,curr_cell + 3,cell_value2,format_wrap)	


def benchmark_comparison(worksheet1,worksheet2,compare_worksheet,num_rows1,num_cells1,row,format_wrap,format_green,format_red,format_shrink):
	"""
	Function is used to retrieve data of each cell from both the benchmarks which 
	are being compared and transfer the data of each cell to excel_update function 

	Arguments:
		worksheet1  	: Excel file 		- data of the version1 pf the benchmark
		worksheet2  	: Excel file 		- data of the version2 pf the benchmark
		compare_worksheet   : file    	- The excel file which has to be updated
		num_rows1		: Integer 			- information about number of rows in the benchmarks
		num_cells1		: Integer 			- information about number of columns in each row of 
										  the benchmarks
		row				: integer 			- Information about which row of the excel sheet 
								  	  	  has to be updated			 		
		format_wrap		: format type excel	-  format for wrapping the text	
		format_green	: format type excel	-  format for coloring the cell green
		format_red		: format type excel	-  format for coloring the cell red	
		format_shrink	: format type excel	-  format for shrinking the cell red	
	"""


	curr_row = -1

	while curr_row < num_rows1:
		curr_row += 1
		row1 = worksheet1.row(curr_row)
		curr_cell = -1
		while curr_cell < num_cells1:
			curr_cell += 1
			cell_value1 = worksheet1.cell_value(curr_row, curr_cell)
			cell_value2 = worksheet2.cell_value(curr_row, curr_cell)
			excel_update(curr_row,row,curr_cell,cell_value1,cell_value2,compare_worksheet,format_wrap,format_green,format_red,format_shrink)

		row += 1	

def sendmail(sender,receivers,SUBJECT,excelname,comp_workbook):
	"""
	Function is used to send the results of the comparison to all the team members through 
	mail

	Arguments:
		sender: string 		- mail ID of the sender	
		receivers: string 	- mail IDs of all the receivers
		SUBJECT: string 	- Header of the mail
		excelname: string 	- variable which holds the information about the excel name
		comp_workbook: string - variable which handles the excel sheet in use

	"""

	msg = MIMEMultipart()
	msg['Subject'] = SUBJECT 
	msg['From'] = sender
	msg['To'] = ', '.join(receivers)

	part = MIMEBase('application', "octet-stream")
	part.set_payload(open(excelname, "rb").read())
	Encoders.encode_base64(part)

	part.add_header('Content-Disposition', 'attachment; filename= comparison_result.xlsx')

	msg.attach(part)

	server = smtplib.SMTP('localhost')
	server.sendmail(sender, receivers, msg.as_string())

def create_excel(worksheet1,worksheet2,num_rows1,num_cells1,bench_excel_name,comp_workbook,header1,header2):
	"""
	Creats an excel file and number of sheets required for the an excel based on 
	user selection.

	Arguments:
		worksheet1 	 	:excel sheet 	- The excel sheet which has to be compared
		worksheet2	 	:excel sheet 	- The excel sheet which has to be compared
		num_rows1		: Integer 		- information about number of rows in the benchmarks
		num_cells1		: Integer 		- information about number of columns in each row of 
										  the benchmarks
		benchmark_name	: String 		- Benchmark name detail
		bench_excel_name: string 		- Gives the name of the benchmark 							  			
		comp_workbook 	: string 		- variable which handles the excel sheet in use
		header1,header2	: string 		- Gives information of the header in the excel sheet
	"""
	
	compare_worksheet = comp_workbook.add_worksheet()
	
	""" Formats for excel writings """
	format_wrap = comp_workbook.add_format()
	format_wrap.set_text_wrap()

	format_red = comp_workbook.add_format()
	format_red.set_bg_color('red')

	format_green = comp_workbook.add_format()
	format_green.set_bg_color('green')

	format_shrink = comp_workbook.add_format()
	format_shrink.set_shrink()
	""" Formatting ends here """
	
	row = 0
	col = 0
	#if (benchmark_name == "all"):
	#	compare_worksheet.write(row,col,bench_excel_name)
	#	row = row + 1

	compare_worksheet.write(row,col,header1,format_shrink)
	compare_worksheet.write(row,col + 3, header2, format_shrink)
	compare_worksheet.write(row,col + 6, 'Results after Comparsion',format_shrink)
	
	row += 1

	benchmark_comparison(worksheet1,worksheet2,compare_worksheet,num_rows1,num_cells1,row,format_wrap,format_green,format_red,format_shrink)

	


def main(argv):

	"""
	Takes 1 argument previous results of all the benchmark
	Opens two excel files for reading benchmark data.
	One excel file for writing data after comparing the benchmark results
	Calls function for accessing data of each cell 
	Calls function to send the results of the comparison to mail all the users in group
	"""
	os.chdir("/home/dnarasim24/dnarasim_localhost_5714/gfx_GIT_Source/VU_PoC/caffe")
	
	if ((len(sys.argv) >= 3)): 
		print "Error: Please give two version number when comparing all the benchmarks "
		sys.exit(0) 

	elif((len(sys.argv) == 2)):	
		benchmark_name = "all"

	path = './tools/benchmark_tools/'

	excelname = path + "comparison_result" + '.xlsx'
	comp_workbook = xlsxwriter.Workbook(excelname)
	
	if (benchmark_name == "all"):

		excel1 = path + "benchmark_previous" + '.xlsx'
		excel2 = path + "benchmark_resultsall_iterations_50.xlsx"	

		header1 = "benchmark_previous"  + '.xlsx' 
		header2 =  "benchmark_results"  + '.xlsx'


		workbook1 = xlrd.open_workbook(excel1)
		worksheet1_1 = workbook1.sheet_by_name('Sheet1')
		worksheet1_2 = workbook1.sheet_by_name('Sheet2')
		worksheet1_3 = workbook1.sheet_by_name('Sheet3')

		workbook2 = xlrd.open_workbook(excel2)
		worksheet2_1 = workbook2.sheet_by_name('Sheet1')
		worksheet2_2 = workbook2.sheet_by_name('Sheet2')
		worksheet2_3 = workbook2.sheet_by_name('Sheet3')

		num_rows1_1 = worksheet1_1.nrows - 1
		num_rows1_2 = worksheet1_2.nrows - 1
		num_rows1_3 = worksheet1_3.nrows - 1
		num_cells1_1 = worksheet1_1.ncols - 1
		num_cells1_2 = worksheet1_2.ncols - 1
		num_cells1_3 = worksheet1_3.ncols - 1

		create_excel(worksheet1_1,worksheet2_1,num_rows1_1,num_cells1_1,"LENET",comp_workbook,header1,header2)
		create_excel(worksheet1_2,worksheet2_2,num_rows1_2,num_cells1_2,"CAFFENET",comp_workbook,header1,header2)
		create_excel(worksheet1_3,worksheet2_3,num_rows1_3,num_cells1_3,"ALEXNET",comp_workbook,header1,header2)

	comp_workbook.close()

	global sender
	global receivers 

	SUBJECT = "Comparsion Results of all the benchmarks "

	sendmail(sender,receivers,SUBJECT,excelname,comp_workbook)


if __name__=="__main__":
	main(sys.argv)