###############################################################################################################
#
# Intel Confidential
# This software is for Intel internal use only. Do not distribute.
# 
# Benchmarkresults.py: Generates csv and xlxs files from the benchmark log
# Author: Deepak Narasimha Murthy
# benchmark_report [no argument] will perform the following steps:
#        1.  check if benchmark_previous.csv is present; 
#               a. if not present:
#                       *       Update the caffe directory for the latest in Perforce, re-build caffe 
#                       *       Call collect_benchmark_excel.py and collect the latest benchmark results 
#                       *       Rename benchmark_results.xlsx to benchmark_previous.xlsx.
#               b. if present:
#                       *       Update the caffe directory for the latest in Perforce, re-build caffe
#                       *       Call collect_benchmark_excel.py and collect the latest benchmark results
#                       *       Call compare_benchmark benchmark_previous.csv
#                       *       Send the output of compare_benchmark, which is comparison_result.xlsx, to the subscribers by email
#                       *       Rename benchmark_results.xlsx to benchmark_previous.xlsx.
#
# Usage: From Caffe_root -> $ python ./tools/benchmark_tools/benchmark_report.py 
#
##########################################################################################################################

import os.path
import sys  
import subprocess
import os
from os import system


def update_perforce():
	"""	
	Function is used for getting the latest files from perforce:
		exports port and server details,user details and client details
		updates the directory to get the latest from perforce
		Rebuils caffe after updatation

	"""

	export_server = subprocess.call('export P4PORT=perforce01-fm.intel.com:3666',shell = True)
	export_user = subprocess.call('export P4USER=dnarasim',shell = True)
	perforce_login = subprocess.call('echo "your password" | /home/dnarasim24/Downloads/p4_command_client/p4 login',shell = True)
	export_client = subprocess.call('export P4CLIENT=dnarasim_localhost_5714',shell = True)
	sync_perforce = subprocess.call('/home/dnarasim24/Downloads/p4_command_client/p4 sync -f //gfx_GIT_Source/VU_PoC/...',shell = True)

	caffe_give_permission = subprocess.call('chmod 777 ./scripts/gencl.sh',shell = True) 
	#caffe_make = subprocess.call('make',shell = True)


def main(argv):
	"""
	Checks benchmark_previous is present or not, 
	Calls function to update perforce files, rebuilds caffe 
	Collects latest benchmark results, compares them and notifies everyone within the team 

	"""
	os.chdir("/home/dnarasim24/dnarasim_localhost_5714/gfx_GIT_Source/VU_PoC/caffe")
	#change_directory = subprocess.call('cd /home/dnarasim24/dnarasim_localhost_5714/gfx_GIT_Source/VU_PoC/caffe',shell = True)  
	if (os.path.exists("./tools/benchmark_tools/benchmark_previous.xlsx")):
		#print "previous is present"
		update_perforce() 

		remove_current = "./tools/benchmark_tools/benchmark_resultsall_iterations_50.xlsx"
		if os.path.isfile(remove_current):
			os.remove(remove_current)

		collect_latest_benchmark = subprocess.call('python ./tools/benchmark_tools/collect_benchmark_excel.py gpu',shell = True)
		benchmark_comaprison = subprocess.call('python ./tools/benchmark_tools/compare_benchmark.py benchmark_previous.xlsx',shell = True)

		remove_previous = "./tools/benchmark_tools/benchmark_previous.xlsx"
		if os.path.isfile(remove_previous):
			os.remove(remove_previous)
		copy_bench = subprocess.call('cp ./tools/benchmark_tools/benchmark_resultsall_iterations_50.xlsx ./tools/benchmark_tools/benchmark_previous.xlsx',shell = True)	

		
	else:
		#print "previous is not present"

		update_perforce()		
		run_collect = subprocess.call('python ./tools/benchmark_tools/collect_benchmark_excel.py gpu',shell = True)
		copy_bench = subprocess.call('cp ./tools/benchmark_tools/benchmark_resultsall_iterations_50.xlsx ./tools/benchmark_tools/benchmark_previous.xlsx',shell = True)
		
		remove_current = "./tools/benchmark_tools/benchmark_resultsall_iterations_50.xlsx"
		if os.path.isfile(remove_current):
			os.remove(remove_current)

if __name__=="__main__":
	main(sys.argv)