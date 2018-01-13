import subprocess
import argparse
import re
import os
import time
import json
import socket
import logging

topology_list = ["alexnet", "googlenet", "googlenet_v2", "resnet_50", 'all']
class CaffeBenchmark(object):
    '''Used to do caffe benchmarking'''
    def __init__(self, topology, bkm_batch_size, host_file = "", network = "", tcp_netmask = "", engine = "", dummy_data_use = True, caffe_bin = "", scal_test = True):
        '''params initialization'''
        self.topology = topology
        self.host_file = host_file
        self.network = network
        self.tcp_netmask = tcp_netmask
        self.dummy_data_use = dummy_data_use
        self.engine = engine
        self.caffe_bin = caffe_bin
        self.scal_test = scal_test
        self.num_nodes = 1
        self.cpu_model = "skx"
        # flag used to mark if we have detected which cpu model we're using
        self.unknown_cpu = False
        self.iterations = 100
        self.caffe_root = os.path.dirname(os.path.dirname(__file__))
        # model template path
        self.model_path = self.caffe_root + "models/intel_optimized_models"
        # specific script used to run intelcaffe 
        self.caffe_run_script = self.caffe_root + "scripts/run_intelcaffe.sh"
        self.bkm_batch_size = bkm_batch_size
        self.check_parameters()
        current_time = time.strftime("%Y%m%d%H%M%S")
        logging.basicConfig(filename = 'result-benchmark-{}.log'.format(current_time),level = logging.INFO)

    def is_supported_topology(self):
        '''check if input topology is supported'''
        if self.topology not in topology_list:
            logging.exception("The topology you specified as {} is not supported. Supported topologies are {}".format(self.topology, topology_list))
    
    def calculate_numnodes(self):
        '''calculate current using nodes'''
        if os.path.isfile(self.host_file):
            with open(self.host_file, 'r') as f:
                self.num_nodes = len([line for line in f.readlines() if line.strip() != ''])
                if self.num_nodes == 0:
                    logging.exception("Error: empty host list. Exit.")
        return self.num_nodes
    
    def _exec_command(self, cmd):
        '''execute shell command'''
        return subprocess.check_output(cmd, stderr = subprocess.STDOUT, shell = True)

    def _exec_command_and_show(self, cmd):
        '''execute shell command and print it out'''
        def _exec_command_and_iter_show(cmd):
            out = subprocess.Popen(cmd, shell = True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE, universal_newlines = True)
            for stdout_line in iter(out.stdout.readline, ""):
                yield stdout_line
            return_code = out.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, cmd)
        for line in _exec_command_and_iter_show(cmd):
            print line
        
    
    def detect_cpu(self):
        '''check which IA platform currently using'''
        command_name = "lscpu | grep 'Model name' | awk -F ':' '{print $2}'"
        model_string = self._exec_command(command_name)
        self.model_string = model_string
        #will make it configurable in the future
        knl_pattern = re.compile(".*72[1359]0.*")
        knm_pattern = re.compile(".*72.*")
        skx_pattern = re.compile(".*[86543]1.*")
        bdw_pattern = re.compile(".*(E5-[421]6|E7-[84]8|E3-12|D-?15).*")
        if re.match(knl_pattern, model_string):
            self.cpu_model = "knl"
        elif re.match(knm_pattern, model_string):
            self.cpu_model = "knm"
        elif re.match(skx_pattern, model_string):
            self.cpu_model = "skx"
        elif re.match(bdw_pattern, model_string):
            self.cpu_model = "bdw"
        else:
            self.unknown_cpu = True
            logging.info("Can't detect which cpu model currently using, will use default settings, which may not be the optimal one.")
    
    def obtain_model_file(self, model):
        '''change original model file batch size to bkm batch size'''
        prototxt_file = "train_val_dummydata.prototxt" if self.dummy_data_use else "train_val.prototxt"
        dst_model_file = "/".join([self.model_path, model, '-'.join([self.cpu_model, prototxt_file])])
        if os.path.isfile(dst_model_file):
            os.remove(dst_model_file)
        src_model_file = "/".join([self.model_path, model, prototxt_file])
        if not os.path.isfile(src_model_file):
            logging.exception("template model file {} doesn't exist.".format(src_model_file))
        batch_size_pattern = re.compile(".*shape:.*") if self.dummy_data_use else re.compile("^\s+batch_size:.*")
        # we only care about train phase batch size for benchmarking
        batch_size_cnt = 2 if self.dummy_data_use else 1
        if model not in self.bkm_batch_size or self.cpu_model not in self.bkm_batch_size[model]:
            logging.exception("Can't find batch size of topology {} and cpu model {} within batch size table".format(model, self.cpu_model))
        new_batch_size = self.bkm_batch_size[model][self.cpu_model]
        with open(src_model_file, 'r') as src_f, open(dst_model_file, 'w') as dst_f:
            cnt = 0
            for line in src_f.readlines():
                if re.match(batch_size_pattern, line) and cnt < batch_size_cnt:
                   #change batch size
                    line = re.sub("[0-9]+", new_batch_size, line, count = 1)
                    cnt += 1
                dst_f.write(line) 
        return dst_model_file

    def obtain_solver_file(self, model):
        '''obtain suitable solver file for training benchmark'''
        solver_prototxt_file = "solver_dummydata.prototxt" if self.dummy_data_use else "solver.prototxt"
        dst_solver_file = "/".join([self.model_path, model, '-'.join([self.cpu_model, solver_prototxt_file])])
        if os.path.isfile(dst_solver_file):
            os.remove(dst_solver_file)
        src_solver_file = "/".join([self.model_path, model, solver_prototxt_file])
        if not os.path.isfile(src_solver_file):
            logging.exception("template solver file {} doesn't exist.".format(src_solver_file))
        dst_model_file = self.obtain_model_file(model)
        max_iter = "200"
        display_iter = "1"
        net_path_pattern = re.compile(".*net:.*")
        max_iter_pattern = re.compile(".*max_iter:.*")
        display_pattern = re.compile(".*display:.*")
        with open(src_solver_file, 'r') as src_f, open(dst_solver_file, 'w') as dst_f:
            for line in src_f.readlines():
                if re.match(net_path_pattern, line):
                    dst_f.write('net: "{}"\n'.format(dst_model_file))
                elif re.match(max_iter_pattern, line):
                    dst_f.write('max_iter: {}\n'.format(max_iter))
                elif re.match(display_pattern, line):
                    dst_f.write('display: {}\n'.format(display_iter))
                else:
                    dst_f.write(line) 
        return dst_solver_file

    def run_specific_model(self, model):
        '''run the topology you specified'''
        self.calculate_numnodes()
        if self.num_nodes == 1:
            model_file = self.obtain_model_file(model)
            exec_command = ' '.join([self.caffe_run_script, '--model_file', model_file, '--mode time', '--iteration', str(self.iterations), '--benchmark none'])
        else:
            solver_file = self.obtain_solver_file(model)
            exec_command = ' '.join([self.caffe_run_script, '--hostfile', self.host_file, '--solver', solver_file, '--network', self.network, '--benchmark none'])
            if self.network == "tcp":
                exec_command += " --netmask {}".format(self.tcp_netmask)
    
        if self.engine != "":
            exec_command += " --engine {}".format(self.engine)
        if self.caffe_bin != "":
            exec_command += " --caffe_bin {}".format(self.caffe_bin)
        current_time = time.strftime("%Y%m%d%H%M%S")
        if not self.unknown_cpu:
            self.result_log_file = "-".join(["result", self.cpu_model, model, current_time + ".log"])
        else:
            self.result_log_file = "-".join(["result", "unknown", model, current_time + ".log"])
        exec_command += " 2>&1 | tee {}".format(self.result_log_file)
        self._exec_command_and_show(exec_command)
        self.intelcaffe_log = self.obtain_intelcaffe_log()
        self.calculate_fps()
    
    def obtain_intelcaffe_log(self):
        '''obtain the logfile of 'run_intelcaffe' '''
        logging.info("Result log file: {}".format(self.result_log_file))
        if not os.path.isfile(self.result_log_file):
            logging.exception("Couldn't see result log file {}".format(result_log_file))
        result_dir = ''
        with open(self.result_log_file, 'r') as f:
            for line in f.readlines():
                if line.startswith('Result folder:'):
                    result_dir = line.split('/')[-1].strip()
                    break
        if result_dir == "":
            logging.exception("Couldn't find result folder within file".format(result_file_log))
        if not self.unknown_cpu:
            caffe_log_file = "-".join(["outputCluster", self.cpu_model, str(self.num_nodes) + '.txt'])
        else:
            caffe_log_file = "-".join(["outputCluster", "unknown", str(self.num_nodes) + '.txt'])
        intelcaffe_log = "/".join([result_dir, caffe_log_file])
        logging.info('intelcaffe log: %s' % intelcaffe_log)
        return intelcaffe_log
    
    def obtain_average_fwd_bwd_time(self):
        '''obtain average iteration time of training'''
        result_file = self.intelcaffe_log
        if not os.path.isfile(result_file):
            logging.exception("Error: result file {} does not exist...".format(result_file))
        if self.num_nodes == 1:
            average_time = ""
            with open(result_file, 'r') as f:
                average_fwd_bwd_time_pattern = re.compile(".*Average Forward-Backward:.*")
                for line in f.readlines():
                    if re.match(average_fwd_bwd_time_pattern, line):
                        average_time = line.split()[-2]
            if average_time == "": 
                logging.exception("Error: can't find average forward-backward time within logs, please check logs under: {}".format(result_file))
            average_time = float(average_time)
        else:
            start_iteration = 100
            iteration_num = 100
            total_time = 0.0
            delta_times = []
            with open(result_file, 'r') as f:
                delta_time_pattern = re.compile(".*DELTA TIME.*")
                for line in f.readlines():
                    if re.match(delta_time_pattern, line):
                        delta_times.append(line.split()[-2])
            if len(delta_times) == 0:
                logging.exception("Error: check if you mark 'CAFFE_PER_LAYER_TIMINGS := 1' while building caffe; also ensure you're running at least 200 iterations for multinode trainings; or check if you're running intelcaffe failed, the logs can be found under: {}".format(result_file))
            for delta_time in delta_times[start_iteration : start_iteration + iteration_num]:
                total_time += float(delta_time)
            average_time = total_time / iteration_num * 1000.0
        logging.info("average time: {} ms".format(str(average_time)))
        return average_time
    
    def obtain_batch_size(self):
        '''obtain global batch size for training'''
        log_file = self.intelcaffe_log
        if not os.path.isfile(log_file):
            logging.exception("Error: log file {} does not exist...".format(log_file))
        with open(log_file, 'r') as f:
            batch_size_pattern_time = re.compile(".*SetMinibatchSize.*")
            batch_size_pattern_dummy = re.compile(".*dim:.*")
            batch_size_pattern_real = re.compile("^\s+batch_size:.*")
            batch_size = ''
            for line in f.readlines():
                if re.match(batch_size_pattern_time, line) or re.match(batch_size_pattern_real, line) or re.match(batch_size_pattern_dummy, line):
                    batch_size = line.split()[-1]
                    break
        if batch_size == '':
            logging.exception("Can't find batch size within your log file: {}".format(log_file))
        batch_size = int(batch_size) * self.num_nodes
        logging.info("global batch size: {}".format(str(batch_size)))
        return float(batch_size)
    
    def calculate_fps(self):
        '''calculate fps here'''
        self.batch_size = self.obtain_batch_size()
        self.average_time = self.obtain_average_fwd_bwd_time()
        speed = self.batch_size * 1000.0 / self.average_time
        self.speed = int(speed)
        logging.info("benchmark speed: {} images/sec".format(str(self.speed)))
        return speed
    
    def get_local_ip_lists(self):
        '''get local ip lists'''
        exec_command = 'ip addr'
        out = self._exec_command(exec_command)
        ip_pattern = re.compile(".*inet [0-9]+.*")
        self.local_ips = []
        for line in out.split('\n'):
            if re.match(ip_pattern, line):
                ip = line.split()[1].split('/')[0]
                self.local_ips.append(ip)
        if len(self.local_ips) == 0:
            logging.exception("Can't find available ips on local node.")
        hostname = socket.gethostname()
        self.local_ips.append(hostname)

    def manipulate_host_file(self):
        '''put master node ip or hostname on the first one of the host ip or hostname list'''
        self.get_local_ip_lists()
        self.hosts = []
        with open(self.host_file, 'r') as origin_f:
            for line in origin_f.readlines():
                self.hosts.append(line.rstrip().lstrip())
        for index, ip in enumerate(self.hosts):
            if ip in self.local_ips:
                self.hosts[0], self.hosts[index] = self.hosts[index], self.hosts[0]
                break
        
        
    def obtain_host_file(self, num_nodes):
        '''obtain suitable host file to do scaling test'''
        dst_host_file = 'scal_hostfile'
        with open(dst_host_file, 'w') as dst_f:
            for i in xrange(num_nodes):
                dst_f.write(self.hosts[i] + '\n')
        return dst_host_file 
 
    def run_scal_test(self, model):
        '''scaling test on multinodes'''
        num_nodes = self.calculate_numnodes()
        if num_nodes <= 1 or ((num_nodes & (num_nodes - 1))) != 0:
            logging.exception("nodes number: {} is not a power of 2.".format(num_nodes))
        self.manipulate_host_file()
        origin_hostfile = self.host_file
        fps_table = {}
        while num_nodes > 0:
            self.host_file = self.obtain_host_file(num_nodes)
            self.run_specific_model(model)
            fps_table[num_nodes] = self.speed
            num_nodes /= 2
        # roll back actual num_nodes for possible topology 'all'
        os.remove(self.host_file)
        self.host_file = origin_hostfile
        self.print_scal_test_results(fps_table)

    def print_scal_test_results(self, fps_table):
        '''print scaling test results out'''
        logging.info('')
        logging.info('-------scaling test results----------')
        logging.info('num_nodes, fps(images/s), scaling efficiency')
        num_nodes, total_num_nodes = 1, self.calculate_numnodes()
        while num_nodes <= total_num_nodes:
            scal_efficiency = round(float(fps_table[num_nodes]) / float((num_nodes * fps_table[1])), 3)
            logging.info('{}, {}, {}'.format(str(num_nodes), str(fps_table[num_nodes]), str(scal_efficiency)))
            num_nodes *= 2
        logging.info('')
        
    def run_benchmark(self):
        '''run intelcaffe training benchmark'''
        self.detect_cpu()
        logging.info("Cpu model: {}".format(self.model_string))
        if self.topology == 'all':
            for model in topology_list:
                if model == 'all':
                    continue
                logging.info("--{}".format(model))
                if self.scal_test:
                    self.run_scal_test(model)
                else:
                    self.run_specific_model(model)
        else:
            if self.scal_test:
                self.run_scal_test(self.topology)
            else:
                self.run_specific_model(self.topology)

    def check_parameters(self):
        '''check program parameters'''
        if self.topology == "":
            logging.exception("Error: topology is not specified.")
        self.is_supported_topology()
        if self.host_file != "":
            if self.network == "tcp" and self.tcp_netmask == "":
                logging.exception("Error: need to specify tcp network's netmask")

def main():
    '''main routine'''
    arg_parser = argparse.ArgumentParser(description = "Used to run intelcaffe training benchmark")
    arg_parser.add_argument('--configfile', help = "config file which contains the parameters you want and the batch size you want to use on all topologies and platforms")
    main_args = arg_parser.parse_args()
    if main_args.configfile == '' or not os.path.isfile(main_args.configfile):
        logging.exception("Cant't find config file {}.".format(main_args.configfile))
    with open(main_args.configfile, 'r') as f:
        config = json.load(f)
        topology = config['params']['topology']
        host_file = config['params']['hostfile']
        network = config['params']['network']
        tcp_netmask = config['params']['netmask']
        engine = config['params']["engine"]
        dummy_data_use = config['params']['dummy_data_use']
        scal_test = config['params']['scal_test']
        caffe_bin = config['params']['caffe_bin']
        bkm_batch_size = config['batch_size_table']
    
    caffe_benchmark = CaffeBenchmark(topology, bkm_batch_size, host_file, network, tcp_netmask, engine, dummy_data_use, caffe_bin, scal_test)
    caffe_benchmark.run_benchmark()

if __name__ == "__main__":
    main()
