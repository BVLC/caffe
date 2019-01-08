import subprocess
import argparse
import re
import os
import time
import json
import socket
import logging

class CaffeBenchmark(object):
    '''Used to do caffe benchmarking'''
    def __init__(self, bench_params):
        '''params initialization'''
        self.topology = bench_params.topology
        self.host_file = bench_params.host_file
        self.network = bench_params.network
        self.tcp_netmask = bench_params.tcp_netmask
        self.dummy_data_use = bench_params.dummy_data_use
        self.engine = bench_params.engine
        self.caffe_bin = bench_params.caffe_bin
        self.test_mode = bench_params.test_mode
        self.inf_instances = bench_params.inf_instances
        self.num_omp_threads = bench_params.num_omp_threads
        self.num_nodes = 1
        self.cpu_model = 'skx'
        # flag used to mark if we have detected which cpu model we're using
        self.unknown_cpu = False
        self.iterations = 100 
        self.caffe_root = os.path.dirname(os.path.dirname(__file__))
        # model template path
        self.model_path = os.path.join(self.caffe_root, "models/intel_optimized_models")
        # specific script used to run intelcaffe 
        self.caffe_run_script = os.path.join(self.caffe_root, "scripts/run_intelcaffe.sh")
        self.train_bkm_batch_size = bench_params.train_bkm_batch_size
        self.inf_bkm_batch_size = bench_params.inf_bkm_batch_size
        self.support_topologies = self.train_bkm_batch_size.keys()
        self.support_inf_topologies = self.inf_bkm_batch_size.keys()
        self.support_topologies.append('all_train')
        self.support_inf_topologies.append('all_inf')
        self.check_parameters()
        current_time = time.strftime("%Y%m%d%H%M%S")
        logging.basicConfig(filename = 'result-benchmark-{}.log'.format(current_time),level = logging.INFO)

    def is_supported_topology(self):
        '''check if input topology is supported'''
        if self.test_mode == "train_throughput" or self.test_mode == "scal_test" :
            if self.topology not in self.support_topologies:
                logging.exception("The topology you specified as {} is not supported. Supported topologies are {}".format(self.topology, self.support_topologies))
        else:
            if self.topology not in self.support_inf_topologies:
                logging.exception("The topology you specified as {} is not supported. Supported topologies are {}".format(self.topology, self.support_inf_topologies))
    
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
            print(line)
        
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
        clx_pattern = re.compile(".*Genuine.*")
        if re.match(knl_pattern, model_string):
            self.cpu_model = "knl"
        elif re.match(knm_pattern, model_string):
            self.cpu_model = "knm"
        elif re.match(skx_pattern, model_string):
            self.cpu_model = "skx"
        elif re.match(bdw_pattern, model_string):
            self.cpu_model = "bdw"
        elif re.match(clx_pattern, model_string):
            self.cpu_model = "clx"
        else:
            self.unknown_cpu = True
            logging.info("Can't detect which cpu model currently using, will use default settings, which may not be the optimal one.")
    
    def gen_model_file(self, model):
        '''generate model file with new batch size which equal to bkm batch size'''
        if model.find("_int8") != -1: 
            if not self.test_mode == "inf_throughput" and not self.test_mode == "inf_latency":
                print("Error: int8 test is only for inference")
                return ""
                  
            prototxt_file=model+".prototxt"
            if model == "resnet50_int8" :
                prototxt_file="resnet50_int8_full_conv.prototxt"

            
            if model == "faster-rcnn_int8":
                 dst_model_file = self.model_path + "/faster-rcnn/pascal_voc/VGG16/faster_rcnn_end2end/" + self.cpu_model + "_" + "test_int8.prototxt"
            elif model == "rfcn_int8":
                 dst_model_file = self.model_path + "/rfcn/pascal_voc/ResNet-101/rfcn_end2end/" + self.cpu_model + "_" + "test_agnostic_int8.prototxt"
            else:
                 dst_model_file = os.path.join(self.model_path, './int8/', '-'.join([self.cpu_model, prototxt_file]))    

            if os.path.isfile(dst_model_file):
                os.remove(dst_model_file)

            if model == "faster-rcnn_int8":
                 src_model_file = self.model_path + "/faster-rcnn/pascal_voc/VGG16/faster_rcnn_end2end/test_int8.prototxt"
            elif model == "rfcn_int8":
                 src_model_file = self.model_path + "/rfcn/pascal_voc/ResNet-101/rfcn_end2end/test_agnostic_int8.prototxt"
            else: 
                 src_model_file = os.path.join(self.model_path, './int8/', prototxt_file)

            print("source model "+src_model_file)
            if not os.path.isfile(src_model_file):
                logging.exception("template model file {} doesn't exist.".format(src_model_file))
            print("dest model "+model)
            if model.find("ssd") == -1 or model.find("yolo") == -1:
                batch_size_cnt = 1
            else:
                batch_size_cnt = 2
            if model not in self.inf_bkm_batch_size or self.cpu_model not in self.inf_bkm_batch_size[model]:
                logging.exception("Can't find batch size of topology {} and cpu model {} within batch size table".format(model, self.cpu_model))
            batch_size_pattern = re.compile(".*dim.*")
            new_batch_size = self.inf_bkm_batch_size[model][self.cpu_model]
            if self.test_mode == "inf_latency":
                new_batch_size="1" 
            with open(src_model_file, 'r') as src_f, open(dst_model_file, 'w') as dst_f:
                cnt = 0
                line2=""
                for line in src_f.readlines():
                    if line2 != "" and ( line2.find("shape") != -1 or line.find("input: \"data\"") != -1 ) and re.match(batch_size_pattern, line) and cnt < batch_size_cnt:
                        #change batch size
                        line = re.sub("[0-9]+", new_batch_size, line, count = 1)
                        cnt += 1
                    dst_f.write(line) 
                    line2=line
            return dst_model_file
        else:
            if self.test_mode == "inf_throughput" or self.test_mode == "inf_latency":
                prototxt_file = "deploy.prototxt"
            else:
                prototxt_file = "train_val_dummydata.prototxt" if self.dummy_data_use else "train_val.prototxt"
            dst_model_file = os.path.join(self.model_path, './benchmark/', model, '-'.join([self.cpu_model, prototxt_file]))
            if os.path.isfile(dst_model_file):
                os.remove(dst_model_file)
            src_model_file = os.path.join(self.model_path, './benchmark/', model, './', prototxt_file)
            if not os.path.isfile(src_model_file):
                logging.exception("template model file {} doesn't exist.".format(src_model_file))

            if model == "ssd" and ( self.test_mode == "inf_throughput" or self.test_mode == "inf_latency" ):
                batch_size_pattern = re.compile(".*input_shape {.*") if self.dummy_data_use else re.compile("^\s+batch_size:.*")
            else: 
                if model == "mobilenet_v2" and ( self.test_mode == "inf_throughput" or self.test_mode == "inf_latency" ):
                    batch_size_pattern = re.compile(".*input_dim.*") if self.dummy_data_use else re.compile("^\s+batch_size:.*")
                else:
                    batch_size_pattern = re.compile(".*shape:.*") if self.dummy_data_use else re.compile("^\s+batch_size:.*")

            batch_size_cnt = 2
            if self.test_mode == "train_throughput" or self.test_mode == "scal_test":
                if model not in self.train_bkm_batch_size or self.test_mode == "train_throughput" and self.cpu_model not in self.train_bkm_batch_size[model]:
                    logging.exception("Can't find batch size of topology {} and cpu model {} within batch size table".format(model, self.cpu_model))
                new_batch_size = self.train_bkm_batch_size[model] if self.test_mode == "scal_test" else self.train_bkm_batch_size[model][self.cpu_model]
            else:
                if model not in self.inf_bkm_batch_size or self.cpu_model not in self.inf_bkm_batch_size[model]:
                    logging.exception("Can't find batch size of topology {} and cpu model {} within batch size table".format(model, self.cpu_model))
                new_batch_size = self.inf_bkm_batch_size[model][self.cpu_model]
                if self.test_mode == "inf_latency":
                    new_batch_size="1"
     
            with open(src_model_file, 'r') as src_f, open(dst_model_file, 'w') as dst_f:
                cnt = 0
                for line in src_f.readlines():
                    if re.match(batch_size_pattern, line) and cnt < batch_size_cnt:
                        #change batch size
                        line = re.sub("[0-9]+", new_batch_size, line, count = 1)
                        cnt += 1
                    dst_f.write(line) 
            return dst_model_file

    def gen_solver_file(self, model):
        '''generate suitable solver file for training benchmark'''
        solver_prototxt_file = "solver_dummydata.prototxt" if self.dummy_data_use else "solver.prototxt"
        dst_solver_file = os.path.join(self.model_path, model, '-'.join([self.cpu_model, solver_prototxt_file]))
        if os.path.isfile(dst_solver_file):
            os.remove(dst_solver_file)
        src_solver_file = os.path.join(self.model_path, model, solver_prototxt_file)
        if not os.path.isfile(src_solver_file):
            logging.exception("template solver file {} doesn't exist.".format(src_solver_file))
        dst_model_file = self.gen_model_file(model)
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
            model_file = self.gen_model_file(model)
            if self.test_mode == "inf_throughput" or self.test_mode == "inf_latency":
               exec_command = ' '.join([self.caffe_run_script, '--model_file', model_file, '--mode inf_time', '--iteration', str(self.iterations), '--benchmark none', '--ppn', str(self.inf_instances), '--num_omp_threads', str(self.num_omp_threads)]) 
            else:
               exec_command = ' '.join([self.caffe_run_script, '--model_file', model_file, '--mode time', '--iteration', str(self.iterations), '--benchmark none'])
        else:
            solver_file = self.gen_solver_file(model)
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
        print(exec_command)
        logging.info(exec_command)
        self._exec_command_and_show(exec_command)
        self.intelcaffe_log = self.obtain_intelcaffe_log()
        print("calculate fps ...")
        self.calculate_fps(model)
    
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
        intelcaffe_log = os.path.join(result_dir, caffe_log_file)
        logging.info('intelcaffe log: %s' % intelcaffe_log)
        return intelcaffe_log
    
    def obtain_average_time(self):
        '''obtain average iteration time of training'''
        result_file = self.intelcaffe_log
        if not os.path.isfile(result_file):
            logging.exception("Error: result file {} does not exist...".format(result_file))
        if self.num_nodes == 1:
            average_time = ""
            total_average_time = 0 
            num = 0 
            #Need update for ppn >1
            with open(result_file, 'r') as f:
                if self.test_mode == "inf_throughput" or self.test_mode == "inf_latency":
                    pattern = re.compile(".*Average Forward pass:.*")
                else:
                    pattern = re.compile(".*Average Forward-Backward:.*")
                inst = 0
                for line in f.readlines():
                    if re.match(pattern, line):
                        average_time = line.split()[-2]
                        if average_time != "":    
                            total_average_time = total_average_time + float(average_time) 
                            inst = inst + 1
            if total_average_time == 0: 
                logging.exception("Error: can't find average forward-backward time or average forward time within logs, please check logs under: {}".format(result_file))
            logging.info("The total average_time of " + str(inst) + " instances is " + str(total_average_time))
            average_time = total_average_time/inst
            logging.info("The average_time is " + str(average_time))
            return average_time

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
            batch_size_pattern_dummy = re.compile(".*dim:.*")
            batch_size_pattern_real = re.compile(".*\s+batch_size:.*")
            batch_size = ''
            for line in f.readlines():
                if re.match(batch_size_pattern_real, line) or re.match(batch_size_pattern_dummy, line):
                    batch_size = line.split()[-1]
                    break
        if batch_size == '':
            logging.exception("Can't find batch size within your log file: {}".format(log_file))
        batch_size = int(batch_size) * self.num_nodes
        logging.info("global batch size: {}".format(str(batch_size)))
        return float(batch_size)
    
    def calculate_fps(self, model):
        '''calculate fps here'''
        self.batch_size = self.obtain_batch_size()
        self.average_time = self.obtain_average_time()
        speed = self.batch_size * 1000.0 / self.average_time
        self.speed = float(speed)
        total_speed = self.speed * int(self.inf_instances)
        logging.info(model + " benchmark average speed: {} images/sec".format(str(self.speed)))
        logging.info(model + " benchmark total speed: {} images/sec".format(str(total_speed)))
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
        
    def gen_host_file(self, num_nodes):
        '''generate suitable host file to do scaling test'''
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
            self.host_file = self.gen_host_file(num_nodes)
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
        if self.topology == 'all_train':
            for model in self.support_topologies:
                if model == 'all_train':
                    continue
                logging.info("--{}".format(model))
                if self.test_mode == "scal_test":
                    self.run_scal_test(model)
                else:
                    self.run_specific_model(model)
        elif self.topology == 'all_inf':
            for model in self.support_inf_topologies:
                print("run " + model)
                if model == 'all_inf':
                    continue
                logging.info("")
                logging.info("--{}".format(model))
                print("--{}".format(model))
                self.run_specific_model(model)
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

class BenchmarkParams(object):
    '''encapsulate benchmark parameters here'''
    def __init__(self, config_file):
        '''initialize benchmark parameters through a config file'''
        if config_file == '' or not os.path.isfile(config_file):
            logging.exception("Cant't find config file {}.".format(config_file))
        with open(config_file, 'r') as f:
            try:
                config = json.load(f)
            except Exception:
                logging.exception("Error: check if your json config file is correct.")
            self.topology = config['params']['topology']
            self.host_file = config['params']['hostfile']
            self.network = config['params']['network']
            self.tcp_netmask = config['params']['netmask']
            self.engine = config['params']["engine"]
            self.dummy_data_use = config['params']['dummy_data_use']
            self.test_mode = config['params']['test_mode']
            self.inf_instances = config['params']['inf_instances']
            self.num_omp_threads = config['params']['num_omp_threads']
            self.caffe_bin = config['params']['caffe_bin']
            self.train_bkm_batch_size = config['scal_batch_size_table'] if self.test_mode == "scal_test" else config['train_perf_batch_size_table']
            self.inf_bkm_batch_size = config['scal_batch_size_table'] if self.test_mode == "scal_test" else config['inference_perf_batch_size_table']
        
def parse_args():
    '''parse arguments'''
    description = 'Used to run intelcaffe throughput performance or scaling efficiency benchmarking.'
    arg_parser = argparse.ArgumentParser(description = description)
    arg_parser.add_argument('-c', '--configfile', default = 'scripts/benchmark_config_default.json', help = "config file which contains the parameters you want and the batch size you want to use on all topologies and platforms. Please check https://github.com/intel/caffe/wiki/Run-benchmark to see how to use it, default is 'scripts/benchmark_config_default.json'")
    return arg_parser.parse_args()

def main():
    '''main routine'''
    main_args = parse_args()
    bench_params = BenchmarkParams(main_args.configfile) 
    caffe_benchmark = CaffeBenchmark(bench_params)
    caffe_benchmark.run_benchmark()

if __name__ == "__main__":
    main()
