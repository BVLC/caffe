import os
import sys
import argparse
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import copy
import utils


def genOptimalModel(net, mkldnn_direct_time_map, mkldnn_winograd_time_map, optimal_model):
    for index in range(0, len(net.layer)):
        l = net.layer[index]
        if l.type == "Convolution":
            if len(l.convolution_param.kernel_size) == 0:
                continue
            kernel_size = l.convolution_param.kernel_size[0]
            stride = 1
            if len(l.convolution_param.stride) != 0:
                stride = l.convolution_param.stride[0]
            if mkldnn_winograd_time_map[l.name] < mkldnn_direct_time_map[l.name] and kernel_size == 3 and stride == 1 and l.convolution_param.num_output % 16 == 0:
                l.convolution_param.conv_algorithm = "winograd"
            else:
                l.convolution_param.conv_algorithm = "direct"

    with open(optimal_model, "w") as f:
        f.write(txtf.MessageToString(net, float_format=".17g"))


def tuneModelDefinition(model_path, iteration, is_test_phase, core_num, socket_num):
    working_dir = sys.path[0]
    caffe_path = os.path.join(working_dir, "..", "..",
                              "build", "tools", "caffe")
    if not os.path.exists(caffe_path):
        print "Caffe binary does not exist; please build Caffe binary first."
        sys.exit(1)
    core_num_per_socket = int([i for i in os.popen('lscpu').readlines(
    ) if i.startswith('Core(s) per socket:')][0].strip().split(':')[-1].strip())
    core_num_per_socket *= socket_num
    if is_test_phase:
        caffe_path += ' time -phase TEST -forward_only'
        if core_num != 0:
            core_num = core_num_per_socket if core_num > core_num_per_socket else core_num
            env_prefix = 'OMP_NUM_THREADS={} KMP_HW_SUBSET=1t KMP_AFFINITY=compact,granularity=fine numactl -C 0-{} -m 0 '.format(
                core_num, core_num)
        elif socket_num == 1:
            env_prefix = 'OMP_NUM_THREADS={} KMP_HW_SUBSET=1t KMP_AFFINITY=compact,granularity=fine numactl -C 0-{} -m 0 '.format(
                core_num_per_socket, core_num_per_socket)
        else:
            env_prefix = 'numactl -l '
        caffe_path = env_prefix + caffe_path

    base_model_name = os.path.basename(model_path)
    model_dir = os.path.dirname(model_path)
    winograd_model_name = base_model_name.split(".")[0] + "_winograd.prototxt"
    winograd_model_path = os.path.join(model_dir, winograd_model_name)
    direct_model_name = base_model_name.split(".")[0] + "_direct.prototxt"
    direct_model_path = os.path.join(model_dir, direct_model_name)

    base_net = caffe_pb2.NetParameter()
    with open(model_path) as f:
        s = f.read()
        txtf.Merge(s, base_net)

    direct_net = copy.deepcopy(base_net)
    for index in range(0, len(direct_net.layer)):
        l = direct_net.layer[index]
        if l.type == "Convolution":
            l.convolution_param.conv_algorithm = "direct"

    with open(direct_model_path, "w") as f:
        f.write(txtf.MessageToString(direct_net, float_format=".17g"))

    winograd_net = copy.deepcopy(base_net)
    for index in range(0, len(winograd_net.layer)):
        l = winograd_net.layer[index]
        if l.type == "Convolution":
            l.convolution_param.conv_algorithm = "winograd"

    with open(winograd_model_path, "w") as f:
        f.write(txtf.MessageToString(winograd_net, float_format=".17g"))

    mkldnn_direct_log = "mkldnn_direct.log"
    mkldnn_winograd_log = "mkldnn_winograd.log"
    mkldnn_direct_log_path = os.path.join(model_dir, mkldnn_direct_log)
    mkldnn_winograd_log_path = os.path.join(model_dir, mkldnn_winograd_log)
    mkldnn_direct_command = caffe_path + " -model " + direct_model_path + \
        " -engine MKLDNN -iterations " + \
        str(iteration) + " >& " + mkldnn_direct_log_path
    os.system(mkldnn_direct_command)
    mkldnn_winograd_command = caffe_path + " -model " + winograd_model_path + \
        " -engine MKLDNN -iterations " + \
        str(iteration) + " >& " + mkldnn_winograd_log_path
    os.system(mkldnn_winograd_command)

    (model_str, mkldnn_direct_time_lines) = utils.parseLog(mkldnn_direct_log_path)
    mkldnn_direct_layer_time_map = utils.parseTimeLines(
        mkldnn_direct_time_lines)
    (model_str, mkldnn_winograd_time_lines) = utils.parseLog(
        mkldnn_winograd_log_path)
    mkldnn_winograd_layer_time_map = utils.parseTimeLines(
        mkldnn_winograd_time_lines)

    hybrid_model_name = base_model_name.split(".")[0] + "_hybrid.prototxt"
    hybrid_model_path = os.path.join(model_dir, hybrid_model_name)
    genOptimalModel(base_net, mkldnn_direct_layer_time_map,
                    mkldnn_winograd_layer_time_map, hybrid_model_path)
    print '{} has been generated.'.format(hybrid_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', action='store', dest='model', default="",
                        help='require the model definition (prototxt)')

    parser.add_argument('-i', '--iteration', action='store', dest='iterations', type=int, default=10,
                        help='require iterations number to run the model')

    parser.add_argument('-t', '--phase', action='store', dest='is_test_phase', type=bool, default=False,
                        help='Train or Test phase')

    parser.add_argument('-c', '--core_num', action='store', dest='core_num', type=int, default=0,
                        help='core number for inference')

    parser.add_argument('-s', '--socket', action='store', dest='socket_num', type=int, default=2,
                        help='socket number for inference')

    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s 1.0')

    params = parser.parse_args()

    model = params.model
    if not os.path.exists(params.model):
        print "[ERROR] Please specify the model definition file with -m"
        exit(1)

    tuneModelDefinition(params.model, params.iterations,
                        params.is_test_phase, params.core_num,  params.socket_num)
