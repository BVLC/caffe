import os
import re
import math
import sys
import numpy as np
import check_winograd as chw

pycaffe = os.path.split(os.path.realpath(__file__))[0] + '/../python'
sys.path.insert(0, pycaffe)
import caffe

calibration_algos = ["DIRECT", "KL", "MAXP"]
scaling_modes = ["SINGLE", "MULTIPLE"]
sampling_layers = ["Convolution", "InnerProduct"]

def get_blob_map(test_net, enable_first_conv = False):
    layers = list(test_net._layer_names)
    conv_layers = []
    for idx in range(0, len(layers)):
        if test_net.layers[idx].type in sampling_layers :
            conv_layers.append(layers[idx])

    if not enable_first_conv:
        conv_layers.pop(0)

    conv_top_blob_layer_map = {}
    for k, v in test_net.top_names.items(): # layer name : top blobs
        if k in conv_layers:
            conv_top_blob_layer_map[v[0]] = k # top blob : layer name

    conv_bottom_blob_layer_map = {}
    for k, v in test_net.bottom_names.items(): # layer name : bottom blobs
        if k in conv_layers:
            conv_bottom_blob_layer_map[v[0]] = k # bottom blob : layer_name

    return (conv_layers, test_net.top_names, test_net.bottom_names, conv_top_blob_layer_map, conv_bottom_blob_layer_map)


def get_winograd_info(model, conv_bottom_blob_layer_map, winograd_algo = False):
    winograd_bottoms = []
    winograd_convolutions = []
    if winograd_algo:
        winograd_convolutions = chw.check(model)
        for c in winograd_convolutions:
            for k,v in conv_bottom_blob_layer_map.items():
                if c == v:
                    winograd_bottoms.append(k) 

    return (winograd_bottoms, winograd_convolutions)


def sample(model, weight, winograd_algo=False, itern=1, enable_first_conv=False):
    caffe.set_mode_cpu()
    test_net = caffe.Net(model, weight, caffe.TEST)
    (conv_layers, top_blobs_map, bottom_blobs_map, conv_top_blob_layer_map, conv_bottom_blob_layer_map) = get_blob_map(test_net, enable_first_conv)

    blobs = {}
    for iter_index in range(0, itern):
        print "Iteration:", (iter_index + 1)
        test_net.forward()
        for k, _ in test_net.blobs.items(): # top blob
            output = test_net.blobs[k].data
            if k not in blobs.keys():
                blobs[k] = [output]
            else:
                new_outputs = blobs[k]
                new_outputs.append(output)
                blobs[k] = new_outputs

    params = {}
    for k, _ in test_net.params.items():
        if k not in conv_layers:
            continue
        param = np.abs(test_net.params[k][0].data) # ignore bias
        params[k] = [param]

    (winograd_bottoms, winograd_convolutions) = get_winograd_info(model, conv_bottom_blob_layer_map, winograd_algo)
    return (blobs, params, top_blobs_map, bottom_blobs_map, conv_top_blob_layer_map, conv_bottom_blob_layer_map, winograd_bottoms, winograd_convolutions)


def expand_quantized_bins(quantized_bins, reference_bins):
    expanded_quantized_bins = [0]*len(reference_bins)
    num_merged_bins = len(reference_bins)/len(quantized_bins)
    j_start = 0
    j_end = num_merged_bins
    for idx in xrange(len(quantized_bins)):
        zero_count = reference_bins[j_start:j_end].count(0)
        num_merged_bins = j_end-j_start
        if zero_count == num_merged_bins:
            avg_bin_ele = 0
        else:
            avg_bin_ele = quantized_bins[idx]/(num_merged_bins - zero_count + 0.0)
        for idx1 in xrange(j_start, j_end):
            expanded_quantized_bins[idx1] = (0 if reference_bins[idx1] == 0 else avg_bin_ele)
        j_start += num_merged_bins
        j_end += num_merged_bins
        if (idx+1) == len(quantized_bins) - 1:
            j_end = len(reference_bins)
    return expanded_quantized_bins

def safe_entropy(reference_distr_P, P_sum, candidate_distr_Q, Q_sum):
    assert len(reference_distr_P) == len(candidate_distr_Q)
    tmp_sum1 = 0
    tmp_sum2 = 0
    for idx in range(len(reference_distr_P)):
        p_idx = reference_distr_P[idx]
        q_idx = candidate_distr_Q[idx]
        if p_idx == 0:
            tmp_sum1 += 0
            tmp_sum2 += 0
        else:
            if q_idx == 0:
	        print "Fatal error!, idx = " + str(idx) + " qindex = 0! p_idx = " + str(p_idx)
            tmp_sum1 += p_idx * (math.log(Q_sum*p_idx))
            tmp_sum2 += p_idx * (math.log(P_sum*q_idx))
    return (tmp_sum1 - tmp_sum2)/P_sum

# Reference: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
def get_optimal_scaling_factor(activation_blob, num_quantized_bins = 255):
    max_val = np.max(activation_blob)
    min_val = np.min(activation_blob)
    print min_val, max_val
    if min_val >= 0:
        hist, hist_edeges = np.histogram(activation_blob, bins=2048, range=(min_val, max_val))
        ending_iter = 2047
        starting_iter = int(ending_iter * 0.7)
    else:
        th = max(abs(max_val), abs(min_val))
        hist, hist_edeges = np.histogram(activation_blob, bins=2048, range=(-th, th))
        starting_iter = 0
        ending_iter = 2047
        if abs(max_val) > abs(min_val):
            while starting_iter < ending_iter:
                if hist[starting_iter] == 0:
                    starting_iter += 1
                    continue
                else:
                    break
            starting_iter += int((ending_iter - starting_iter)*0.6)
        else:
            while ending_iter > 0:
                if hist[ending_iter] == 0:
                    ending_iter -= 1
                    continue
                else:
                    break
            starting_iter = int(0.6 * ending_iter)
    bin_width = hist_edeges[1]-hist_edeges[0]
    P_sum = len(activation_blob)
    min_kl_divergence = 0
    min_kl_index = 0
    kl_inited = False
    for i in range(starting_iter, ending_iter+1):
        reference_distr_P = hist[0:i].tolist()
        outliers_count = sum(hist[i:2048])
        if reference_distr_P[i-1] == 0:
            continue
        reference_distr_P[i-1] += outliers_count
        reference_distr_bins = reference_distr_P[:]
        candidate_distr_Q = hist[0:i].tolist()
        num_merged_bins = i/num_quantized_bins
        candidate_distr_Q_quantized = [0]*num_quantized_bins
        j_start = 0
        j_end = num_merged_bins
        for idx in xrange(num_quantized_bins):
	    candidate_distr_Q_quantized[idx] = sum(candidate_distr_Q[j_start:j_end])
            j_start += num_merged_bins
            j_end += num_merged_bins
            if (idx+1) == num_quantized_bins - 1:
	        j_end = i
	candidate_distr_Q = expand_quantized_bins(candidate_distr_Q_quantized, reference_distr_bins)
        Q_sum = sum(candidate_distr_Q)
        kl_divergence = safe_entropy(reference_distr_P, P_sum, candidate_distr_Q, Q_sum)
        if not kl_inited:
            min_kl_divergence = kl_divergence
            min_kl_index = i
            kl_inited = True
        elif kl_divergence < min_kl_divergence:
	    min_kl_divergence = kl_divergence
            min_kl_index = i
	else:
	    pass
    if min_kl_index == 0:
        while starting_iter > 0:
            if hist[starting_iter] == 0:
                starting_iter -= 1
                continue
            else:
                break
        min_kl_index = starting_iter
    return (min_kl_index+0.5)*bin_width

# calibrator info: data, algo, scaling_mode, max_p
INDEX_DATA = 0
INDEX_ALGO = 1
INDEX_SCALE = 2
INDEX_MAXP = 3
INDEX_TYPE = 4
INDEX_CONV = 5
def calibrate_min(calibrator_info):
    iteration_outputs = calibrator_info[INDEX_DATA]
    n = iteration_outputs[0].shape[0]
    c = iteration_outputs[0].shape[1]
    min_map = {}
    for ci in range(0, c):
        min_map[ci] = 0.0

    for iteration_output in iteration_outputs:
        for ni in range(0, n):
            for ci in range(0, c):
                min_map[ci] = min_map[ci] + np.min(iteration_output[ni][ci])

    for ci in range(0, c):
        min_map[ci] = min_map[ci] / (len(iteration_outputs) * n + 0.0)
    
    inputs_min = []
    for ci in range(0, c):
        inputs_min.append(min_map[ci])        

    return inputs_min
   
def calibrate_max(calibrator_info):
    is_activation = calibrator_info[INDEX_TYPE]
    calibration_algo = "DIRECT"
    if calibrator_info[INDEX_ALGO] in calibration_algos:
        calibration_algo = calibrator_info[INDEX_ALGO]
    scaling_mode = "SINGLE"
    if calibrator_info[INDEX_SCALE] in scaling_modes:
        scaling_mode = calibrator_info[INDEX_SCALE]
   
    is_wino_conv = False
    if calibrator_info[INDEX_CONV]: 
        is_wino_conv = calibrator_info[INDEX_CONV]

    iteration_outputs = calibrator_info[INDEX_DATA]
    maxp = calibrator_info[INDEX_MAXP]
    if scaling_mode == "SINGLE":
        layer_max = sys.float_info.min
        if calibration_algo == "DIRECT":
            for iteration_output in iteration_outputs:
                if is_wino_conv:
                    if is_activation:
                        iteration_output_wino = wino_transform_data(iteration_output)
                    else: 
                        iteration_output_wino = wino_transform_param(iteration_output)
                else:
                    iteration_output_wino = iteration_output
                iteration_max = np.max(iteration_output_wino)
                if iteration_max > layer_max:
                    layer_max = iteration_max
        elif calibration_algo == "MAXP":
            iteration_sum = 0
            for iteration_output in iteration_outputs:
                if is_wino_conv:
                    if is_activation:
                        iteration_output_wino = wino_transform_data(iteration_output)
                    else:
                        iteration_output_wino = wino_transform_param(iteration_output)  
                else:
                    iteration_output_wino = iteration_output
                iteration_output_sorted = np.sort(iteration_output_wino.flatten())
                iteration_max = iteration_output_sorted[int(len(iteration_output_sorted) * maxp)]
                iteration_sum += iteration_max
            layer_max = iteration_sum / (len(iteration_outputs) + 0.0)
        else:
            iteration_blobs = iteration_outputs[0].ravel()
            for i in range(1, len(iteration_outputs)):
                iteration_blobs = np.append(iteration_blobs, iteration_outputs[i].ravel())
            layer_max = get_optimal_scaling_factor(iteration_blobs, 255)

        layer_max = float(layer_max)
        return [layer_max]
    else:
        layer_max_oc = []
      
        if calibration_algo == "DIRECT":
            for i in range(0, iteration_outputs[0].shape[0]):
                layer_max_oc.append(sys.float_info.min)
            for iteration_output in iteration_outputs:
                if is_wino_conv:
                    if is_activation:
                        iteration_output_wino = wino_transform_data(iteration_output)
                    else:
                        iteration_output_wino = wino_transform_param(iteration_output)
                else:
                    iteration_output_wino = iteration_output
                for i in range(0, iteration_output_wino.shape[0]):
                    oc_max = np.max(iteration_output_wino[i])
                    if oc_max > layer_max_oc[i]:
                        layer_max_oc[i] = float(oc_max)
        elif calibration_algo == "MAXP":
            for i in range(0, iteration_outputs[0].shape[0]):
                layer_max_oc.append(0)
            for iteration_output in iteration_outputs:
                if is_wino_conv:
                    if is_activation:
                        iteration_output_wino = wino_transform_data(iteration_output)
                    else:
                        iteration_output_wino = wino_transform_param(iteration_output)
                else:
                    iteration_output_wino = iteration_output
                for i in range(0, iteration_output_wino.shape[0]):
                    iteration_output_sorted = np.sort(iteration_output_wino[i].flatten())
                    iteration_max = iteration_output_sorted[int(len(iteration_output_sorted) * maxp)]
                    layer_max_oc[i] += iteration_max
            for i in range(0, iteration_outputs[0].shape[0]):
                layer_max_oc[i] = layer_max_oc[i] / (len(iteration_outputs) + 0.0)
        else:
            print "Unsupported..."

        for i in range(len(layer_max_oc)):
            layer_max_oc[i] = float(layer_max_oc[i])
        return layer_max_oc

def np_abs(blobs):
    blobs_abs = []
    for blob_index in range(0, len(blobs)):
        blobs_abs.append(np.abs(blobs[blob_index]))
    return blobs_abs

def calibrate_activations(blobs, conv_top_blob_layer_map, conv_bottom_blob_layer_map, winograd_bottoms, calibration_algo="DIRECT", scaling_mode="SINGLE", is_wino_conv=False, maxp=0.99995):
    conv_blobs = {}
    for k, _ in conv_top_blob_layer_map.iteritems():
        if k not in conv_blobs.keys():
            conv_blobs[k] = blobs[k]

    for k, _ in conv_bottom_blob_layer_map.iteritems():
        if k not in conv_blobs.keys():
            conv_blobs[k] = blobs[k]

    conv_blobs_max = {}
    conv_blobs_min = {}
    for k, v in conv_blobs.items():
        print "Calibrate activation for", k, v[0].shape
        if is_wino_conv:
            if k in winograd_bottoms:
                v_max = calibrate_max((np_abs(v), calibration_algo, scaling_mode, maxp, 1, True))
                conv_blobs_max[k] = v_max
            else:
                v_max = calibrate_max((np_abs(v), calibration_algo, scaling_mode, maxp, 0, False))
                conv_blobs_max[k] = v_max
        else:
            print "-----------------"
            v_max = calibrate_max((np_abs(v), calibration_algo, scaling_mode, maxp, 0, False))
            v_min = calibrate_min((v, calibration_algo, scaling_mode, maxp, 0, False))
            conv_blobs_max[k] = v_max
            conv_blobs_min[k] = v_min

    outputs_max = {}
    for k, v in conv_top_blob_layer_map.iteritems():
        outputs_max[v] = conv_blobs_max[k]

    inputs_max = {}
    inputs_min = {}
    for k, v in conv_bottom_blob_layer_map.iteritems():
        inputs_max[v] = conv_blobs_max[k]
        inputs_min[v] = conv_blobs_min[k]
    
    return (inputs_max, outputs_max, inputs_min)

def calibrate_parameters(params, winograd_convolutions, calibration_algo="DIRECT", scaling_mode="SINGLE", is_wino_conv=False, maxp=0.99995):
    params_max = {}
    for k, v in params.items():
        print "Calibrate parameters for", k, v[0].shape
        if is_wino_conv:
            if k in winograd_convolutions:
                v_max = calibrate_max((v, calibration_algo, scaling_mode, maxp, 0, True))
                params_max[k] = v_max
            else:
                v_max = calibrate_max((v, calibration_algo, scaling_mode, maxp, 0, False))
                params_max[k] = v_max
        else:
            v_max = calibrate_max((v, calibration_algo, scaling_mode, maxp, 0, False))
            params_max[k] = v_max

    return params_max
    
def wino_transform_data(blob): #B^tdB
    blob_wino = blob.copy()
    blob_cell = ([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    B_t = np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]])
    B = B_t.T
    for n in range(blob.shape[0]):
        for c in range(blob.shape[1]):
            for h in range(0,blob.shape[2],2):
                for w in range(0,blob.shape[3],2):
                    if (w+4) <= blob.shape[3] and (h+4) <= blob.shape[2]:
                        for i in range(4):
                            for j in range(4):
                                blob_cell[i][j] = blob[n][c][h+i][w+j]
                        blob_wino_tmp = np.dot(B_t, blob_cell)
                        blob_wino_cell = np.dot(blob_wino_tmp, B)
                        for i in range(4):
                            for j in range(4):
                                blob_wino[n][c][h+i][w+j] = blob_wino_cell[i][j]
    return blob_wino

def wino_transform_param(blob): #GgG^t
    blob_wino = np.zeros([blob.shape[0],blob.shape[1],4,4])
    blob_cell = ([[1,1,1],[1,1,1],[1,1,1]])
    G = np.array([[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]])
    G_t = G.T
    for n in range(blob.shape[0]):
        for c in range(blob.shape[1]):
            for h in range(blob.shape[2]):
                for w in range(blob.shape[3]):
                    blob_cell[h][w] = blob[n][c][h][w]
            blob_wino_tmp = np.dot(G, blob_cell)
            blob_wino_cell = np.dot(blob_wino_tmp, G_t)
            blob_wino[n][c] = blob_wino_cell
    return blob_wino

 
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usgae: python sampling.py $prototxt $weights $whether use winograd(True or False)"
        sys.exit(0)
    (blobs, params, top_blobs_map, bottom_blobs_map, conv_top_blob_layer_map, conv_bottom_blob_layer_map, winograd_bottoms, winograd_convolutions) = sample(sys.argv[1], sys.argv[2], False, 2, True)
    (inputs_max, outputs_max, inputs_min) = calibrate_activations(blobs, conv_top_blob_layer_map, conv_bottom_blob_layer_map,winograd_bottoms, "DIRECT", "SINGLE")
    params_max = calibrate_parameters(params, winograd_convolutions, "DIRECT", "MULTIPLE", True)
#    for k, v in inputs_max.items():
#        print "***", k, v[0], outputs_max[k][0]
#    for k, v in params_max.items():
#        print "###", k, v[0]
