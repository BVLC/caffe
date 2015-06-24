#include <string>
#include <string>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <glog/logging.h>
#include "caffe/util/io.hpp"
#include "caffe/caffe.hpp"
using namespace std;
using namespace caffe;

#define FEATURE_VEC_SIZE 65
#define PROB_VEC_SIZE 8
#define Number_Samples 800

int load_features(float** in, string feature_file, int vec_size) {
  // Read in features from file
  // First need to detect how many feature vectors
  ifstream inFile(feature_file.c_str(), ifstream::in);
  int feat_cnt = count(istreambuf_iterator<char>(inFile),
                       istreambuf_iterator<char>(), '\n') -
                 1;

  // Allocate memory for input feature array
  *in = (float*)malloc(sizeof(float) * feat_cnt * vec_size);

  // Read the feature in
  int idx = 0;
  ifstream featFile(feature_file.c_str(), ifstream::in);
  string line;
  getline(featFile, line);  // Get rid of the first line
  while (getline(featFile, line)) {
    istringstream iss(line);
    float temp;
    while (iss >> temp) {
      (*in)[idx] = temp;
      idx++;
    }
  }

  // Everything should be in, check for sure
  assert(idx == feat_cnt * vec_size && "Error: Read-in feature not correct.");

  return feat_cnt;
}

void process_label_train(float* labels, float* labels2d){
	for ( int row = 0 ; row < Number_Samples ; row++ ){
		for ( int col = 0 ; col < PROB_VEC_SIZE ; col ++){
			if( (int)labels[row] == (col+1) )
				labels2d[row*PROB_VEC_SIZE+col] = 1;
			else
				labels2d[row*PROB_VEC_SIZE+col] = -1;
		}
	}
}

int check_output(float* elm_output, float* exp_out){
	int label[Number_Samples];
	int count=0;
	for (int i = 0; i < Number_Samples; i++){
		float max=0;
		for (int j = 0; j < PROB_VEC_SIZE; j++){
			if(elm_output[i*PROB_VEC_SIZE+j]>max){
				max = elm_output[i*PROB_VEC_SIZE+j];
				label[i]=j+1;
			}

		}
		if(exp_out[i]==(float)label[i])
			count++;
	}
	return count;

}

void dnn_fwd(float* in, float* out, Net<float>* net, float* predicted){
	cout << "going to get input blobs" << endl;
	vector<caffe::Blob<float>*> in_blobs = net->input_blobs();
	cout << "got input blobs" << endl;
	vector<caffe::Blob<float>*> out_blobs;
	float loss;
	cout << "going to set data" << endl;
	in_blobs[0]->set_cpu_data(in);
	cout << "setted in_blob[0]" << endl;
	in_blobs[1]->set_cpu_data(out);
	cout << "setted in_blob[1]" << endl;
  out_blobs =	net->ForwardPrefilled(&loss);
	memcpy(predicted,out_blobs[0]->cpu_data(), sizeof(float) *PROB_VEC_SIZE*Number_Samples);
  cout << "ForwardPrefilled finished"<< endl;

}

int main(int argc, char** argv){
	if(argc!=6){
		cout<<"Incorrect number of inputs"<<endl;
		return 0;
	}
	string network(argv[1]);
	string phase(argv[2]);
	string features(argv[3]);
	string labels(argv[4]);
	string weights(argv[5]);
	float* input = NULL; //features
	float* exp_out1d = NULL; //1-D array containing labels
	float* exp_out2d = (float*)malloc(sizeof(float)*PROB_VEC_SIZE*Number_Samples); //2-D array of labels for one vs one classification
	float* elm_output = (float*)malloc(sizeof(float)*PROB_VEC_SIZE*Number_Samples); //array to store output of elm
	int feat_cnt = load_features(&input, features, FEATURE_VEC_SIZE);
	int exp_out_cnt = load_features(&exp_out1d, labels, PROB_VEC_SIZE);
	process_label_train(exp_out1d,exp_out2d); //conver 1-D array of label to 2-D

	if(phase == "TRAIN"){
		// Train a new network

		cout<<"Create a network"<<endl;
		Net<float>* elm = new Net<float>(network,TRAIN);

		cout<<"Forward pass"<<endl;
		dnn_fwd(input,exp_out2d,elm,elm_output);

		// Compare output with actual labels and find accuracy
		int count = check_output(elm_output,exp_out1d);
		float accu = (count/Number_Samples)*100;
		cout << "Training accuracy is : "<<accu<<endl;

		// Saving trained Network to file
		NetParameter elm_param;
		elm->ToProto(&elm_param,false);
		WriteProtoToBinaryFile(elm_param,weights);

	}
	else if(phase == "TEST"){
		//Test a network

		// Create a network
		Net<float>* elm = new Net<float>(network,TEST);

		// Read trained network from file
		elm->CopyTrainedLayersFrom(weights);

		// Forward pass
		dnn_fwd(input,exp_out2d,elm,elm_output);

		// Compare output with actual labels and find accuracy
		int count = check_output(elm_output,exp_out1d);
		float accu = (count/Number_Samples)*100;
		cout << "Testing accuracy is : "<<accu<<endl;
	}

}
