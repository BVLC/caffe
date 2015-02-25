#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

using boost::shared_ptr;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::NetParameter;
using caffe::NetState;
using caffe::Layer;
using std::vector;

/*
 * A crude re-implementation of the SGD solver on the CPU. This code takes a
 * network proto file as its only argument and runs SGD on that network.
 *
 * See http://caffe.berkeleyvision.org/tutorial/solver.html for a description
 * of the calculations, and see the file solver.cpp for the original implementation
 * of the computations.
 *
 * To compile this code, simply place it in the tools subdirectory of your root caffe folder,
 * then run "make."
 *
 * To run this code on e.g. the MNIST example net, run the following command from your
 * caffe root folder:
 *
 *      build/tools/niel_sgd_code examples/mnist/lenet_train_test.prototxt
 */

int main(int argc, char** argv) {
    Caffe::set_mode(Caffe::CPU);

    //create the net object, given the parameters from the proto file
    NetParameter netParam;
    char* netFile = argv[1];
    ReadNetParamsFromTextFileOrDie(netFile, &netParam);
    NetState myNetState;
    myNetState.set_phase(caffe::TRAIN);
    myNetState.MergeFrom(netParam.state());
    netParam.mutable_state()->CopyFrom(myNetState);
    Net<float> myNet(netParam);

    vector<shared_ptr<Blob<float> > > & netParams = myNet.params();
    vector<float> & netParamsLR = myNet.params_lr();
    vector<float> & netParamsWeightDecay = myNet.params_weight_decay();


    //initialize history: used in SGD to store previous weight update
    vector<shared_ptr<Blob<float> > > history;
    for (int i = 0; i < netParams.size(); i++) {
        const Blob<float> * param = netParams[i].get();
        history.push_back(shared_ptr<Blob<float> >(new Blob<float>(
            param->num(), param->channels(), param->height(),
            param->width())));
    }

    //SGD parameters
    //In theory I think you're supposed to adjust these parameters as learning
    //goes on, but I was going for the most barebones possible implementation
    float rate = 0.01;
    float momentum = 0.9;
    float weightDecay = 0;

    for (int iter = 0; iter < 100; iter++) {
        vector<Blob<float>* > bottomVec;
        float iterLoss;

        //run forward pass on network
        const vector<Blob<float>* > & result = myNet.Forward(bottomVec, &iterLoss);

        //output value of loss function at this iteration
        const float * resultVec = result[0]->cpu_data();
        for (int i = 0; i < result[0]->count(); i++) {
            std::cout << "loss function value at iteration " << iter << " is " << resultVec[i] << std::endl;
        }

        //run backward pass on network
        myNet.Backward();

        //update params
        //mostly lifted from SGDSolver::ComputeUpdateValue()
        for (int paramID = 0; paramID < netParams.size(); paramID++) {
            float localRate = rate * netParamsLR[paramID];
            float localDecay = weightDecay * netParamsWeightDecay[paramID];

            //Add in L2 regularization term
            caffe::caffe_axpy(netParams[paramID]->count(),
                localDecay,
                netParams[paramID]->cpu_data(),
                netParams[paramID]->mutable_cpu_diff());

            //Get current weight update value
            caffe::caffe_cpu_axpby(netParams[paramID]->count(),
                localRate,
                netParams[paramID]->cpu_diff(),
                momentum,
                history[paramID]->mutable_cpu_data());

            //The last step put the weight update values in the history blobs
            //(which we'll need for the next step), so now we need to copy them
            //back into the parameters' diff fields
            caffe::caffe_copy(netParams[paramID]->count(),
                history[paramID]->cpu_data(),
                netParams[paramID]->mutable_cpu_diff());
        }

        //Update the parameters' data fields given the gradients in their diff
        //fields
        myNet.Update();
    }
}
