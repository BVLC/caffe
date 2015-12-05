#ifndef CAFFE_UTIL_UPGRADE_PROTO_H_
#define CAFFE_UTIL_UPGRADE_PROTO_H_

#include <string>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Return true iff the net is not the current version.
bool NetNeedsUpgrade(const NetParameter& net_param);

// Check for deprecations and upgrade the NetParameter as needed.
bool UpgradeNetAsNeeded(const string& param_file, NetParameter* param);

// Read parameters from a file into a NetParameter proto message.
void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                    NetParameter* param);
void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                      NetParameter* param);

// Return true iff any layer contains parameters specified using
// deprecated V0LayerParameter.
bool NetNeedsV0ToV1Upgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade a V0NetParameter into a
// NetParameter (including upgrading padding layers and LayerParameters).
bool UpgradeV0Net(const NetParameter& v0_net_param, NetParameter* net_param);

// Upgrade NetParameter with padding layers to pad-aware conv layers.
// For any padding layer, remove it and put its pad parameter in any layers
// taking its top blob as input.
// Error if any of these above layers are not-conv layers.
void UpgradeV0PaddingLayers(const NetParameter& param,
                            NetParameter* param_upgraded_pad);

// Upgrade a single V0LayerConnection to the V1LayerParameter format.
bool UpgradeV0LayerParameter(const V1LayerParameter& v0_layer_connection,
                             V1LayerParameter* layer_param);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp

V1LayerParameter_LayerType UpgradeV0LayerType(const string& type);

// Return true iff any layer contains deprecated data transformation parameters.
bool NetNeedsDataUpgrade(const NetParameter& net_param);
<<<<<<< HEAD

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge

V1LayerParameter_LayerType UpgradeV0LayerType(const string& type);

// Return true iff any layer contains deprecated data transformation parameters.
bool NetNeedsDataUpgrade(const NetParameter& net_param);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD

=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD

>>>>>>> pod/device/blob.hpp
=======

>>>>>>> pod-caffe-pod.hpp-merge
=======

>>>>>>> pod/caffe-merge
=======

>>>>>>> pod/device/blob.hpp
=======

>>>>>>> pod-caffe-pod.hpp-merge
// Perform all necessary transformations to upgrade old transformation fields
// into a TransformationParameter.
void UpgradeNetDataTransformation(NetParameter* net_param);

// Return true iff the Net contains any layers specified as V1LayerParameters.
bool NetNeedsV1ToV2Upgrade(const NetParameter& net_param);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge

// Perform all necessary transformations to upgrade a NetParameter with
// deprecated V1LayerParameters.
bool UpgradeV1Net(const NetParameter& v1_net_param, NetParameter* net_param);

bool UpgradeV1LayerParameter(const V1LayerParameter& v1_layer_param,
                             LayerParameter* layer_param);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod/device/blob.hpp

// Perform all necessary transformations to upgrade a NetParameter with
// deprecated V1LayerParameters.
bool UpgradeV1Net(const NetParameter& v1_net_param, NetParameter* net_param);

bool UpgradeV1LayerParameter(const V1LayerParameter& v1_layer_param,
                             LayerParameter* layer_param);

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

>>>>>>> pod-caffe-pod.hpp-merge
=======

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

>>>>>>> pod/caffe-merge
=======

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

>>>>>>> pod/device/blob.hpp
=======

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

>>>>>>> pod-caffe-pod.hpp-merge
// Return true iff the solver contains any old solver_type specified as enums
bool SolverNeedsTypeUpgrade(const SolverParameter& solver_param);

bool UpgradeSolverType(SolverParameter* solver_param);

// Check for deprecations and upgrade the SolverParameter as needed.
bool UpgradeSolverAsNeeded(const string& param_file, SolverParameter* param);

// Read parameters from a file into a SolverParameter proto message.
void ReadSolverParamsFromTextFileOrDie(const string& param_file,
                                       SolverParameter* param);
=======

// Perform all necessary transformations to upgrade old transformation fields
// into a TransformationParameter.
void UpgradeNetDataTransformation(NetParameter* net_param);
=======

V1LayerParameter_LayerType UpgradeV0LayerType(const string& type);
>>>>>>> caffe

// Return true iff any layer contains deprecated data transformation parameters.
bool NetNeedsDataUpgrade(const NetParameter& net_param);

<<<<<<< HEAD
// Check for deprecations and upgrade the NetParameter as needed.
void UpgradeNetAsNeeded(NetParameter* param);

// Read parameters from a file into a NetParameter proto message.
void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                    NetParameter* param);
void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                      NetParameter* param);
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======

V1LayerParameter_LayerType UpgradeV0LayerType(const string& type);

// Return true iff any layer contains deprecated data transformation parameters.
bool NetNeedsDataUpgrade(const NetParameter& net_param);

>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
// Perform all necessary transformations to upgrade old transformation fields
// into a TransformationParameter.
void UpgradeNetDataTransformation(NetParameter* net_param);

// Return true iff the Net contains any layers specified as V1LayerParameters.
bool NetNeedsV1ToV2Upgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade a NetParameter with
// deprecated V1LayerParameters.
bool UpgradeV1Net(const NetParameter& v1_net_param, NetParameter* net_param);

bool UpgradeV1LayerParameter(const V1LayerParameter& v1_layer_param,
                             LayerParameter* layer_param);

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

// Return true iff the solver contains any old solver_type specified as enums
bool SolverNeedsTypeUpgrade(const SolverParameter& solver_param);

bool UpgradeSolverType(SolverParameter* solver_param);

// Check for deprecations and upgrade the SolverParameter as needed.
bool UpgradeSolverAsNeeded(const string& param_file, SolverParameter* param);

// Read parameters from a file into a SolverParameter proto message.
void ReadSolverParamsFromTextFileOrDie(const string& param_file,
                                       SolverParameter* param);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

}  // namespace caffe

#endif   // CAFFE_UTIL_UPGRADE_PROTO_H_
