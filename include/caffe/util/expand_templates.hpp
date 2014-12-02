#ifndef CAFFE_UTIL_EXPAND_TEMPLATES_H_
#define CAFFE_UTIL_EXPAND_TEMPLATES_H_

#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_pretty_print.pb.h"

namespace caffe {

// Return true iff contains at least one TEMPLATE layer.
bool NetContainsTemplates(const NetParameter& net_param);

// Perform all necessary transformations to expand NetParameter containing
// TEMPLATE layers into a NetParameter without them.
// Doing all the string substitutions needed.
void ExpandTemplatesNet(const NetParameter& in_net_param,
                        NetParameter* out_net_param);

}  // namespace caffe

#endif   // CAFFE_UTIL_EXPAND_TEMPLATES_H_
