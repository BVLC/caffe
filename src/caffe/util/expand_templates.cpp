#include <boost/algorithm/string.hpp>
#include <google/protobuf/text_format.h>

#include <string>

#include "caffe/util/expand_templates.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

string ResolveTemplateNames(const string& path, const string& pwd) {
  CHECK(!boost::starts_with(pwd, "/") && !boost::ends_with(pwd, "/"));
  if (boost::starts_with(path, "/"))
    return path.substr(1, path.size() - 1);
  string cpath = path;
  string cpwd = pwd;
  while (boost::starts_with(cpath, "../")) {
    cpath = cpath.substr(3, cpath.size() - 3);
    size_t i = cpwd.find_last_of('/');
    cpwd = i == string::npos ? "" : cpwd.substr(0, i);
  }
  if (!cpwd.size())
    return cpath;
  if (!cpath.size() || cpath == ".")
    return cpwd;
  return cpwd + '/' + cpath;
}

void ExpandTemplates(const NetParameter& source, NetParameter* target,
    const string& pwd) {
  for (int i = 0; i < source.layers_size(); ++i) {
    if (source.layers(i).type() == LayerParameter_LayerType_TEMPLATE) {
      const LayerParameter& layer = source.layers(i);
      CHECK(layer.has_template_param()) << "Missing template_param";
      const TemplateParameter& template_param = layer.template_param();
      string proto;
      if (ReadFileToString(template_param.source(), &proto)) {
        // Replace variables and references
        for (int j = 0; j < template_param.variable_size(); ++j) {
          const NameValue& p = template_param.variable(j);
          boost::replace_all(proto, "${" + p.name() + "}", p.value());
        }
        NetParameter net;
        CHECK(google::protobuf::TextFormat::ParseFromString(proto, &net))
            << "Failed to parse Template file: " << template_param.source();
        CHECK(layer.has_name() && layer.name().length() > 0)
            << "Template layer must have a name";
        std::size_t found = pwd.find(layer.name());
        if (found != std::string::npos) {
          LOG(FATAL) << "Recursion in " << template_param.source()
            << " due to layer " << layer.name();
        } else {
          ExpandTemplates(net, target,
                          ResolveTemplateNames(layer.name(), pwd));
        }
      } else {
        LOG(ERROR) << "Failed to read Template file: "
          << template_param.source();
      }
    } else {
      LayerParameter *t = target->add_layers();
      t->CopyFrom(source.layers(i));
      t->set_name(ResolveTemplateNames(t->name(), pwd));
      for (int j = 0; j < source.layers(i).top_size(); ++j)
        t->set_top(j,
          ResolveTemplateNames(source.layers(i).top(j), pwd));
      for (int j = 0; j < source.layers(i).bottom_size(); ++j)
        t->set_bottom(j,
          ResolveTemplateNames(source.layers(i).bottom(j), pwd));
    }
  }
}

// Return true iff contains at least one TEMPLATE layer.
bool NetContainsTemplates(const NetParameter& net_param) {
  for (int i = 0; i < net_param.layers_size(); ++i) {
    if (net_param.layers(i).type() == LayerParameter_LayerType_TEMPLATE) {
      return true;
    }
  }
  return false;
}

void ExpandTemplatesNet(const NetParameter& in_net_param,
                        NetParameter* out_net_param) {
  out_net_param->CopyFrom(in_net_param);
  if (NetContainsTemplates(in_net_param)) {
    out_net_param->clear_layers();
    ExpandTemplates(in_net_param, out_net_param, "");
  } else {
    LOG(WARNING) << "Net doesn't contain any Templates";
  }
}

}  // namespace caffe


