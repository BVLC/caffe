// Copyright 2015 George Papandreou

#ifndef CAFFE_UTIL_RANK_ELEMENT_H_
#define CAFFE_UTIL_RANK_ELEMENT_H_

#include <vector>
#include <algorithm>
#include <utility>

namespace caffe {


template <typename T, typename Compare>
class MetaCompare {
 public:
  MetaCompare(const std::vector<T> *val, Compare comp) : val_(val), comp_(comp) {}
  bool operator() (const std::pair<int, int>& p1, const std::pair<int, int>& p2) const {
    return comp_((*val_)[p1.first], (*val_)[p2.first]);
  }
 protected:
  const std::vector<T> *val_;
  Compare comp_;
};


template <typename T, typename Compare>
void rank_element(std::vector<int> &rank, const std::vector<T> &val, Compare comp) {
  std::vector<std::pair<int, int> > valp(val.size());
  for (int i = 0; i < val.size(); ++i) {
    valp[i].first = i;
    valp[i].second = i;
  }
  MetaCompare<T, Compare> meta_comp(&val, comp);
  std::sort(valp.begin(), valp.end(), meta_comp);
  if (rank.size() < val.size()) {
    rank.resize(val.size());
  }
  for (int i = 0; i < val.size(); ++i) {
    rank[i] = valp[i].second;
  }
}

template <typename T, typename Compare>
void partial_rank_element(std::vector<int> &rank, const std::vector<T> &val, int top_k, Compare comp) {
  if (top_k > val.size()) {
    top_k = val.size();
  }
  if (top_k < 1) {
    top_k = 1;
  }
  std::vector<std::pair<int, int> > valp(val.size());
  for (int i = 0; i < val.size(); ++i) {
    valp[i].first = i;
    valp[i].second = i;
  }
  MetaCompare<T, Compare> meta_comp(&val, comp);
  std::partial_sort(valp.begin(), valp.begin() + top_k,
		    valp.end(), meta_comp);
  if (rank.size() < top_k) {
    rank.resize(top_k);
  }
  for (int j = 0; j < top_k; ++j) {
    rank[j] = valp[j].second;
  }
}

}  // namespace caffe

#endif
