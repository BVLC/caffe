#ifndef CAFFE_ARRAY_EXPRESSION_HPP_
#define CAFFE_ARRAY_EXPRESSION_HPP_

#include <string>
#include <vector>
#include "caffe/array/base.hpp"
#include "caffe/array/math.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class Expression: public ArrayBase<Dtype> {
 protected:
  typedef shared_ptr<typename ArrayBase<Dtype>::Reference> Ref;
  struct Implementation: public ArrayBase<Dtype>::Reference {
    Implementation(ArrayShape s, ArrayMode m);
    virtual void evaluate(Array<Dtype> * r) const = 0;
    virtual Array<Dtype> eval() const;
  };

  template<typename F>
  struct UnaryImplementation: public Implementation {
    Ref a_;
    explicit UnaryImplementation(Ref a):Implementation(a->shape, a->mode), a_(a)
    { }
    virtual void evaluate(Array<Dtype> * r) const {
      caffe::Unary<Dtype, F>::eval(a_->eval(), r);
    }
  };

  template<typename F>
  struct BinaryImplementation: public Implementation {
    Ref a_, b_;
    Dtype ca_, cb_;
    BinaryImplementation(Ref a, Ref b):
    Implementation(a->shape, a->mode), a_(a), b_(b), ca_(0), cb_(0) { }
    BinaryImplementation(Ref a, Dtype b):
    Implementation(a->shape, a->mode), a_(a), ca_(0), cb_(b) { }
    BinaryImplementation(Dtype a, Ref b):
    Implementation(b->shape, b->mode), b_(b), ca_(a), cb_(0) { }
    virtual void evaluate(Array<Dtype> * r) const {
      if (a_ && b_)
        caffe::Binary<Dtype, F>::eval(a_->eval(), b_->eval(), r);
      else if (a_)
        caffe::Binary<Dtype, F>::eval(a_->eval(), cb_, r);
      else if (b_)
        caffe::Binary<Dtype, F>::eval(ca_, b_->eval(), r);
      else
        LOG(FATAL) << "Binary expression between two constants";
    }
  };

  shared_ptr<Implementation> imp_;
  virtual Ref ref() const {
    return imp_;
  }
  explicit Expression(Implementation* imp):
  ArrayBase<Dtype>(imp->shape, imp->mode), imp_(imp) { }

 public:
  Expression(const ArrayShape &s, ArrayMode m);
  Array<Dtype> eval() const;
  virtual void evaluate(Array<Dtype> * r) const;
  template<typename F> static Expression Binary(Ref a, Ref b) {
    return Expression(new BinaryImplementation<F>(a, b));
  }
  template<typename F> static Expression Binary(Dtype a, Ref b) {
    return Expression(new BinaryImplementation<F>(a, b));
  }
  template<typename F> static Expression Binary(Ref a, Dtype b) {
    return Expression(new BinaryImplementation<F>(a, b));
  }
  template<typename F> static Expression Unary(Ref a) {
    return Expression(new UnaryImplementation<F>(a));
  }
};


}  // namespace caffe

#endif  // CAFFE_ARRAY_EXPRESSION_HPP_
