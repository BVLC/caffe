#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <functional>

#include "caffe/layer.hpp"
#include "caffe/layers/pad_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void PadLayer<Dtype>::LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  const PadParameter& param = this->layer_param_.pad_param();

  PAD_TYPE_ = param.padtype();
  PAD_ = param.pad();
  CHECK_EQ(bottom.size(), 1) << "Pad Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Pad Layer takes a single blob as output.";
  CHECK_EQ(bottom[0]->num_axes(), 4) << "Pad Layer must have four axes.";
  NUM_ = bottom[0]->num();
  CHANNEL_ = bottom[0]->channels();
  HEIGHT_IN_ = bottom[0]->height();
  WIDTH_IN_ = bottom[0]->width();
  HEIGHT_OUT_ = HEIGHT_IN_ + PAD_ * 2;
  WIDTH_OUT_ = WIDTH_IN_ + PAD_ * 2;
}

template <typename Dtype>
void PadLayer<Dtype>::Reshape(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  std::vector<int> shape(4, 0);
  shape[0] = NUM_;
  shape[1] = CHANNEL_;
  shape[2] = HEIGHT_OUT_;
  shape[3] = WIDTH_OUT_;
  top[0]->Reshape(shape);
}


template <typename Dtype>
void PadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		 const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNEL_; ++c) {

      // First copy the main body into place
      for (int h = 0; h < HEIGHT_IN_; ++h) {
        // copy the width part
	caffe_copy(WIDTH_IN_,
		   bottom_data + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h) * WIDTH_IN_,
		   top_data + ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h + PAD_)
		   * WIDTH_OUT_ + PAD_);
      }

      // Now pad, first width, then height. This order may affect the
      // corners
      switch (PAD_TYPE_) {
      case PadParameter::ZERO:
	{
	  // Left and right. Loop over the rows not in the vertical padding
	  for (int h = PAD_; h < HEIGHT_OUT_ - PAD_; ++h) {
	    // Offset to current row start (in padding of this row)
	    int off = ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h) * WIDTH_OUT_;
	    // Left pad
	    for (int wdst = 0; wdst < PAD_; ++wdst) {
	      *(top_data + off + wdst) = static_cast<Dtype>(0);
	    }
	    // Right
	    for (int wdst = WIDTH_OUT_-PAD_; wdst < WIDTH_OUT_; ++wdst) {
	      *(top_data + off + wdst) = static_cast<Dtype>(0);
	    }
	  }
	  // Top
	  for (int h = 0; h < PAD_; ++h) {
	    int off = ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h)
	      * WIDTH_OUT_;
	    std::fill(top_data+off, top_data+off+WIDTH_OUT_, static_cast<Dtype>(0));
	  }
	  // Bottom
	  for (int h = HEIGHT_OUT_-PAD_; h < HEIGHT_OUT_; ++h) {
	    int off = ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h) * WIDTH_OUT_;
	    std::fill(top_data+off, top_data+off+WIDTH_OUT_, static_cast<Dtype>(0));
	  }
	}
	break;
      case PadParameter::REPLICATE:
	{
	  // Left and right. Loop over the rows not in the vertical padding
	  for (int h = PAD_; h < HEIGHT_OUT_ - PAD_; ++h) {
	    // Offset to current row start (in padding of this row)
	    int off = ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h) * WIDTH_OUT_;
	    const Dtype lval = *(top_data + off + PAD_),
	      rval = *(top_data + off + WIDTH_OUT_ - 1 - PAD_);
	    // Left
	    for (int wdst = 0; wdst < PAD_; ++wdst) {
	      *(top_data + off + wdst) = lval;
	    }
	    // Right
	    for (int wdst = WIDTH_OUT_-PAD_; wdst < WIDTH_OUT_; ++wdst) {
	      *(top_data + off + wdst) = rval;
	    }
	  }
	  // Top
	  // Beginning of this image's data, including padding
	  Dtype * dstptr = top_data + ((n * CHANNEL_ + c) * HEIGHT_OUT_) * WIDTH_OUT_;
	  // First row not in the vertical padding
	  Dtype * srcptr = dstptr + PAD_ * WIDTH_OUT_;
	  for (int h = 0; h < PAD_; ++h) {
	    std::copy(srcptr, srcptr + WIDTH_OUT_,
		      dstptr + h * WIDTH_OUT_);
	  }
	  // Bottom
	  // Start of last row not in the vertical padding
	  srcptr = top_data + ((n * CHANNEL_ + c) * HEIGHT_OUT_ +
			       HEIGHT_OUT_ - 1 - PAD_) * WIDTH_OUT_;
	  // Start of first row in bottom padding
	  dstptr = srcptr + WIDTH_OUT_;
	  for (int h = 0; h < PAD_; ++h) {
	    std::copy(srcptr, srcptr + WIDTH_OUT_,
		      dstptr + h*WIDTH_OUT_);
	  }
	}
	break;
      case PadParameter::REFLECT:
	{
	  // Left and right. Loop over the rows not in the vertical padding
	  for (int h = PAD_; h < HEIGHT_OUT_ - PAD_; ++h) {
	    // Offset to current row start (in padding of this row)
	    int off = top[0]->offset(n, c, h, 0);
	    // Left
	    for (int wdst = PAD_-1, wsrc = PAD_; wdst >= 0; --wdst, ++wsrc) {
	      *(top_data + off + wdst) = *(top_data + off + wsrc);
	    }
	    // Right
	    for (int wdst = WIDTH_OUT_-PAD_, wsrc = wdst-1; wdst < WIDTH_OUT_;
		 ++wdst, --wsrc) {
	      *(top_data + off + wdst) = *(top_data + off + wsrc);
	    }
	  }
	
	  // Top
	  for (int hdst = PAD_-1, hsrc = PAD_; hdst >= 0; --hdst, ++hsrc) {
	    caffe_copy(WIDTH_OUT_,
		       top_data + top[0]->offset(n, c, hsrc,0),
		       top_data + top[0]->offset(n, c, hdst,0));
	  }
	  // Bottom
	  for (int hdst = HEIGHT_OUT_-PAD_, hsrc = hdst-1; hdst < HEIGHT_OUT_;
	       ++hdst, --hsrc) {
	    caffe_copy(WIDTH_OUT_,
		       top_data + top[0]->offset(n, c, hsrc,0),
		       top_data + top[0]->offset(n, c, hdst,0));
	  }
	}
	break;
      }
    }
  }
}


template <typename Dtype>
void PadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down,
		  const vector<Blob<Dtype>*>& bottom) {
  Dtype* top_diff = top[0]->mutable_cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  // Very similar to Forward, except reverse the order.
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNEL_; ++c) {
      // First do the padding. We need to reverse the order, and actually need to
      // manipulate the diffs in top, before copying to bottom. First
      // height, then width. This order may affect the corners
      switch (PAD_TYPE_) {
      case PadParameter::ZERO:
	// There's no information in the padding, since it's constant.
	break;
      case PadParameter::REPLICATE:
	{
	  // Top
	  // Beginning of this image's diff, including padding (h, w = 0)
	  Dtype * srcptr = top_diff + ((n * CHANNEL_ + c) * HEIGHT_OUT_) * WIDTH_OUT_,
	    // First row in top not in the vertical padding
	    *dstptr =  srcptr + PAD_ * WIDTH_OUT_;
	  for (int h = 0; h < PAD_; ++h) {
	    std::transform(srcptr + h * WIDTH_OUT_, srcptr + (h+1) * WIDTH_OUT_,
			   dstptr, dstptr, std::plus<Dtype>());
	  }
	  // Bottom
	  // Start of last row not in the vertical padding
	  dstptr = top_diff + ((n * CHANNEL_ + c) * HEIGHT_OUT_ +
			       HEIGHT_OUT_ - 1 - PAD_) * WIDTH_OUT_;
	  // Start of first row in bottom padding
	  srcptr = dstptr + WIDTH_OUT_;
	  for (int h = 0; h < PAD_; ++h) {
	    std::transform(srcptr + h*WIDTH_OUT_, srcptr + (h+1)*WIDTH_OUT_,
			   dstptr, dstptr, std::plus<Dtype>());
	  }
	  // Left and right. Loop over the rows not in the vertical padding
	  for (int h = PAD_; h < HEIGHT_OUT_ - PAD_; ++h) {
	    // Offset to current row start (in padding of this row)
	    int off = ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h) * WIDTH_OUT_;
	    Dtype *lptr = top_diff + off + PAD_,
	      *rptr = top_diff + off + WIDTH_OUT_ - 1 - PAD_;
	    // Left
	    for (int wdst = 0; wdst < PAD_; ++wdst) {
	      *lptr += *(top_diff + off + wdst);
	    }
	    // Right
	    for (int wdst = WIDTH_OUT_-PAD_; wdst < WIDTH_OUT_; ++wdst) {
	      *rptr += *(top_diff + off + wdst);
	    }
	  }
	}
	break;
      case PadParameter::REFLECT:
	{
	  // Bottom. I'm keeping the "dst" and "src" labels from
	  // forward, even though the information is flowing the other
	  // way.
	  for (int hdst = HEIGHT_OUT_-PAD_, hsrc = hdst-1; hdst < HEIGHT_OUT_;
	       ++hdst, --hsrc) {
	    Dtype * const dstptr = top_diff + top[0]->offset(n, c, hdst,0);
	    Dtype * const srcptr = top_diff + top[0]->offset(n, c, hsrc,0);
	    std::transform(dstptr, dstptr + WIDTH_OUT_,
			   srcptr, srcptr, std::plus<Dtype>());
	  }
	
	  // Top
	  for (int hdst = PAD_-1, hsrc = PAD_; hdst >= 0; --hdst, ++hsrc) {
	    Dtype * const dstptr = top_diff + top[0]->offset(n, c, hdst,0);
	    Dtype * const srcptr = top_diff + top[0]->offset(n, c, hsrc,0);
	    std::transform(dstptr, dstptr + WIDTH_OUT_,
			   srcptr, srcptr, std::plus<Dtype>());
	  }

	  // Left and right. Loop over the rows not in the vertical padding
	  for (int h = PAD_; h < HEIGHT_OUT_ - PAD_; ++h) {
	    // Offset to current row start (in padding of this row)
	    int off = top[0]->offset(n, c, h, 0);
	    // Left
	    for (int wdst = PAD_-1, wsrc = PAD_; wdst >= 0; --wdst, ++wsrc) {
	      *(top_diff + off + wsrc) += *(top_diff + off + wdst);
	    }
	    // Right
	    for (int wdst = WIDTH_OUT_-PAD_, wsrc = wdst-1; wdst < WIDTH_OUT_;
		 ++wdst, --wsrc) {
	      *(top_diff + off + wsrc) += *(top_diff + off + wdst);
	    }
	  }
	}
	break;
      } // switch over types

      // Now copy the main body into place
      for (int h = 0; h < HEIGHT_IN_; ++h) {
        // copy the width part
	caffe_copy(WIDTH_IN_,
		   top_diff +
		   ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h + PAD_) * WIDTH_OUT_ +
		   PAD_,
		   bottom_diff + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h) * WIDTH_IN_);
      }
    } // c
  } // n
}

#ifdef CPU_ONLY
STUB_GPU(PadLayer);
#endif

INSTANTIATE_CLASS(PadLayer);
REGISTER_LAYER_CLASS(Pad);

}  // namespace caffe
