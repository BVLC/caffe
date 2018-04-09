#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/layers/data_augmentation_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>

using std::max;

namespace caffe {
  
template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}  

template <typename Dtype>
Dtype caffe_rng_generate(const RandomGeneratorParameter& param, Dtype discount_coeff = 1) {
  float spread;
  if (param.apply_schedule())
    spread = param.spread() * discount_coeff;
  else
    spread = param.spread();
  const std::string rand_type =  param.rand_type();
  //std::cout << rand_type << " " << rand_type.compare("uniform") << " " << rand_type.compare("gaussian") << " " << rand_type.compare("bernoulli");
  Dtype rand;
  if (rand_type.compare("uniform") == 0) {
    float tmp;
    if (spread > 0.)
      caffe_rng_uniform(1, param.mean() - spread, param.mean() + spread, &tmp);
    else
      tmp = param.mean();
    if (param.exp())
      tmp = exp(tmp);
    rand = static_cast<Dtype>(tmp);
  }
  else if (rand_type.compare("gaussian") == 0) {
    float tmp;
    if (spread > 0.)
      caffe_rng_gaussian(1, param.mean(), spread, &tmp);
    else
      tmp = param.mean();
    if (param.exp())
      tmp = exp(tmp);
    rand = static_cast<Dtype>(tmp);
  }
  else if (rand_type.compare("bernoulli") == 0) {
    int tmp;
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp);
    else
      tmp = 0;
    rand = static_cast<Dtype>(tmp);
  }
  else if (rand_type.compare("uniform_bernoulli") == 0) {
    float tmp1;
    int tmp2;
    
    if (spread > 0.) 
      caffe_rng_uniform(1, param.mean() - spread, param.mean() + spread, &tmp1);
    else
      tmp1 = param.mean();
    
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp2);
    else
      tmp2 = 0;
    
    tmp1 = tmp1 * static_cast<float>(tmp2);
    
    if (param.exp())
      tmp1 = exp(tmp1);
    
    rand = static_cast<Dtype>(tmp1);
  }
  else if (rand_type.compare("gaussian_bernoulli") == 0) {
    float tmp1;
    int tmp2;
    
    if (spread > 0.) 
      caffe_rng_gaussian(1, param.mean(), spread, &tmp1);
    else
      tmp1 = param.mean();
    
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp2);
    else
      tmp2 = 0;
    
    tmp1 = tmp1 * static_cast<float>(tmp2);
    
    if (param.exp())
      tmp1 = exp(tmp1);
    
    rand = static_cast<Dtype>(tmp1);
  }
  else {
    LOG(ERROR) << "Unknown random type " << rand_type;
    rand = NAN;
  }
  return rand;
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom.size(), 1) << "Data aumentation layer takes one or two input blobs.";
  CHECK_LE(bottom.size(), 2) << "Data aumentation layer takes one or two input blobs.";
  CHECK_GE(top.size(), 1) << "Data Layer takes one or two output blobs.";
  CHECK_LE(top.size(), 2) << "Data Layer takes one or two output blobs.";

  if (top.size() == 1) {
    output_params_ = false;
  } else {
    output_params_ = true;
  }
  
  if (bottom.size() == 1) {
    input_params_ = false;
  } else {
    input_params_ = true;
  }
  
  if (output_params_) {
    AugmentationCoeff coeff;
    num_params_ = coeff.GetDescriptor()->field_count();
    LOG(INFO) << "Writing " << num_params_ << " augmentation params";
  }  
  if (input_params_) {
    AugmentationCoeff coeff;
    num_params_ = coeff.GetDescriptor()->field_count();
    LOG(INFO) << "Reading " << num_params_ << " augmentation params";
  }
  
  num_iter_ = 0;
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  
  aug_ = this->layer_param_.augmentation_param();
  
  discount_coeff_schedule_ = this->layer_param_.coeff_schedule_param();
  
  if (!aug_.has_crop_size())
    LOG(ERROR) << "Please enter crop_size if you want to perform augmentation";
  int crop_size = aug_.crop_size();
  CHECK_GE(height, crop_size) << "crop size greater than original";
  CHECK_GE(width, crop_size) << "crop size greater than original";
  
  cropped_height_ = crop_size;
  cropped_width_ = crop_size;

  (top)[0]->Reshape(num, channels, crop_size, crop_size);
  
  if (aug_.recompute_mean()) {
    data_mean_.Reshape(1, channels, crop_size, crop_size);
  }
  
  if (output_params_) {
    (top)[1]->Reshape(num, num_params_, 1, 1);
  }  
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
   num_iter_++;
 
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (top)[0]->mutable_cpu_data();  
  
  std::string write_augmented;
  if (aug_.has_write_augmented())
    write_augmented = aug_.write_augmented();
  else
    write_augmented = std::string("");
  
  bool augment_during_test = aug_.augment_during_test();  
  bool train_phase = (this->phase_ == TRAIN);
  //LOG(INFO) <<  " === train_phase " << train_phase;
  
  Dtype mean_eig [3];
  Dtype mean_rgb[3] = {0., 0., 0.};
  Dtype max_abs_eig[3] = {0., 0., 0.};
  Dtype max_rgb[3] = {0., 0., 0.};
  Dtype min_rgb[3] = {0., 0., 0.};
  Dtype max_l;
  Dtype rgb [3];
  Dtype eig [3];
//   const Dtype eigvec [9] = {0.5579, 0.5859, 0.5878, 0.8021, -0.1989, -0.5631, -0.2130, 0.7856, -0.5809};
//   const Dtype eigvec [9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
//   const Dtype eigvec [9] = {0.57, 0.58, 0.57, -0.72, 0.03, 0.68, -0.38, 0.81, -0.44};
//  const Dtype eigvec [9] = {0.5878, 0.5859, 0.5579, -0.5631, -0.1989, 0.8021, -0.5809, 0.7856, -0.2130};
  const Dtype eigvec [9] = {0.51, 0.56, 0.65, 0.79, 0.01, -0.62, 0.35, -0.83, 0.44};
  if (channels == 3) {  
    for (int item_id = 0; item_id < num; ++item_id) {
      for (int x=0; x < width; x++) {
        for (int y=0; y < height; y++) {
          for (int c=0; c<channels; c++) {          
            rgb[c] = bottom_data[((item_id*channels + c)*width + x)*height + y];      
          }
          //if (x==0 && y ==0)
          //    LOG(INFO) << "item_id=" << item_id << ", x=" << x << ", y=" << y << ", rgb[0]=" << rgb[0] << ", rgb[1]=" << rgb[1] << ", rgb[2]=" << rgb[2];
          for (int c=0; c<channels; c++) {
            eig[c] = eigvec[3*c] * rgb[0] + eigvec[3*c+1] * rgb[1] + eigvec[3*c+2] * rgb[2];
            if (fabs(eig[c]) > max_abs_eig[c])
              max_abs_eig[c] = fabs(eig[c]);
            if (rgb[c] > max_rgb[c])
              max_rgb[c] = rgb[c];
            if (rgb[c] < min_rgb[c])
              min_rgb[c] = rgb[c];
            mean_rgb[c] = mean_rgb[c] + rgb[c]/width/height;
          }
        }
      }
    }
    for (int c=0; c<channels; c++)
      mean_rgb[c] = mean_rgb[c] / num;
    
    for (int c=0; c<channels; c++) {
      mean_eig[c] = eigvec[3*c] * mean_rgb[0] + eigvec[3*c+1] * mean_rgb[1] + eigvec[3*c+2] * mean_rgb[2];
      if ( max_abs_eig[c] > 1e-2 )
        mean_eig[c] = mean_eig[c] / max_abs_eig[c];
    }  
    max_l = sqrt(max_abs_eig[0]*max_abs_eig[0] + max_abs_eig[1]*max_abs_eig[1] + max_abs_eig[2]*max_abs_eig[2]);
  }
   
  //LOG(INFO) << "mean_rgb[0]=" << mean_rgb[0] << ", mean_rgb[1]=" << mean_rgb[1] << ", mean_rgb[2]=" << mean_rgb[2];
  //LOG(INFO) << "max_abs_eig[0]=" << max_abs_eig[0];
  
  AugmentationParameter aug = aug_;
  Dtype* top_params;
  Dtype* bottom_params;
  if (output_params_)
    top_params = (top)[1]->mutable_cpu_data();
  if (input_params_)
    bottom_params = bottom[1]->mutable_cpu_data();
  bool output_params = output_params_;
  bool input_params = input_params_;
  int num_params = num_params_;
  
  Dtype discount_coeff = discount_coeff_schedule_.initial_coeff() + 
      (discount_coeff_schedule_.final_coeff() - discount_coeff_schedule_.initial_coeff()) * (Dtype(2) /
      (Dtype(1) + exp(-discount_coeff_schedule_.gamma() * num_iter_)) - Dtype(1));
      
//   LOG(INFO) << "num_iter=" << num_iter_ << ", discount_coeff=" << discount_coeff;
  
#pragma omp parallel for shared(aug, train_phase, write_augmented, augment_during_test, mean_rgb, mean_eig, max_abs_eig, max_rgb, min_rgb, max_l, eigvec,  output_params, input_params, num_params, discount_coeff)  private(rgb, eig)
  for (int item_id = 0; item_id < num; ++item_id) {
    int x, y, c, top_idx, bottom_idx, h_off, w_off;
    Dtype x1, y1, x2, y2;
    bool do_spatial_transform, do_chromatic_transform;
    
    AugmentationCoeff coeff; 

    //   We only do transformations during training or if specifically asked to do them during testing.
    if (!(train_phase || aug.augment_during_test())) {
      do_spatial_transform   = false;
      do_chromatic_transform = false;
    }
    else {
      if (input_params) {
        array_to_coeff(bottom_params + item_id * num_params, coeff);
      }
      do_spatial_transform   = true;
      do_chromatic_transform = true;
    }  
      
      // sample the parameters of the transformations  
    if (do_spatial_transform) {
      int counter = 0;
      int max_num_tries = 20;    
      int good_params = 0;
      
      // try to sample parameters for which transformed image doesn't go outside the borders of the original one
      // in order to check this, just apply the transformations to 4 corners
      while (good_params < 4 && counter < max_num_tries) {
        good_params = 0;
        generate_spatial_coeffs(aug, coeff, discount_coeff);
  
        //LOG(INFO) << "angle: " << angle << ", zoom: " << zoom_coeff << ", dx: " << dx << ", dy: " << dy << ", mirror: " << mirror;
        
        for (x = 0; x < cropped_width_; x += cropped_width_-1) {
          for (y = 0; y < cropped_height_; y += cropped_height_-1) {
            // move the origin and mirror
            if (coeff.mirror()) {
              x1 =  static_cast<Dtype>(x) - .5 * static_cast<Dtype>(cropped_width_);
              y1 = -static_cast<Dtype>(y) + .5 * static_cast<Dtype>(cropped_height_);            
            } 
            else {
              x1 = static_cast<Dtype>(x) - .5 * static_cast<Dtype>(cropped_width_);
              y1 = static_cast<Dtype>(y) - .5 * static_cast<Dtype>(cropped_height_);
            }
            // rotate
            x2 =  cos(coeff.angle()) * x1 - sin(coeff.angle()) * y1;
            y2 =  sin(coeff.angle()) * x1 + cos(coeff.angle()) * y1;
            // translate
            x2 = x2 + coeff.dx() * static_cast<Dtype>(cropped_width_);
            y2 = y2 + coeff.dy() * static_cast<Dtype>(cropped_height_);
            // zoom
            x2 = x2 / coeff.zoom_x();
            y2 = y2 / coeff.zoom_y();
            // move the origin back
            x2 = x2 + .5 * static_cast<Dtype>(width);
            y2 = y2 + .5 * static_cast<Dtype>(height);
            
            if (!(floor(x2) < 0 || floor(x2) > static_cast<Dtype>(width - 2) || floor(y2) < 0 || floor(y2) > static_cast<Dtype>(height - 2)))
                good_params++;
            //mexPrintf(" (%f,%f) ", x2, y2);
          }
        }
        //mexPrintf("\n");
        counter++;
      }
      if (counter >= max_num_tries) {
        clear_spatial_coeffs(coeff);
        do_spatial_transform = false;
      } 
      else {
        clear_defaults(coeff);
        do_spatial_transform  = coeff.has_mirror() || coeff.has_dx() || coeff.has_dy() || coeff.has_angle() || coeff.has_zoom_x() || coeff.has_zoom_y();
      }      
    } 
    
    if (do_chromatic_transform) {
      generate_chromatic_coeffs(aug, coeff, discount_coeff);
      clear_defaults(coeff);
      do_chromatic_transform =  coeff.has_pow_nomean0()     || coeff.has_pow_nomean1()     || coeff.has_pow_nomean2()    ||
                                coeff.has_add_nomean0()     || coeff.has_add_nomean1()     || coeff.has_add_nomean2()    ||
                                coeff.has_mult_nomean0()    || coeff.has_mult_nomean1()    || coeff.has_mult_nomean2()   ||
                                coeff.has_pow_withmean0()   || coeff.has_pow_withmean1()   || coeff.has_pow_withmean2()  ||
                                coeff.has_add_withmean0()   || coeff.has_add_withmean1()   || coeff.has_add_withmean2()  ||
                                coeff.has_mult_withmean0()  || coeff.has_mult_withmean1()  || coeff.has_mult_withmean2() ||
                                coeff.has_lmult_pow()       || coeff.has_lmult_add()       || coeff.has_lmult_mult()     ||
                                coeff.has_col_angle();    
      
    }
    
    // Augment chromatically only 3-channel inputs
    if (do_chromatic_transform) {
      CHECK_EQ(channels, 3) << "Augment chromatically only 3-channel inputs";
    }
    
    if (write_augmented.size()) {
      if (do_spatial_transform)
        LOG(INFO) << "Augmenting " << item_id 
                  << ", mirror: "  << coeff.mirror()  << ", angle: " << coeff.angle()   
                  << ", zoom_x: "  << coeff.zoom_x()  << ", zoom_y: "  << coeff.zoom_y() 
                  << ", dx: "      << coeff.dx()      << ", dy: " << coeff.dy() ;
      else
        LOG(INFO) << "Not augmenting " << item_id << " spatially";
      if (do_chromatic_transform) 
        LOG(INFO) << "Augmenting "   << item_id 
                  << ", pow_nm0: "   << coeff.pow_nomean0()   << ", mult_nm0: "   << coeff.mult_nomean0()   << ", add_nm0: "   << coeff.add_nomean0()
                  << ", pow_nm1: "   << coeff.pow_nomean1()   << ", mult_nm1: "   << coeff.mult_nomean1()   << ", add_nm1: "   << coeff.add_nomean1()
                  << ", pow_nm2: "   << coeff.pow_nomean2()   << ", mult_nm2: "   << coeff.mult_nomean2()   << ", add_nm2: "   << coeff.add_nomean2()
                  << ", pow_wm0: "   << coeff.pow_withmean0() << ", mult_wm0: "   << coeff.mult_withmean0() << ", add_wm0: "   << coeff.add_withmean0()
                  << ", pow_wm1: "   << coeff.pow_withmean1() << ", mult_wm1: "   << coeff.mult_withmean1() << ", add_wm1: "   << coeff.add_withmean1()
                  << ", pow_wm2: "   << coeff.pow_withmean2() << ", mult_wm2: "   << coeff.mult_withmean2() << ", add_wm2: "   << coeff.add_withmean2()
                  << ". lmult_pow: " << coeff.lmult_pow()     << ", lmult_mult: " <<  coeff.lmult_mult()    << ", lmult_add: " << coeff.lmult_add() 
                  << ", col_angle: " << coeff.col_angle();
      else
        LOG(INFO) << "Not augmenting " << item_id << " chromatically";
    }

    
    if (output_params) {
//       LOG(INFO) << "coeff_to_array item_id=" << item_id << ", num_params=" << num_params; 
      coeff_to_array(coeff, top_params + item_id * num_params);
    }
    
    // actually apply the transformation
    if (do_spatial_transform) { 
      int i00,i01,i10,i11;
      for (x = 0; x < cropped_width_; x++) {
        for (y = 0; y < cropped_height_; y++) {
          // move the origin and mirror
          if (coeff.mirror()) {
            x1 =  static_cast<Dtype>(x) - .5 * static_cast<Dtype>(cropped_width_);
            y1 = -static_cast<Dtype>(y) + .5 * static_cast<Dtype>(cropped_height_);            
          } 
          else {
            x1 = static_cast<Dtype>(x) - .5 * static_cast<Dtype>(cropped_width_);
            y1 = static_cast<Dtype>(y) - .5 * static_cast<Dtype>(cropped_height_);
          }
          // rotate
          if (coeff.has_angle()) {
            x2 =  cos(coeff.angle()) * x1 - sin(coeff.angle()) * y1;
            y2 =  sin(coeff.angle()) * x1 + cos(coeff.angle()) * y1;
          }
          else {
            x2 = x1;
            y2 = y1;
          }
          // translate
          if (coeff.has_dx())
            x2 = x2 + coeff.dx() * static_cast<Dtype>(cropped_width_);
          if (coeff.has_dy())
            y2 = y2 + coeff.dy() * static_cast<Dtype>(cropped_height_);
          // zoom
          if (coeff.has_zoom_x())
            x2 = x2 / coeff.zoom_x();
          if (coeff.has_zoom_y())
            y2 = y2 / coeff.zoom_y();
          // move the origin back
          x2 = x2 + .5 * static_cast<Dtype>(width);
          y2 = y2 + .5 * static_cast<Dtype>(height);
          

          for (c = 0; c < channels; c++) {
            top_idx = ((item_id*channels + c)*cropped_width_ + x)*cropped_height_ + y;
            if (floor(x2) < 0. || floor(x2) > static_cast<Dtype>(width - 2) || floor(y2) < 0. || floor(y2) > static_cast<Dtype>(height - 2))
              top_data[top_idx] = 0.;
            else {
              if (coeff.has_angle() || coeff.has_zoom_x() || coeff.has_zoom_y()) {
                i00 = static_cast<int>(((item_id*channels + c) * width +  floor(x2)) *height + floor(y2));
                i01 = i00 + 1;
                i10 = i00 + height;
                i11 = i00 + height + 1;
                
                top_data[top_idx] = bottom_data[i00] * ((floor(x2)+1)  - x2) * ((floor(y2)+1)  - y2) +
                                    bottom_data[i01] * ((floor(x2)+1)  - x2) * (y2 - floor(y2))      +
                                    bottom_data[i10] * (x2 - floor(x2))      * ((floor(y2)+1)  - y2) +
                                    bottom_data[i11] * (x2 - floor(x2))      * (y2 - floor(y2));                
              } 
              else {
                i00 = static_cast<int>(((item_id*channels + c) * width +  floor(x2)) *height + floor(y2));              
                top_data[top_idx] = bottom_data[i00];
              }
            }         
            // TODO Alexey check this
            //top_data[i] = (top_data[i] - 127.5) * scale;
          }
          //mexPrintf(" (%f,%f) ", x2, y2);        
        }
      }
    }
    else {
      h_off = (height - cropped_height_)/2;
      w_off = (width - cropped_width_)/2;
      for (x = 0; x < cropped_width_; x++) {
        for (y = 0; y < cropped_height_; y++) {
          for (c = 0; c < channels; c++) {
            top_idx = ((item_id*channels + c)*cropped_width_ + x)*cropped_height_ + y;
            bottom_idx = ((item_id*channels + c)*width + x + w_off)*height + y + h_off;
            top_data[top_idx] = bottom_data[bottom_idx];
          }
        }
      }
    }
    
    if (do_chromatic_transform) {
//       LOG(INFO) << " >>> do chromatic transform " << item_id;
      Dtype s, s1, l, l1;      
        
      for (x=0; x<cropped_width_; x++) {
        for (y=0; y<cropped_height_; y++) {
          // subtracting the mean
          for (c=0; c<channels; c++) {
            rgb[c] = top_data[((item_id*channels + c)*cropped_width_ + x)*cropped_height_ + y];
            rgb[c] = rgb[c] - mean_rgb[c];
          }
          // doing the nomean stuff
          for (c=0; c<channels; c++) {
            eig[c] = eigvec[3*c] * rgb[0] + eigvec[3*c+1] * rgb[1] + eigvec[3*c+2] * rgb[2];
            if ( max_abs_eig[c] > 1e-2 ) {
              eig[c] = eig[c] / max_abs_eig[c];
              if (c==0) {
                if (coeff.has_pow_nomean0())            
                  eig[c] = static_cast<float>(sgn(eig[c])) * pow(fabs(eig[c]), coeff.pow_nomean0());
                if (coeff.has_add_nomean0())                 
                  eig[c] = eig[c] + coeff.add_nomean0();
                if (coeff.has_mult_nomean0())
                  eig[c] = eig[c] * coeff.mult_nomean0();
              }
              else if (c==1) {
                if (coeff.has_pow_nomean1())            
                  eig[c] = static_cast<float>(sgn(eig[c])) * pow(fabs(eig[c]), coeff.pow_nomean1());
                if (coeff.has_add_nomean1())                 
                  eig[c] = eig[c] + coeff.add_nomean1();
                if (coeff.has_mult_nomean1())
                  eig[c] = eig[c] * coeff.mult_nomean1();
              } else if (c==2) {
                if (coeff.has_pow_nomean2())            
                  eig[c] = static_cast<float>(sgn(eig[c])) * pow(fabs(eig[c]), coeff.pow_nomean2());
                if (coeff.has_add_nomean2())                 
                  eig[c] = eig[c] + coeff.add_nomean2();
                if (coeff.has_mult_nomean2())
                  eig[c] = eig[c] * coeff.mult_nomean2();
              }
            }
          }
          // re-adding the mean          
          for (c=0; c<channels; c++)
            eig[c] = eig[c] + mean_eig[c];

          // doing the withmean stuff
          if ( max_abs_eig[c] > 1e-2 && (coeff.has_pow_withmean0() || coeff.has_add_withmean0() || coeff.has_mult_withmean0())) {
            if (coeff.has_pow_withmean0())            
              eig[0] = static_cast<float>(sgn(eig[0])) * pow(fabs(eig[0]), coeff.pow_withmean0());
            if (coeff.has_add_withmean0())                 
              eig[0] = eig[0] + coeff.add_withmean0();
            if (coeff.has_mult_withmean0())
              eig[0] = eig[0] * coeff.mult_withmean0();
          }
          if (coeff.has_pow_withmean1() || coeff.has_add_withmean1() || coeff.has_mult_withmean1()) {
            s = sqrt(eig[1]*eig[1] + eig[2]*eig[2]);
            s1 = s;
            if (s > 1e-2) {
              if (coeff.has_pow_withmean1())            
                s1 = pow(s1, coeff.pow_withmean1());
              if (coeff.has_add_withmean1())                 
                s1 = fmax(s1 + coeff.add_withmean1(), 0.);
              if (coeff.has_mult_withmean1())
                s1 = s1 * coeff.mult_withmean1();              
            }
          }
          if (coeff.has_col_angle()) {
            Dtype temp1, temp2;
            temp1 =  cos(coeff.col_angle()) * eig[1] - sin(coeff.col_angle()) * eig[2];
            temp2 =  sin(coeff.col_angle()) * eig[1] + cos(coeff.col_angle()) * eig[2]; 
            eig[1] = temp1;
            eig[2] = temp2;
          }
          for (c=0; c<channels; c++) {
            if ( max_abs_eig[c] > 1e-2 ) {
              eig[c] = eig[c] * max_abs_eig[c]; 
            }
          }
          if (max_l > 1e-2 && (coeff.has_lmult_pow() || coeff.has_lmult_add() || coeff.has_lmult_mult()) || 
                (coeff.has_pow_withmean1() || coeff.has_add_withmean1() || coeff.has_mult_withmean1())) {
            l1 = sqrt(eig[0]*eig[0] + eig[1]*eig[1] + eig[2]*eig[2]);
            l1 = l1 / max_l;
          }
          if (s > 1e-2 && (coeff.has_pow_withmean1() || coeff.has_add_withmean1() || coeff.has_mult_withmean1())) {
            eig[1] = eig[1] / s * s1;
            eig[2] = eig[2] / s * s1;
          }
          if ( max_l > 1e-2 && (coeff.has_lmult_pow() || coeff.has_lmult_add() || coeff.has_lmult_mult()) || 
                (coeff.has_pow_withmean1() || coeff.has_add_withmean1() || coeff.has_mult_withmean1())) {            
            l = sqrt(eig[0]*eig[0] + eig[1]*eig[1] + eig[2]*eig[2]);
            if (coeff.has_lmult_pow())
              l1 = pow(l1, coeff.lmult_pow());
            if (coeff.has_lmult_add())
              l1 = fmax(l1 + coeff.lmult_add(), 0.);
            if (coeff.has_lmult_mult())
              l1 = l1 * coeff.lmult_mult();
            l1 = l1 * max_l;
            if (l > 1e-2)
              for (c=0; c<channels; c++) {
                eig[c] = eig[c] / l * l1;
                if (eig[c] > max_abs_eig[c])
                  eig[c] = max_abs_eig[c];
              }
          }                             
          for (c=0; c<channels; c++) {
            rgb[c] = eigvec[c] * eig[0] + eigvec[3+c] * eig[1] + eigvec[6+c] * eig[2];
//             if (rgb[c] > aug.max_multiplier()*max_rgb[c])
//               rgb[c] = aug.max_multiplier()*max_rgb[c];
//             if (rgb[c] < aug.max_multiplier()*min_rgb[c])
//               rgb[c] = aug.max_multiplier()*min_rgb[c]; 
            if (rgb[c] > aug.max_multiplier())
              rgb[c] = aug.max_multiplier();
            if (rgb[c] < 0.)
              rgb[c] = 0.; 
            top_data[((item_id*channels + c)*cropped_width_ + x)*cropped_height_ + y] = rgb[c];
          }          
        }
      } 
    }    
  }
  
  if(aug.recompute_mean() > 0 && num_iter_ <= aug.recompute_mean() ) {
    Dtype* data_mean_cpu = data_mean_.mutable_cpu_data();
    int count = cropped_width_*cropped_height_*channels;    
    for (int c = 0; c < count; ++c) {
      data_mean_cpu[c] = data_mean_cpu[c]*(static_cast<Dtype>(num_iter_)-1);
      for (int item_id = 0; item_id < num; ++item_id) 
        data_mean_cpu[c] = data_mean_cpu[c] + top_data[item_id*count + c] / num;
      data_mean_cpu[c] = data_mean_cpu[c] / static_cast<Dtype>(num_iter_);      
    }  
  }
    
  if(aug.recompute_mean() > 0) {
    Dtype* data_mean_cpu = data_mean_.mutable_cpu_data();
    int count = cropped_width_*cropped_height_*channels;   
    for (int item_id = 0; item_id < num; ++item_id) {
      for (int c = 0; c < count; ++c) {
        top_data[item_id*count + c] = top_data[item_id*count + c] - data_mean_cpu[c];
      }
    }    
  }
  
  if (write_augmented.size()) {  
    std::ofstream out_file (write_augmented.data(), std::ios::out | std::ios::binary);
    if (out_file.is_open()) { 
      uint32_t imsize[4];
      imsize[0] = num; 
      imsize[1] = channels; 
      imsize[2] = cropped_width_; 
      imsize[3] = cropped_height_;
      LOG(INFO) << "Writing blob size " << imsize[0] << "x" << imsize[1] << "x" << imsize[2] << "x" << imsize[3];
      out_file.write(reinterpret_cast<char*>(&imsize[0]), 4*4);
      out_file.write(reinterpret_cast<const char*>(top_data), imsize[0]*imsize[1]*imsize[2]*imsize[3]*sizeof(float));
      out_file.close();
      LOG(INFO) << " finished augmenting a batch. train=" << train_phase << " === PAUSED === ";
      std::cout << " finished augmenting a batch. train=" << train_phase << " === PAUSED === ";
      std::cin.get();
    }
    else
      LOG(INFO) << "WARNING: Could not open the file" << write_augmented;
  }
  
  if (aug.write_mean().size()) {  
    std::ofstream out_file (aug.write_mean().data(), std::ios::out | std::ios::binary);
    if (out_file.is_open()) { 
      uint32_t imsize[4];
      imsize[0] = 1; 
      imsize[1] = channels; 
      imsize[2] = cropped_width_; 
      imsize[3] = cropped_height_;
      Dtype* data_mean_cpu = data_mean_.mutable_cpu_data();
      LOG(INFO) << "Writing blob size " << imsize[0] << "x" << imsize[1] << "x" << imsize[2] << "x" << imsize[3];
      out_file.write(reinterpret_cast<char*>(&imsize[0]), 4*4);
      out_file.write(reinterpret_cast<const char*>(data_mean_cpu), imsize[0]*imsize[1]*imsize[2]*imsize[3]*sizeof(float));
      out_file.close();
      LOG(INFO) << " finished writing the mean. num_iter_=" << num_iter_ << " === PAUSED === ";
      std::cout << " finished writing the mean. num_iter_=" << num_iter_ << " === PAUSED === ";
      std::cin.get();
    }
    else
      LOG(INFO) << "WARNING: Could not open the file" << write_augmented;
  }
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::generate_spatial_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff) {    
  if (aug.has_mirror())
    coeff.set_mirror(static_cast<float>(caffe_rng_generate<bool>(aug.mirror())));
  if (aug.has_translate()) {
    coeff.set_dx(caffe_rng_generate<float>(aug.translate(), discount_coeff));
    coeff.set_dy(caffe_rng_generate<float>(aug.translate(), discount_coeff));
  } 
  if (aug.has_rotate())
    coeff.set_angle(caffe_rng_generate<float>(aug.rotate(), discount_coeff));
  if (aug.has_zoom()) {
    coeff.set_zoom_x(caffe_rng_generate<float>(aug.zoom(), discount_coeff));
    coeff.set_zoom_y(coeff.zoom_x());
  }
  if (aug.has_squeeze()) {
    float squeeze_coeff = caffe_rng_generate<float>(aug.squeeze(), discount_coeff);
    coeff.set_zoom_x(coeff.zoom_x() * squeeze_coeff);
    coeff.set_zoom_y(coeff.zoom_y() / squeeze_coeff);
  }
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::clear_spatial_coeffs(AugmentationCoeff& coeff) {    
  coeff.clear_mirror();
  coeff.clear_dx();
  coeff.clear_dy();
  coeff.clear_angle();
  coeff.clear_zoom_x();
  coeff.clear_zoom_y();
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::generate_chromatic_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff) {  
  if (aug.has_ladd_pow())
    coeff.set_pow_nomean0(caffe_rng_generate<float>(aug.ladd_pow(), discount_coeff));
  if (aug.has_col_pow()) {
    coeff.set_pow_nomean1(caffe_rng_generate<float>(aug.col_pow(), discount_coeff));
    coeff.set_pow_nomean2(caffe_rng_generate<float>(aug.col_pow(), discount_coeff));
  }
  
  if (aug.has_ladd_add())
    coeff.set_add_nomean0(caffe_rng_generate<float>(aug.ladd_add(), discount_coeff));
  if (aug.has_col_add()) {
    coeff.set_add_nomean1(caffe_rng_generate<float>(aug.col_add(), discount_coeff));
    coeff.set_add_nomean2(caffe_rng_generate<float>(aug.col_add(), discount_coeff));
  }
  
  if (aug.has_ladd_mult())
    coeff.set_mult_nomean0(caffe_rng_generate<float>(aug.ladd_mult(), discount_coeff));
  if (aug.has_col_mult()) {
    coeff.set_mult_nomean1(caffe_rng_generate<float>(aug.col_mult(), discount_coeff));
    coeff.set_mult_nomean2(caffe_rng_generate<float>(aug.col_mult(), discount_coeff));
  }     

  if (aug.has_sat_pow()) {
    coeff.set_pow_withmean1(caffe_rng_generate<float>(aug.sat_pow(), discount_coeff));
    coeff.set_pow_withmean2(coeff.pow_withmean1());
  }
  
  if (aug.has_sat_add()) {
    coeff.set_add_withmean1(caffe_rng_generate<float>(aug.sat_add(), discount_coeff));
    coeff.set_add_withmean2(coeff.add_withmean1());
  }
  
  if (aug.has_sat_mult()) {
    coeff.set_mult_withmean1(caffe_rng_generate<float>(aug.sat_mult(), discount_coeff));
    coeff.set_mult_withmean2(coeff.mult_withmean1());
  }
  
  if (aug.has_lmult_pow())
    coeff.set_lmult_pow(caffe_rng_generate<float>(aug.lmult_pow(), discount_coeff));
  if (aug.has_lmult_mult())
    coeff.set_lmult_mult(caffe_rng_generate<float>(aug.lmult_mult(), discount_coeff));
  if (aug.has_lmult_add())
    coeff.set_lmult_add(caffe_rng_generate<float>(aug.lmult_add(), discount_coeff));
  if (aug.has_col_rotate())
    coeff.set_col_angle(caffe_rng_generate<float>(aug.col_rotate(), discount_coeff));  
}

template <typename Dtype>
void DataAugmentationLayer<Dtype>::clear_defaults(AugmentationCoeff& coeff) {
  const google::protobuf::Reflection* ref = coeff.GetReflection();
  const google::protobuf::Descriptor* desc = coeff.GetDescriptor();
  for (int fn = 0 ; fn < desc->field_count(); fn++) {
    const google::protobuf::FieldDescriptor* field_desc = desc->field(fn);
    float field_val = ref->GetFloat(coeff, field_desc);
    if (fabs(field_desc->default_value_float() - field_val) < 1e-3)
      ref->ClearField(&coeff, field_desc);
//     LOG(INFO) << "field " << fn << " cleared";
  }
} 

template <typename Dtype>
void DataAugmentationLayer<Dtype>::coeff_to_array(const AugmentationCoeff& coeff, Dtype* out) {
  const google::protobuf::Reflection* ref = coeff.GetReflection();
  const google::protobuf::Descriptor* desc = coeff.GetDescriptor();
  for (int fn = 0 ; fn < desc->field_count(); fn++) {
    const google::protobuf::FieldDescriptor* field_desc = desc->field(fn);
    float field_val = ref->GetFloat(coeff, field_desc);
    if (fabs(field_desc->default_value_float()) < 1e-3)
      out[fn] = field_val;
    else
      out[fn] = log(field_val);
//     LOG(INFO) << "writing field " << fn << ": " << out[fn];
  }
//   LOG(INFO) << "Finished writing params";
} 

template <typename Dtype>
void DataAugmentationLayer<Dtype>::array_to_coeff(Dtype* in, AugmentationCoeff& coeff) {
  const google::protobuf::Reflection* ref = coeff.GetReflection();
  const google::protobuf::Descriptor* desc = coeff.GetDescriptor();
  for (int fn = 0 ; fn < desc->field_count(); fn++) {
    const google::protobuf::FieldDescriptor* field_desc = desc->field(fn);
    if (fabs(field_desc->default_value_float()) < 1e-3)
      ref->SetFloat(&coeff, field_desc, in[fn]);
    else
      ref->SetFloat(&coeff, field_desc, exp(in[fn]));
//     LOG(INFO) << "reading field " << fn << ": " << ref->GetFloat(coeff, field_desc);
  }
}


#ifdef CPU_ONLY
STUB_GPU(DataAugmentationLayer);
#endif

INSTANTIATE_CLASS(DataAugmentationLayer);
REGISTER_LAYER_CLASS(DataAugmentation);

}  // namespace caffe

