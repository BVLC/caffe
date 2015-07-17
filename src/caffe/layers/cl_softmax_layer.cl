__kernel void kernel_channel_max( int num, int channels, int spatial_dim,
                                 __global Dtype* data, __global Dtype* out) 
    { 
        OCL_KERNEL_LOOP(index, num * spatial_dim) 
        {
             int n = index / spatial_dim;
             int s = index % spatial_dim;
             Dtype maxval = -FLT_MAX;
             for (int c =0; c < channels; ++c)
             {
                maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);                 
             }            
             out[index] = maxval;
        }
    }
    
__kernel void kernel_channel_subtract( int num, int channels, int spatial_dim,
                                      __global Dtype* data, __global Dtype* channel_max)
    {
        OCL_KERNEL_LOOP(index, num * spatial_dim) 
        {
            int n = index / spatial_dim;
            int s = index % spatial_dim;
            for (int c = 0; c < channels; ++c)
            {
                data[(n * channels + c) * spatial_dim + s] -= channel_max[index];
            }
        }        
    }
    
__kernel void kernel_exp( int count, __global Dtype* data, __global Dtype* out)
    {
        OCL_KERNEL_LOOP(index, count) 
        {
            out[index] = exp(data[index]);
        }        
    }   
    
__kernel void kernel_channel_sum( int num, int channels, int spatial_dim,
                                  __global Dtype* data, __global Dtype* channel_sum)
    {
        OCL_KERNEL_LOOP(index, num * spatial_dim) 
        {
            int n = index / spatial_dim;
            int s = index % spatial_dim;
            Dtype sum = 0;
            for (int c = 0; c < channels; ++c)
            {
                sum += data[(n * channels + c) * spatial_dim + s];
            }
            channel_sum[index] = sum;
        }        
    }
    
__kernel void kernel_channel_div( int num, int channels, int spatial_dim,
                                  __global Dtype* data, __global Dtype* channel_sum)
    {
        OCL_KERNEL_LOOP(index, num * spatial_dim) 
        {
            int n = index / spatial_dim;
            int s = index % spatial_dim;
            for (int c = 0; c < channels; ++c)
            {
                data[(n * channels + c) * spatial_dim + s] /= channel_sum[index];
            }
        }        
    }    
        
__kernel void kernel_channel_dot( int num, int channels, int spatial_dim,
                                  __global Dtype* data_1, __global Dtype* data_2,
                                  __global Dtype* channel_dot)
    {
        OCL_KERNEL_LOOP(index, num * spatial_dim) 
        {
            int n = index / spatial_dim;
            int s = index % spatial_dim;
            Dtype dot = 0;
            for (int c = 0; c < channels; ++c)
            {
                dot += (data_1[(n * channels + c) * spatial_dim + s] 
                        * data_2[(n * channels + c) * spatial_dim + s]);
            }
            channel_dot[index] = dot;
        }        
    }          
              