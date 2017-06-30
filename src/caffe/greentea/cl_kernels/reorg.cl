#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(reorg, Dtype)(const int_tp n,__global const Dtype* x,
				     int_tp w, int_tp h, int_tp c,
				     int_tp batch, int_tp stride, int_tp forward,
				     __global Dtype* out) {
	int_tp size = batch*c*h*w;
	for (int_tp index = get_global_id(0); index < n; index += get_global_size(0))
	{
	        int_tp i;
                i = index;
		if(i >= size) return;
		int_tp in_index = i;
		int_tp in_w = i%w;
		i = i/w;
		int_tp in_h = i%h;
		i = i/h;
		int_tp in_c = i%c;
		i = i/c;
		int_tp b = i%batch;

		int_tp out_c = c/(stride*stride);

		int_tp c2 = in_c % out_c;
		int_tp offset = in_c / out_c;
		int_tp w2 = in_w*stride + (offset % stride);
		int_tp h2 = in_h*stride + offset / stride;
		int_tp out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

		if(forward)
		{
			out[out_index] = x[in_index];
		}
		else
		{
			out[in_index] = x[out_index];
		}
	}
}

