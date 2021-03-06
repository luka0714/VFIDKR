#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

int FilterInterpolationLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const  int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int input4_b_stride, const int input4_c_stride, const int input4_h_stride, const int input4_w_stride,

		at::Tensor& input1,    		at::Tensor& input2,    	at::Tensor& input3, 	
		at::Tensor& input4,      
		at::Tensor& output

		);

int FilterInterpolationLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,    		const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int input4_b_stride, const int input4_c_stride, const int input4_h_stride, const int input4_w_stride,

		at::Tensor& input1,        		at::Tensor& input2,		at::Tensor& input3,       
		at::Tensor& input4,

		at::Tensor& gradoutput,    		at::Tensor& gradinput1,  		at::Tensor& gradinput2,  		at::Tensor& gradinput3,        
		at::Tensor& gradinput4
		);


// add deformable conv
int FilterInterpolationLayer_gpu_forward_kernel_deforconv(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const  int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int input4_b_stride, const int input4_c_stride, const int input4_h_stride, const int input4_w_stride,

		at::Tensor& input1,    		at::Tensor& input2,    	at::Tensor& input3, 	
		at::Tensor& input4,      
		at::Tensor& output

		);

int FilterInterpolationLayer_gpu_backward_kernel_deforconv(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,    		const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int input4_b_stride, const int input4_c_stride, const int input4_h_stride, const int input4_w_stride,

		at::Tensor& input1,        		at::Tensor& input2,		at::Tensor& input3,       
		at::Tensor& input4,

		at::Tensor& gradoutput,    		at::Tensor& gradinput1,  		at::Tensor& gradinput2,  		at::Tensor& gradinput3,        
		at::Tensor& gradinput4
		);


// add deformable conv with no kernel filter
int FilterInterpolationLayer_gpu_forward_kernel_nofilterwithdeforconv(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const  int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		at::Tensor& input1,    		at::Tensor& input2,    	at::Tensor& input3, 	  
		at::Tensor& output

		);

int FilterInterpolationLayer_gpu_backward_kernel_nofilterwithdeforconv(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,    		const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		at::Tensor& input1,        		at::Tensor& input2,		at::Tensor& input3,       

		at::Tensor& gradoutput,    		at::Tensor& gradinput1,  		at::Tensor& gradinput2,  		at::Tensor& gradinput3
		);


// ori version
int FilterInterpolationLayer_gpu_forward_kernel_ori(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const  int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		at::Tensor& input1,    		at::Tensor& input2,    	at::Tensor& input3, 	at::Tensor& output

		);

int FilterInterpolationLayer_gpu_backward_kernel_ori(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,    		const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		at::Tensor& input1,        		at::Tensor& input2,		at::Tensor& input3,

		at::Tensor& gradoutput,    		at::Tensor& gradinput1,  		at::Tensor& gradinput2,  		at::Tensor& gradinput3
		);
