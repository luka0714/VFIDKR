#include <torch/torch.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h> //works for 1.0.0

#include "filterinterpolation_cuda_kernel.cuh"



int FilterInterpolationLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  input4,
		at::Tensor&  output

		)
		{
	int error = 1 ;

	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;

    int filter_size2 = input3.size( 1);
    int filter_size = (int) sqrt((float) filter_size2);
	// if(inpu4.size(1) != filter_size2) return error;
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

	int input4_b_stride = input4.stride(0);
	int input4_c_stride = input4.stride(1);
	int input4_h_stride = input4.stride(2);
	int input4_w_stride = input4.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(input4_w_stride !=1) return error;

	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = FilterInterpolationLayer_gpu_forward_kernel(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement,w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			input4_b_stride,input4_c_stride,input4_h_stride,input4_w_stride,

			input1,
			input2,
			input3,
			input4,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

		}
int FilterInterpolationLayer_gpu_backward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  input4,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2,
		at::Tensor&  gradinput3,
		at::Tensor&  gradinput4
		)
		{


    int error = 1 ;
	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;


    int filter_size2 = input3.size( 1);
    int filter_size = (int) sqrt((float) filter_size2);
	// if(inpu4.size(1) != filter_size2) return error;
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

	int input4_b_stride = input4.stride(0);
	int input4_c_stride = input4.stride(1);
	int input4_h_stride = input4.stride(2);
	int input4_w_stride = input4.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(input4_w_stride !=1) return error;

    if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = FilterInterpolationLayer_gpu_backward_kernel(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement, //to let the nummous
			w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			input4_b_stride,input4_c_stride,input4_h_stride,input4_w_stride,

			input1,
			input2,
			input3,
			input4,
			gradoutput,
			gradinput1,
			gradinput2,
			gradinput3,
			gradinput4
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;
}


// add deformable conv
int FilterInterpolationLayer_gpu_forward_deforconv(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  input4,
		at::Tensor&  output

		)
		{
	int error = 1 ;

	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;

    int filter_size2 = input3.size( 1);
    int filter_size = (int) sqrt((float) filter_size2);
	// if(inpu4.size(1) != filter_size2) return error;
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

	int input4_b_stride = input4.stride(0);
	int input4_c_stride = input4.stride(1);
	int input4_h_stride = input4.stride(2);
	int input4_w_stride = input4.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(input4_w_stride !=1) return error;

	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = FilterInterpolationLayer_gpu_forward_kernel_deforconv(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement,w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			input4_b_stride,input4_c_stride,input4_h_stride,input4_w_stride,

			input1,
			input2,
			input3,
			input4,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

		}
int FilterInterpolationLayer_gpu_backward_deforconv(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  input4,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2,
		at::Tensor&  gradinput3,
		at::Tensor&  gradinput4
		)
		{


    int error = 1 ;
	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;


    int filter_size2 = input3.size( 1);
    int filter_size = (int) sqrt((float) filter_size2);
	// if(inpu4.size(1) != filter_size2) return error;
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

	int input4_b_stride = input4.stride(0);
	int input4_c_stride = input4.stride(1);
	int input4_h_stride = input4.stride(2);
	int input4_w_stride = input4.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(input4_w_stride !=1) return error;

    if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = FilterInterpolationLayer_gpu_backward_kernel_deforconv(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement, //to let the nummous
			w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			input4_b_stride,input4_c_stride,input4_h_stride,input4_w_stride,

			input1,
			input2,
			input3,
			input4,
			gradoutput,
			gradinput1,
			gradinput2,
			gradinput3,
			gradinput4
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;
}




//	add deformable conv field with no kernel filter

int FilterInterpolationLayer_gpu_forward_nofilterwithdeforconv(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  output

		)
		{
	int error = 1 ;

	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;

    int filter_size2 = input3.size(1) / 2;
    int filter_size = (int) sqrt((float) filter_size2);
	// if(inpu4.size(1) != filter_size2) return error;
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;

	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = FilterInterpolationLayer_gpu_forward_kernel_nofilterwithdeforconv(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement,w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,

			input1,
			input2,
			input3,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

		}
int FilterInterpolationLayer_gpu_backward_nofilterwithdeforconv(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2,
		at::Tensor&  gradinput3
		)
		{


    int error = 1 ;
	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;


    int filter_size2 = input3.size( 1) / 2;
    int filter_size = (int) sqrt((float) filter_size2);
	// if(inpu4.size(1) != filter_size2) return error;
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	// if(input4_w_stride !=1) return error;

    if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = FilterInterpolationLayer_gpu_backward_kernel_nofilterwithdeforconv(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
           at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement, //to let the nummous
			w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,

			input1,
			input2,
			input3,
			gradoutput,
			gradinput1,
			gradinput2,
			gradinput3
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;
}


// ori version
int FilterInterpolationLayer_gpu_forward_ori(
			at::Tensor&  input1,
			at::Tensor&  input2,
			at::Tensor&  input3,
			at::Tensor&  output
		){
	int error = 1 ;

	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;

    int filter_size2 = input3.size( 1);
    int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = FilterInterpolationLayer_gpu_forward_kernel_ori(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
            at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement,w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,


			input1,
			input2,
			input3,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

}

int FilterInterpolationLayer_gpu_backward_ori(
			at::Tensor&  input1,
			at::Tensor&  input2,
			at::Tensor&  input3,
			at::Tensor&  gradoutput,
			at::Tensor&  gradinput1,
			at::Tensor&  gradinput2,
			at::Tensor&  gradinput3
		){
	int error = 1 ;
	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;


    int filter_size2 = input3.size( 1);
    int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = FilterInterpolationLayer_gpu_backward_kernel_ori(
//			at::globalContext().getCurrentCUDAStream(), //works for 0.4.1
            at::cuda::getCurrentCUDAStream(), //works for 1.0.0
			nElement, //to let the nummous
			w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,

			input1,
			input2,
			input3,
			gradoutput,
			gradinput1,
			gradinput2,
			gradinput3
			);
	if (error) {AT_ERROR("CUDA call failed");}

	return error;

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("FilterInterpolationLayer_gpu_forward_ori", &FilterInterpolationLayer_gpu_forward_ori, "FilterInterpolation forward ori version(CUDA)");
  m.def("FilterInterpolationLayer_gpu_backward_ori", &FilterInterpolationLayer_gpu_backward_ori, "FilterInterpolation backward ori version(CUDA)"); 
  m.def("FilterInterpolationLayer_gpu_forward", &FilterInterpolationLayer_gpu_forward, "FilterInterpolation forward (CUDA)");
  m.def("FilterInterpolationLayer_gpu_backward", &FilterInterpolationLayer_gpu_backward, "FilterInterpolation backward (CUDA)");
  m.def("FilterInterpolationLayer_gpu_forward_deforconv", &FilterInterpolationLayer_gpu_forward_deforconv, "FilterInterpolation forward deforconv (CUDA)");	
  m.def("FilterInterpolationLayer_gpu_backward_deforconv", &FilterInterpolationLayer_gpu_backward_deforconv, "FilterInterpolation backward deforconv (CUDA)");
  m.def("FilterInterpolationLayer_gpu_forward_nofilterwithdeforconv", &FilterInterpolationLayer_gpu_forward_nofilterwithdeforconv, "FilterInterpolation forward no filter with deforconv (CUDA)");	
  m.def("FilterInterpolationLayer_gpu_backward_nofilterwithdeforconv", &FilterInterpolationLayer_gpu_backward_nofilterwithdeforconv, "FilterInterpolation backward no filter with deforconv (CUDA)");
}
