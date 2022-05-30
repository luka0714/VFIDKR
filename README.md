# Video Frame Interpolation Based on Deformable Kernel Region

This is the implement of "Video Frame Interpolation Based on Deformable Kernel Region".

### Abstract ###
Video frame interpolation task has recently become more and more prevalent in the computer vision
field. At present, a number of researches based on deep learning have achieved great success. Most
of them are either based on optical flow information, or interpolation kernel, or a combination of
these two methods. However, these methods have ignored that there are grid restrictions on the position of kernel region during synthesizing each target pixel. These limitations result in that they cannot well adapt to the irregularity of object shape and uncertainty of motion, which may lead to irrelevant
reference pixels used for interpolation. In order to solve this problem, we revisit the deformable convolution for video interpolation, which can break
the fixed grid restrictions on the kernel region, making the distribution of reference points more suitable for the shape of the object, and thus warp a
more accurate interpolation frame. Experiments are conducted on four datasets to demonstrate the
superior performance of the proposed model in comparison to the state-of-the-art alternatives.

<img src="/pic/img1.png" width="85%" height="85%">

Paper Source: https://arxiv.org/abs/2204.11396

### Requirment ###

* python == 3.6.8 
* torch == 1.4.0 
* torchvision == 0.5.0 
* cudatoolkit == 9.2 
* cudnn == 7.6.0 
* gcc == 8.3.0 
* nvcc == 10.1 

Please note that the code is very demanding on the version of the deep learning framework, and it may not run if you are not careful.

### Citation ###
<pre><code>@article{tian2022video,
  title={Video Frame Interpolation Based on Deformable Kernel Region},
  author={Tian, Haoyue and Gao, Pan and Peng, Xiaojiang},
  journal={arXiv preprint arXiv:2204.11396},
  year={2022}
}</code></pre>

### Installation ###
Download repository:
<pre><code>$ git clone https://github.com/luka0714/VFIDKR.git</code></pre>

Generate PyTorch extensions:
<pre><code>$ cd VFI-based-DKR
$ cd my_package 
$ ./build.sh</code></pre>

Generate the Correlation package required by <a href='https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/external_packages/correlation-pytorch-master'>PWCNet</a>(we use PWCnet as the network for the flow prediction module according to the method of DAIN[<sup>[1]</sup>](#ref1)):
<pre><code>$ cd ../PWCNet/correlation_package_pytorch1_0
$ ./build.sh</code></pre>

### Train Model ###
Download the <a href='http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip'>Vimeo90K triplet</a> dataset for video frame interpolation training task.

Run the training script:
<pre><code>$ CUDA_VISIBLE_DEVICES=0 python train.py --datasetPath /yourdataset/ --batch_size 3 --lr 0.002 --patience 3 --factor 0.2</code></pre>

When the loss function does not decrease after patience epochs, the learning rate=factor*lr.

### Test Model ###
In order to verify the effectiveness of our model, we also evaluate the trained model on the following four datasets:
* Vimeo90K Test Set: The dataset consists of 3758 video sequences, each with a frame resolution of 448 × 256.
* <a href='https://www.crcv.ucf.edu/data/UCF101/UCF101.rar'>UCF101</a>: The UCF101 dataset is a large-scale human behavior dataset. It consists of video sequences containing camera motion and cluttered background. We selected 333 triples from it for model test, of which the resolution is 320 × 240.
* <a href='https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip'>DAVIS480p</a>: This dataset is composed of 50 highquality, full HD video sequences, covering many common video object segmentation challenges. We select 50 groups of three consecutive frames as the test data, and the resolution of each frame is 854 × 480.
* <a href='http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip'>MPI-Sintel</a>: MPI-Sintel introduces a new optical flow dataset from an open source 3D animation short film Sintel, which has some important features such as long sequence, large motion, specular reflection, motion blur, defocus blur and atmospheric effect.

### Reference ###
<div id = 'ref1'> </div>
[1] Bao W, Lai W S, Ma C, et al. Depth-aware video frame interpolation[C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 3703-3712.

### Contact ###
<a href="tianhy@nuaa.edu.cn">Haoyue Tian</a>
<a href="pan.gao@nuaa.edu.cn">Pan Gao</a>
