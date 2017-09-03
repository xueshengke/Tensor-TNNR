This code includes the detailed implementation of the paper:

Reference:
S. Xue, et~al., Low-rank Tensor Completion by Truncated Nuclear Norm 
Regularization,?IEEE Conference on Computer Vision and Pattern Recognition, 
submitted, 2018.

The code contains:
|--------------
|-- DW_TNNR_main.m            entrance to start the real image experiment
|-- DW_TNNR_synthetic.m       entrance to start the synthetic experiment
|-- function/                 
    |-- DW_TNNR_algorithm.m   main part of DW-TNNR implementation
    |-- PSNR.m                compute the PSNR and Erec for recovered image
    |-- weight_exp.m          compute weight matrix by exponential function	
    |-- weight_matrix.m       compute weight matrix in an increasing order
    |-- weight_sort.m         sort the sequence of weight value according to
                                  observed elements; rows with more observed 
                                  elements are given smaller weights
|-- image/                    directory for original images
|-- mask/                     directory for various mask types, 300x300
|-- result/                   directory for saving experimental results
|-------------

For the algorithm interpretation, please read our Xue et al. Tensor TNNR 
paper (2018), in which more details are demonstrated.

If you have any questions about this implementation, please do not hesitate 
to contact me.

Ph.D. Candidate, Shengke Xue, 
College of Information Science and Electronic Engineering,
Zhejiang University, Hangzhou, P. R. China,
e-mail: (either one is o.k.)
xueshengke@zju.edu.cn, xueshengke1993@gmail.com.