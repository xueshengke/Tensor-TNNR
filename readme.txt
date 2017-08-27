This code includes the detailed implementation of the paper:

Reference:
S. Xue, W. Qiu, F. Liu, et~al., “Double Weighted Truncated Nuclear 
Norm Regularization for Efficient Matrix Completion,” IEEE Transactions on 
Image Processing, submitted, 2017.

It is partially composed of TNNR code implementation. We would like to thank 
Dr. Debing Zhang for sharing his code.

Reference:
Y. Hu, D. Zhang, J. Ye, X. Li, and X. He, “Fast and accurate matrix completion 
via truncated nuclear norm regularization,” IEEE Transactions on Pattern 
Analysis Machine Intelligence, vol. 35, no. 9, pp. 2117–2130, 2013.

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

For the algorithm interpretation, please read our Xue et al. DW-TNNR paper 
(2017), in which more details are demonstrated.

If you have any questions about this implementation, please do not hesitate 
to contact me.

Ph.D. Candidate, Shengke Xue, 
College of Information Science and Electronic Engineering,
Zhejiang University, Hangzhou, P. R. China,
e-mail: (either one is o.k.)
xueshengke@zju.edu.cn, xueshengke1993@gmail.com.