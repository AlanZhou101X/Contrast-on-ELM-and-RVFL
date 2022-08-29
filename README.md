# Contrast on ELM and RVFL
 A Simple Contrast Experiment on RVFL^1,2^, edRVFL^3^,  ELM^4,5^ and MELM^6^ in python.

We have reimplemented the classification tasks in these codes on our own data set. 

## Introduction of Parameters 

```
n_types:
               The Total number of label types in the dataset.
seed:
                Random seeds.
N/n_nodes:
                The number of hidden neurons.
L:
                The number of hidden layers.
C/lam:
                The regularization parameter λ
nCV:
                The number of folds in cross validation.
scale：
                Linearly scale the random features before feedinto the nonlinear activation function. In this implementation, we consider the threshold which lead to 0.99 of the maximum/minimum value of the activation function as the saturating threshold. option.scale=0.9 means all the random features will be linearly scaledinto 0.9* [lower_saturating_threshold,upper_saturating_threshold].
w_random_vec_range: 
				A list, [min, max], the range of generating random weights.
b_random_vec_range: 
				A list, [min, max], the range of generating random bias.
random_weights:
				A Numpy array shape is [n_feature, n_nodes], weights of neuron.
random_bias: 
				A Numpy array shape is [n_nodes], bias of neuron.
beta: 
				A Numpy array shape is [n_feature + n_nodes, n_class], the 
task_type: 
				A string of ML task type, 'classification' or 'regression'.
```

## Paper Reference

> 1. Pao, Yoh-Han, Gwang-Hoon Park, and Dejan J. Sobajic. "Learning and generalization characteristics of the random vector functional-link net." *Neurocomputing* 6.2 (1994): 163-180.
> 2. Zhang, Le, and Ponnuthurai N. Suganthan. "A comprehensive evaluation of random vector functional link networks." *Information sciences* 367 (2016): 1094-1105.
> 3. Shi, Qiushi, et al. "Random vector functional link neural network based ensemble deep learning." *Pattern Recognition* 117 (2021): 107978.
> 4. Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory and applications." *Neurocomputing* 70.1-3 (2006): 489-501.
> 5. Fernández-Navarro, Francisco, et al. "MELM-GRBF: A modified version of the extreme learning machine for generalized radial basis function neural networks." *Neurocomputing* 74.16 (2011): 2502-2510.
> 6. Xiao, Dong, Beijing Li, and Yachun Mao. "A multiple hidden layers extreme learning machine method and its application." *Mathematical Problems in Engineering* 2017 (2017).









