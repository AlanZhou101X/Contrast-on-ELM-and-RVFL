'''
option.n_types:
               The Total number of label types in the dataset.
option.seed:
                Random seeds
option.N:
                The number of hidden neurons.
option.L:
                The number of hidden layers.
option.C:
                The regularization parameter λ
option.nCV:
                The number of folds in cross validation
option.scale
                Linearly scale the random features before feedinto the nonlinear activation function.
                in this implementation, we consider the threshold which lead to 0.99 of the maximum/minimum value of the activation function as the saturating threshold.
                option.scale=0.9 means all the random features will be linearly scaled
                into 0.9* [lower_saturating_threshold,upper_saturating_threshold].

'''

class option:
    def __init__(self, N, L, C, scale, seed, nCV, n_types):
        self.N = N
        self.L = L
        self.C = C
        self.nCV = nCV
        self.scale = scale
        self.seed = seed
        self.n_types = n_types