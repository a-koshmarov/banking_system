class SVM():
    def __init__(self, max_iter=1000, kernel_type='linear', C=1.0, epsilon=0.00001):
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic,
            'gaussian' : self.kerel_gaussian,
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):
        # varaibles
        self.alpha = []
        self.tol = 0.001
        self.eps = 0.001
        self.sigma = 2.
        self.b = 0

        self.X = X
        self.y = y

        num_changed = 0
        examine_all = 1

        # main loop
        while (num_changed > 0 or examine_all):
            num_changed = 0
            if examine_all:
                for i in range(len(X)):
                    # num_changed += examine_example(i)
                    pass
            else: 
                for i in range(len(X)):
                    if (self.alpha[i] != 0 and self.alpha[i] != self.C):
                        # num_changed += examine_example(i)
                        pass
            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

    def examine_example(self, i):
        y1 = self.y[i]
        a1 = self.alpha[i]

        if (a1 > 0 and a1 < self.C):
            E1 = error_cache[i]
        else:
            E1 = learned_func(i) - y1
        
        r1 = y1 * E1
        if ((r1 < -self.tol and a1 < self.C)
            or (r1 > self.tol and a1 > 0)):
            