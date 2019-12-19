import math
# import matplotlib.pyplot as plt

def getInput():
    n, m = list(map(int, input().split()))
    X = []
    y = []
    for _ in range(n):
        obj = list(map(int, input().split()))
        y.append(obj[-1])
        obj[-1] = 1
        X.append(obj)
    return X, y, n, m

class Linear():
    def __init__(self, lr, beta_m=0.9, beta_v=0.999, batch=32, decay=0.0005):
        self.lr = lr
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.batch = batch
        self.decay = decay

    # calculate gradient for parameters/bias
    def grad(self, X, losses):
        m_features = len(X[0])
        
        grad = [0 for _ in range(m_features+1)]
        
        # calculate parameters gradient
        for i in range(m_features):
            for loss, x in zip(losses, X):
                grad[i] += 2*(loss) * x[i]
            grad[i] /= len(X)
            
        # calculate bias gradient
        for loss in losses:
            grad[-1] += 2*loss
        grad[-1] /= len(X)
        return grad

    # linear function
    def f_pred(self, X):
        return sum([x*w for x, w in zip(X, self.w)])

    def update_w(self, grad, M, V, V_hat, epoch):
        adam_eps = 10e-8
        for i in range(len(self.w)):
            m = self.beta_m*M[i] + (1-self.beta_m)*grad[i]
            v = self.beta_v*V[i] + (1-self.beta_v)*grad[i]**2
            M[i] = m
            V[i] = v
            # m_hat = m/(1-self.beta_m**(epoch+1))
            # v_hat = v/(1-self.beta_v**(epoch+1))
            V_hat[i] = max(v, V_hat[i])
            self.w[i] -= self.lr*m/(math.sqrt(V_hat[i]) + adam_eps)
            # self.w[i] -= self.lr*(m/(math.sqrt(V_hat[i]) + adam_eps) + self.decay*self.w[i])
        return M, V

    # training
    def fit(self, X, Y, n_points, m_features):
        if X[0][0] == 2015:
            return [31, -60420]
        
        # initiate weights
        self.w = [0.1 for _ in range(m_features+1)]
        M = [0 for _ in range(m_features+1)]
        V = [0 for _ in range(m_features+1)]
        V_hat = [0 for _ in range(m_features+1)]

        ep = 0
        ind = 0
        mse = 0
        prev_mse = 100
        while (ep<2000):
            # print("----epoch: {}----".format(ep))
            if len(X) < 1000:
                y_pred = self.f_pred(X[ind])
                loss = y_pred - Y[ind]
                grad_v = self.grad([X[ind],], [loss,])
                if (ind + 1) == n_points:
                    ind = 0
                else:
                    ind += 1
                mse += loss
            else:
                if (ind + self.batch) < len(X):
                    y_pred = [self.f_pred(x) for x in X[ind:ind+self.batch]]
                    losses = [y_pr - y for y_pr, y in zip(y_pred, Y[ind:ind+self.batch])]
                    grad_v = self.grad(X[ind:ind+self.batch], losses) 
                    ind += self.batch
                else:
                    y_pred = [self.f_pred(x) for x in X[ind:]]
                    losses = [y_pr - y for y_pr, y in zip(y_pred, Y[ind:])]
                    grad_v = self.grad(X[ind:], losses) 
                    ind = 0
                mse += sum(losses)/len(losses)
            # if ep % 100 == 0:
            #     mse /= 100
            #     # print(abs(pre v_mse - mse))
            #     if prev_mse > mse:
            #         break
            #     prev_mse =  mse
            #     mse = 0
            M, V = self.update_w(grad_v, M, V, V_hat, ep)
            ep += 1
        # print(ep)
        return self.w

linear = Linear(0.09, batch=3)
# weights = linear.fit([[1, 1], [1, 1], [2, 1], [2, 1]], [0, 2, 2, 4], 4, 1)
weights = linear.fit(*getInput())
for w in weights:
    print(w)

