import numpy as np


class LinearRegression:
    def __init__(self, alpha=0.01, iteration=10000, normalize=True):
        self.alpha = alpha
        self.iteration = iteration
        self.normalize = normalize
    
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.samples = self.x.shape[0]
        self.coef = np.ones(self.x.shape[1])
        
        if self.normalize:
            self.x = self._normalization()
            
        self._gradient_descend()
        
    
    def _normalization(self):
        return (self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))
        
        
    def _gradient_descend(self):
        for j in range(self.iteration):
            new_coef = self.coef

            for i in range(len(self.coef)):
                new_coef[i] = self.coef[i] - self.alpha * ( (self.x.T[i]) @ (self.x @ self.coef - self.y) / self.samples) 

            self.coef = new_coef
    
    
    def predict(self, x_test):
        return x_test @ self.coef
    
    
    def accuracy(self, y_test, y_pred):
        rss = sum((y_test - y_pred) ** 2)
        tss = sum((y_test - np.mean(y_test)) ** 2)
        
        r_squared = 1 - (rss / tss)
        return r_squared
        
