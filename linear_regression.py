class LinearRegression:
    def __init__(self, alpha=0.01, iteration=10000, fit_intercept=True):
        self.alpha = alpha
        self.iteration = iteration
        self.fit_intercept = fit_intercept
    
    
    def fit(self, x, y):
        self.x = x
        self.y = y
            
        if self.fit_intercept:
            self.x = self._add_intercept(self.x)
            
        self.samples = self.x.shape[0]
        self.coef = np.ones(self.x.shape[1])
            
        self._gradient_descend()
        
        
    def _add_intercept(self, x):
        return np.insert(x, 0, 1, 1)
        
        
    def _gradient_descend(self):
        self.list_of_cost_func = []
        
        for j in range(self.iteration):
            new_coef = self.coef

            for i in range(len(self.coef)):
                new_coef[i] = self.coef[i] - self.alpha * ( (self.x.T[i]) @ (self.x @ self.coef - self.y) / self.samples) 
            
            self.coef = new_coef
            
            sum_of_residual = np.sum((self.x @ self.coef - self.y)**2) / self.samples
            self.list_of_cost_func.append(sum_of_residual)
            
    
    def visualization(self):
        plt.plot(range(self.iteration), self.list_of_cost_func)
        plt.xlabel('Number of Iteration')
        plt.ylabel('Sum of Residual')
        plt.show()
    
    
    def predict(self, x_test):
        if self.fit_intercept:
            x_test = self._add_intercept(x_test)
            
        return x_test @ self.coef
    
    
    def accuracy(self, y_test, y_pred):
        rss = np.sum((y_test - y_pred) ** 2)
        tss = np.sum((y_test - np.mean(y_test)) ** 2)
        
        r_squared = 1 - (rss / tss)
        return r_squared
        
