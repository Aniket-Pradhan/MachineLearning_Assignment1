import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import paths

class second:

    def read_data(self):
        filename = "/best_fit_line_data.csv"
        with open(self.path.Linear_Regression + filename, 'r') as file:
            self.data = pd.read_csv(file)
        s = self.data.iloc[:,0]
        self.X = s.to_numpy()
        X = []
        for i in self.X:
            X.append([i])
        self.X = np.array(X)
        ones = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((ones, self.X),axis=1)
        # normalizing data
        # self.X = (self.X - self.X.mean())/self.X.std()
        s = self.data.iloc[:,1]
        self.Y = s.to_numpy()
        Y = []
        for i in self.Y:
            Y.append([i])
        self.Y = np.array(Y)

    def error_function(self):
        error_values = np.power(np.mean(np.power(((self.X @ self.theta.T) - self.Y), 2)), 0.5)
        return error_values
    
    def gradient_descent(self):
        for i in range(self.iters):
            # T = T - (\alpha/2N) * X*(XT - Y)
            self.theta = self.theta - (self.step_size/len(self.X)) * np.sum(self.X * (self.X @ self.theta.T - self.Y), axis=0)
        errors = self.error_function()
        print(errors)
    
    def error_function_ridge(self):
        regularization = 0
        for i in self.theta_ridge.T.tolist():
            regularization += i[0]**2
        error_values = np.sqrt(np.mean(((self.X @ self.theta_ridge.T) - self.Y)**2))
        error_values += self.alpha_ridge * regularization / len(self.X)
        return error_values

    def gradient_descent_ridge(self):
        for i in range(self.iters):
            # gradient descent
            # T = T - (\alpha/2N) * X*(XT - Y) + \alpha'*T
            self.theta_ridge = self.theta_ridge - (self.step_size/len(self.X)) * (self.X.T @ (self.X @ self.theta_ridge.T - self.Y)).T - ((self.alpha_ridge/len(self.X)) * self.theta_ridge)
        errors = self.error_function_ridge()
        print(errors)

    def error_function_lasso(self):
        regularization = 0
        for i in self.theta_lasso.T.tolist():
            regularization += i[0]
        error_values = np.sqrt(np.mean(((self.X @ self.theta_ridge.T) - self.Y)**2))
        error_values += self.alpha_lasso * regularization / len(self.X)
        return error_values

    def gradient_descent_lasso(self):
        for i in range(self.iters):
            self.theta_lasso = self.theta_lasso - (self.step_size/len(self.X)) * (self.X.T @ (self.X @ self.theta_lasso.T - self.Y)).T - ((self.alpha_lasso/len(self.X)) * np.sign(self.theta_lasso))
        errors = self.error_function()
        print(errors)

    def __init__(self):
        self.path = paths()

        self.iters = 10000
        self.step_size = 0.0001
        self.alpha_ridge = 0.6
        self.alpha_lasso = 0.6

        self.theta = np.zeros([1,2]) # the parameter
        self.theta_ridge = np.zeros([1,2]) # the parameter
        self.theta_lasso = np.zeros([1,2]) # the parameter

        self.read_data()
        self.gradient_descent()
        self.read_data()
        self.gradient_descent_ridge()
        self.read_data()
        self.gradient_descent_lasso()

        plt.scatter(self.X[:,1], self.Y)
        
        Y_reg = self.X @ self.theta.T
        Y_reg_ridge = self.X @ self.theta_ridge.T
        Y_reg_lasso = self.X @ self.theta_lasso.T
        
        plt.plot(self.X[:,1], Y_reg, 'g')
        plt.plot(self.X[:,1], Y_reg_ridge, 'r')
        plt.plot(self.X[:,1], Y_reg_lasso, 'm')
        
        plt.legend(["Normal Linear Regression", "Ridge Linear Regression", "Lasso Linear Regression"])

        plt.show()

if __name__ == "__main__":
    second = second()