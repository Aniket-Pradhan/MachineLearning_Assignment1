import numpy as np
import pandas as pd
import chart_studio.plotly as py
import matplotlib.pyplot as plt
from matplotlib import style

from paths import paths
from download_datasets import download_datasets

class main:

    def plot_features(self):
        plt.figure(figsize=(10, 14))
        title = "Relationship bw %s and rings"
        i = 0
        print(len(self.features))
        for feature in self.features:
            i+=1
            plt.subplot(4,2,i)
            plt.plot(self.data[feature],self.data["rings"],marker='.',linestyle='none')
            plt.title(title % (feature))   
            plt.tight_layout()
        plt.show()

    def read_data(self):
        self.data = pd.read_csv(self.datapath, delim_whitespace=True, header=None)
        self.columns = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"]
        self.data.columns = self.columns
        self.features = self.columns[:len(self.columns)-1]
        mapping = {'M': 1, 'F': 2, 'I': 3}
        self.data = self.data.replace({'sex': mapping})
        # self.plot_features()
        
        # normalizing data
        # self.data = (self.data - self.data.mean())/self.data.std()
        
        # split data into multiple frames
        self.data = np.array_split(self.data, self.k)
    
    def error_function(self):
        # Error function: (1/2N) * (XT - Y)^2 where T is theta
        tobesummed = np.power(((self.X @ self.theta.T) - self.Y), 2)
        return np.sum(tobesummed)/(2 * len(self.X))
    
    def get_errors(self):
        for theta in self.thetas:
            tobesummed = np.power(((self.X @ theta.T) - self.Y), 2)
            self.test_errors.append(np.sum(tobesummed)/(2 * len(self.X)))
    
    def gradientDescent(self):
        # array of error values for respective iteration.
        errors = np.zeros(self.iters)
        for i in range(self.iters):
            # gradient descent
            # T = T - (\alpha/2N) * X*(XT - Y)
            self.theta = self.theta - (self.alpha/len(self.X)) * np.sum(self.X * (self.X @ self.theta.T - self.Y), axis=0)
            self.thetas.append(self.theta)
            errors[i] = self.error_function()
        return errors

    def linear_regression_test(self):
        self.X = self.test_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        self.Y = self.test_set.iloc[:,8:9].values
        self.get_errors()
        test_error = self.error_function()
        self.final_test_error.append(test_error)
        print("Testing error for k = ", self.testing_index, ": ", test_error)

    def linear_regression_train(self):
        self.X = self.training_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        self.Y = self.training_set.iloc[:,8:9].values
        self.theta = np.zeros([1,9]) # the parameters
        # gradient descent
        errors = self.gradientDescent()
        self.train_errors.extend(errors)
        train_error = self.error_function()
        self.final_train_error.append(train_error)
        print("Training error for k = ", self.testing_index, ": ", train_error)

    def __init__(self):
        self.path = paths()
        self.dl_data = download_datasets()
        
        self.datapath = self.path.abalone_dataset + "/Dataset.data"
        self.alpha = .1
        self.iters = 1000
        self.k = 5
        self.read_data()

        self.train_errors = []
        self.test_errors = []
        self.final_train_error = []
        self.final_test_error = []

        for i in range(self.k):
            self.thetas = []
            self.testing_index = i
            self.test_set = pd.DataFrame(columns = self.columns)
            self.training_set = pd.DataFrame(columns = self.columns)
            for data_frame_index in range(self.k):
                if data_frame_index == self.testing_index:
                    self.test_set = pd.concat([self.test_set, self.data[data_frame_index]])
                else:
                    self.training_set = pd.concat([self.training_set, self.data[data_frame_index]])
            self.linear_regression_train()
            self.linear_regression_test()
            print()
        print("Average train error: ", np.mean(np.array(self.final_train_error)))
        print("Average test error: ", np.mean(np.array(self.final_test_error)))

        train_errors = []
        test_errors = []
        for i in range(self.k):
            train_errors.append(self.train_errors[i*self.iters:(i+1)*self.iters])
            test_errors.append(np.array(self.test_errors[i*self.iters:(i+1)*self.iters]))

        self.train_errors = np.mean(np.array(train_errors), axis=0)
        self.test_errors = np.mean(np.array(test_errors), axis=0)

        # plot error vs iterations
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.iters), self.train_errors, 'r')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
        ax.set_title('Error vs. Training Epoch')

        fig, ax = plt.subplots()
        ax.plot(np.arange(self.iters), self.test_errors, 'r')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
        ax.set_title('Error vs. Testing Epoch')

        plt.show()


if __name__ == "__main__":
    main = main()
