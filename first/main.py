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
        self.features = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"]
        self.data.columns = self.features
        self.features.pop(len(self.features)-1)
        mapping = {'M': 1, 'F': 2, 'I': 3}
        self.data = self.data.replace({'sex': mapping})
        # self.plot_features()

        # normalizing data
        self.data = (self.data - self.data.mean())/self.data.std()
        # print(self.data)
    
    def cost(self):
        tobesummed = np.power(((self.X @ self.theta.T) - self.Y),2)
        return np.sum(tobesummed)/(2 * len(self.X))
    
    def gradientDescent(self):
        cost = np.zeros(self.iters)
        for i in range(self.iters):
            self.theta = self.theta - (self.alpha/len(self.X)) * np.sum(self.X * (self.X @ self.theta.T - self.Y), axis=0)
            cost[i] = self.cost()
        return cost

    def linear_regression(self):
        self.read_data()
        self.X = self.data.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        self.Y = self.data.iloc[:,8:9].values
        self.theta = np.zeros([1,9])
        #running the gd and cost function
        temp_cost = self.gradientDescent()
        finalCost = self.cost()
        print(finalCost)
        
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.iters), temp_cost, 'r')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
        ax.set_title('Error vs. Training Epoch')
        
        # plt.show()

    def __init__(self):
        self.path = paths()
        self.dl_data = download_datasets()
        
        self.datapath = self.path.abalone_dataset + "/Dataset.data"
        self.alpha = 0.3
        self.iters = 1000
        self.linear_regression()

if __name__ == "__main__":
    main = main()
