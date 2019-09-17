import os
import pickle
import numpy as np
import pandas as pd
import chart_studio.plotly as py
import matplotlib.pyplot as plt
from matplotlib import style

from paths import paths
from download_datasets import download_datasets

class main:

    def check_create_directory(self, path_given):
        if not os.path.isdir(path_given):
            os.makedirs(path_given)
    
    def pickle_save(self, model, model_name):
        self.check_create_directory(self.path.abalone_models)
        with open(str(self.path.abalone_models) + "/" + str(self.question) + str(model_name), 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def pickle_load(self, model_name):
        with open(str(self.path.abalone_models) + "/" + str(self.question) + str(model_name), 'rb') as handle:
            model = pickle.load(handle)
        return model

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
        self.data_k_split = np.array_split(self.data, self.k)
    
    def error_function(self):
        # Error function: (1/2N) * (XT - Y)^2 where T is theta
        error_values = np.power(((self.X @ self.theta.T) - self.Y), 2)
        return np.sum(error_values)/(2 * len(self.X))
    
    def get_errors(self):
        for theta in self.thetas:
            error_values = np.power(((self.X @ theta.T) - self.Y), 2)
            self.test_errors.append(np.sum(error_values)/(2 * len(self.X)))
    
    def gradientDescent(self):
        # array of error values for respective iteration.
        errors = np.zeros(self.iters)
        for i in range(self.iters):
            # gradient descent
            # T = T - (\alpha/2N) * X*(XT - Y)
            self.theta = self.theta - (self.alpha/len(self.X)) * np.sum(self.X * (self.X @ self.theta.T - self.Y), axis=0)
            self.thetas.append(self.theta)
            errors[i] = self.error_function()
        self.pickle_save(self.theta, self.testing_index)
        self.pickle_save(self.thetas, str(self.testing_index) + "_")
        return errors

    def linear_regression_test(self):
        self.X = self.test_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        self.Y = self.test_set.iloc[:,8:9].values
        self.get_errors()
        test_error = self.error_function()
        self.final_test_error.append(test_error)
        print("Testing error for fold number: = ", self.testing_index, ": ", test_error)

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
        print("Training error for fold number: = ", self.testing_index, ": ", train_error)
    
    def linear_regression(self):
        self.train_errors = []
        self.test_errors = []
        self.final_train_error = []
        self.final_test_error = []
        skip = False

        if self.skip_train:
            skip_ = input("previous cached models available. use them? (y for yes, default is no): ")
            if skip_ == 'y' or skip_ == 'Y':
                skip = True

        for i in range(self.k):
            self.thetas = []
            self.testing_index = i
            self.test_set = pd.DataFrame(columns = self.columns)
            self.training_set = pd.DataFrame(columns = self.columns)
            for data_frame_index in range(self.k):
                if data_frame_index == self.testing_index:
                    self.test_set = pd.concat([self.test_set, self.data_k_split[data_frame_index]])
                else:
                    self.training_set = pd.concat([self.training_set, self.data_k_split[data_frame_index]])
            if skip:
                self.theta = self.pickle_load(self.testing_index)
                self.thetas = self.pickle_load(str(self.testing_index) + "_")
            else:
                self.linear_regression_train()
            self.linear_regression_test()
            print()
        if not skip:
            print("Average train error: ", np.mean(np.array(self.final_train_error)))
            self.pickle_save(self.final_train_error, 'train_errors')
        print("Average test error: ", np.mean(np.array(self.final_test_error)))
        self.pickle_save(self.final_test_error, 'test_errors')

        train_errors = []
        test_errors = []
        for i in range(self.k):
            if not skip:
                train_errors.append(self.train_errors[i*self.iters:(i+1)*self.iters])
            test_errors.append(np.array(self.test_errors[i*self.iters:(i+1)*self.iters]))

        if not skip:
            self.train_errors = np.mean(np.array(train_errors), axis=0)
        self.test_errors = np.mean(np.array(test_errors), axis=0)

        # plot error vs iterations
        if not skip:
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
    
    def lr_closed_form(self):
        # β^=(X'X)^−1 * X'y
        self.theta = (np.linalg.inv(self.X.T @ self.X)) @ (self.X.T @ self.Y)
        self.theta = self.theta.T
    
    def linear_regression_closed_form_train(self):
        self.X = self.training_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X), axis=1)
        self.X = np.array(self.X, dtype = 'float')
        self.Y = self.training_set.iloc[:,8:9].values
        self.Y = np.array(self.Y, dtype = 'float')
        self.lr_closed_form()
        train_error = self.error_function()
        self.final_train_error.append(train_error)
        print("Training error for fold number: = ", self.testing_index, ": ", train_error)
    
    def linear_regression_closed_form_test(self):
        self.X = self.test_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        self.Y = self.test_set.iloc[:,8:9].values
        test_error = self.error_function()
        self.final_test_error.append(test_error)
        print("Testing error for fold number: = ", self.testing_index, ": ", test_error)

    def linear_regression_closed_form(self):
        self.final_train_error = []
        self.final_test_error = []

        for i in range(self.k):
            self.thetas = []
            self.testing_index = i
            self.test_set = pd.DataFrame(columns = self.columns)
            self.training_set = pd.DataFrame(columns = self.columns)
            for data_frame_index in range(self.k):
                if data_frame_index == self.testing_index:
                    self.test_set = pd.concat([self.test_set, self.data_k_split[data_frame_index]])
                else:
                    self.training_set = pd.concat([self.training_set, self.data_k_split[data_frame_index]])
            self.linear_regression_closed_form_train()
            self.linear_regression_test()
            print()
        print("Average train error: ", np.mean(np.array(self.final_train_error)))
        self.pickle_save(self.final_train_error, 'train_errors')
        print("Average test error: ", np.mean(np.array(self.final_test_error)))
        self.pickle_save(self.final_test_error, 'test_errors')

    def check_pre_models(self):
        model_dir = self.path.abalone_models
        self.check_create_directory(model_dir)
        ls_models = os.listdir(model_dir)
        if self.question == 'a':
            check_models = ['a0', 'a1', 'a2', 'a3', 'a4']
            self.skip_train = any(True for model in check_models if model in ls_models)
    
    def plot_errors_part_ab(self):
        self.question = 'a'
        atest_errors = self.pickle_load("test_errors")
        atrain_errors = self.pickle_load("train_errors")
        self.question = 'b'
        btrain_errors = self.pickle_load("train_errors")
        btest_errors = self.pickle_load("test_errors")

        ind = np.arange(len(atrain_errors))  # the x locations for the groups
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots()
        ax.bar(ind - width/2, atrain_errors, width, label='Gradient Descent')
        ax.bar(ind + width/2, btrain_errors, width, label='Normal Form')
        ax.set_xlabel('Different k\'s')
        ax.set_ylabel('Error')
        ax.set_title('Training RMSE between GD and Normal Form')
        ax.legend()

        fig, ax = plt.subplots()
        ax.bar(ind - width/2, atest_errors, width, label='Gradient Descent')
        ax.bar(ind + width/2, btest_errors, width, label='Normal Form')
        ax.set_xlabel('Different k\'s')
        ax.set_ylabel('Error')
        ax.set_title('Testing RMSE between GD and Normal Form')
        ax.legend()

        plt.show()

    def __init__(self):
        self.path = paths()
        self.dl_data = download_datasets()
        self.datapath = self.path.abalone_dataset + "/Dataset.data"
        self.alpha = .01
        self.iters = 10000
        self.k = 5

        self.read_data()

        # # Part a
        self.question = 'a'
        self.check_pre_models()
        self.linear_regression()

        # Part b
        self.question = 'b'
        self.check_pre_models()
        self.linear_regression_closed_form()

        # Part c
        self.question = 'c'
        self.plot_errors_part_ab()

        ## Explanation/Observation
        """
        RMSE calculated in part 'b' is less than the RMSE from part 'a'. This is because in part 'b' we are using the analytical/normal solution
        of the linear regression problem, whereas in part 'a' we are computing everything manually using the gradient descent. The gradient
        descent algorithm depends upon the step-size and the number of epochs as well. If we use a bigger step-size, we might end not end up
        at the minimum value of the required function, whereas if we use a smaller step-size, computation could take an indefinite amount of
        time. The number of iterations/epochs determine how long do we want to run the gradient descent algorithm. A smaller number, would lead
        to termincation of the program before we reach the minima, whereas a bigger number would lead to more computation.
        Hence, we the RMSE depends upon the these two parameters, and it varies as we change them.
        On the other hand, the closed/normal form produces a perfect solution to the linear regression problem, but can be computationally 
        expensive based on the size of the data, as it requires inverting matrices and their multiplication as well.
        """


if __name__ == "__main__":
    main = main()
