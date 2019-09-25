import os
import pickle
import numpy as np
import pandas as pd
import chart_studio.plotly as py
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

from paths import paths
from download_datasets import download_datasets

class main:

    def check_create_directory(self, path_given):
        if not os.path.isdir(path_given):
            os.makedirs(path_given)
    
    def pickle_save(self, model, model_name):
        self.check_create_directory(self.path.abalone_models)
        with open(str(self.path.abalone_models) + "/" + str(self.question_number) + str(self.question_part) + str(model_name), 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def pickle_load(self, model_name):
        try:
            with open(str(self.path.abalone_models) + "/" + str(self.question_number) + str(self.question_part) + str(model_name), 'rb') as handle:
                model = pickle.load(handle)
        except:
            print("File not found error. Exiting.")
            exit()
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
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        # self.plot_features()
        
        # normalizing data
        # self.data = (self.data - self.data.mean())/self.data.std()
        
        # split data into multiple frames
        self.data_k_split = np.array_split(self.data, self.k)
    
    def error_function(self):
        # Error function: (1/2N) * (XT - Y)^2 where T is theta
        error_values = np.power(np.mean(np.power(((self.X @ self.theta.T) - self.Y), 2)), 0.5)
        return error_values
    
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
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.test_set.iloc[:,8:9].values
        self.get_errors()
        test_error = self.error_function()
        self.final_test_error.append(test_error)
        print("Testing error for fold number: = ", self.testing_index, ": ", test_error)

    def linear_regression_train(self):
        self.X = self.training_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
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
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
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
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
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
        self.skip_train = False
        if self.question_part == 'aa':
            model_prefix = self.question_number + self.question_part
            check_models = []
            for model in range(self.k):
                check_models.append(model_prefix + str(model))
            self.skip_train = any(True for model in check_models if model in ls_models)
    
    def plot_errors_part_ab(self):
        self.question_part = 'aa'
        atest_errors = self.pickle_load("test_errors")
        atrain_errors = self.pickle_load("train_errors")
        self.question_part = 'ab'
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

    def generate_train_test_set(self):
        self.test_set = self.data_k_split[self.lowest_val_error_index]
        self.remaining_data = pd.DataFrame(columns = self.columns)
        for data_frame_index in range(self.k):
            if data_frame_index == self.lowest_val_error_index:
                continue
            self.remaining_data = pd.concat([self.remaining_data, self.data_k_split[data_frame_index]])
        
        # tune the alpha_ridge based on the 80% remaining data.
        self.X = self.remaining_data.iloc[:,0:8].values
        # normalizing self.X
        # self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.remaining_data.iloc[:,8:9].values
        # split the remaining 80% data for training and testing.
        self.remaining_data = np.concatenate((self.X, self.Y), axis = 1)
        self.remaining_data = pd.DataFrame(data = self.remaining_data, columns=self.columns)
        self.remaining_data = np.array_split(self.remaining_data, self.k)

    def tune_param_ridge(self):
        print("Finding the best hyperparameter for ridge regularization")
        self.alphas = np.logspace(-1, 1, 1000)
        # alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
        tuned_parameters = [{'alpha': self.alphas}]
        model = Ridge()
        self.grid = GridSearchCV(cv = self.k, estimator=model, param_grid=tuned_parameters)
        # print(grid.get_params())
        self.grid.fit(self.X, self.Y)
        # print(self.grid.best_params_)
        print('Best hyperparameter for Ridge regularization: ', self.grid.best_estimator_.alpha)
        self.alpha_ridge = self.grid.best_estimator_.alpha
        self.scores = self.grid.cv_results_['mean_test_score']
        self.scores_std = self.grid.cv_results_['std_test_score']
    
    def error_function_ridge(self):
        regularization = 0
        for i in self.theta.T.tolist():
            regularization += i[0]**2
        error_values = np.sqrt(np.mean(((self.X @ self.theta.T) - self.Y)**2))
        error_values += self.alpha_ridge * regularization / len(self.X)
        return error_values

    def gradientDescent_ridge(self):
        errors = np.zeros(self.iters)
        for i in range(self.iters):
            # gradient descent
            # T = T - (\alpha/2N) * X*(XT - Y) + \alpha'*T
            self.theta = self.theta - (self.alpha/len(self.X)) * (self.X.T @ (self.X @ self.theta.T - self.Y)).T - ((self.alpha_ridge/len(self.X)) * self.theta)
            self.thetas.append(self.theta)
            errors[i] = self.error_function_ridge()
        self.pickle_save(self.theta, self.testing_index)
        self.pickle_save(self.thetas, str(self.testing_index) + "_")
        return errors

    def linear_regression_train_ridge(self):
        self.X = self.training_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.training_set.iloc[:,8:9].values
        self.theta = np.zeros([1,9]) # the parameters
        errors = self.gradientDescent_ridge()
        self.train_errors.extend(errors)
        train_error = self.error_function_ridge()
        self.final_train_error.append(train_error)
        print("Training error for fold number: = ", self.testing_index, ": ", train_error)

    def linear_regression_validation_ridge(self):
        self.X = self.validation_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.validation_set.iloc[:,8:9].values
        self.get_errors()
        test_error = self.error_function_ridge()
        self.final_test_error.append(test_error)
        print("Validation error for fold number: = ", self.testing_index, ": ", test_error)
    
    def linear_regression_test_ridge(self):
        self.X = self.test_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.test_set.iloc[:,8:9].values
        test_error = self.error_function_ridge()
        print("Test error: = ", test_error)

    def linear_regression_ridge(self):
        self.train_errors = []
        self.test_errors = []
        self.final_train_error = []
        self.final_test_error = []
        skip = False
        
        for i in range(self.k):
            self.thetas = []
            self.testing_index = i
            self.validation_set = pd.DataFrame(columns = self.columns)
            self.training_set = pd.DataFrame(columns = self.columns)
            for data_frame_index in range(self.k):
                if data_frame_index == self.testing_index:
                    self.validation_set = pd.concat([self.validation_set, self.remaining_data[data_frame_index]])
                else:
                    self.training_set = pd.concat([self.training_set, self.remaining_data[data_frame_index]])
            if skip:
                self.theta = self.pickle_load(self.testing_index)
                self.thetas = self.pickle_load(str(self.testing_index) + "_")
            else:
                self.linear_regression_train_ridge()
            self.linear_regression_validation_ridge()
            print()
        if not skip:
            print("Average train error: ", np.mean(np.array(self.final_train_error)))
            self.pickle_save(self.final_train_error, 'train_errors')
        print("Average validation error: ", np.mean(np.array(self.final_test_error)))
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

        self.linear_regression_test_ridge()

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

    def plot_tuning_ridge(self):
        plt.figure().set_size_inches(8, 6)
        plt.semilogx(self.alphas, self.scores)
        # plot error lines showing +/- std. errors of the self.scores
        std_error = self.scores_std / np.sqrt(self.k)

        plt.semilogx(self.alphas, self.scores + std_error, 'b--')
        plt.semilogx(self.alphas, self.scores - std_error, 'b--')

        # alpha=0.2 controls the translucency of the fill color
        plt.fill_between(self.alphas, self.scores + std_error, self.scores - std_error, alpha=0.2)

        plt.ylabel('CV score +/- std error')
        plt.xlabel('alpha')
        plt.axhline(np.max(self.scores), linestyle='--', color='.5')
        plt.xlim([self.alphas[0], self.alphas[-1]])
        plt.show()

    def tune_param_lasso(self):
        print("Finding the best hyperparameter for lasso regularization")
        self.alphas = np.logspace(-1, 1, 1000)
        # alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
        tuned_parameters = [{'alpha': self.alphas}]
        model = Lasso()
        self.grid = GridSearchCV(cv = self.k, estimator=model, param_grid=tuned_parameters)
        # print(grid.get_params())
        self.grid.fit(self.X, self.Y)
        # print(self.grid.best_params_)
        print('Best hyperparameter for lasso regularization: ', self.grid.best_estimator_.alpha)
        self.alpha_lasso = self.grid.best_estimator_.alpha
        self.scores = self.grid.cv_results_['mean_test_score']
        self.scores_std = self.grid.cv_results_['std_test_score']
    
    def error_function_lasso(self):
        regularization = 0
        for i in self.theta.T.tolist():
            regularization += i[0]
        error_values = np.sqrt(np.mean(((self.X @ self.theta.T) - self.Y)**2))
        error_values += self.alpha_lasso * regularization / len(self.X)
        return error_values

    def gradientDescent_lasso(self):
        errors = np.zeros(self.iters)
        for i in range(self.iters):
            self.theta = self.theta - (self.alpha/len(self.X)) * (self.X.T @ (self.X @ self.theta.T - self.Y)).T - ((self.alpha_lasso/len(self.X)) * np.sign(self.theta))
            self.thetas.append(self.theta)
            errors[i] = self.error_function()
        self.pickle_save(self.theta, self.testing_index)
        self.pickle_save(self.thetas, str(self.testing_index) + "_")
        return errors

    def linear_regression_train_lasso(self):
        self.X = self.training_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.training_set.iloc[:,8:9].values
        self.theta = np.zeros([1,9]) # the parameters
        errors = self.gradientDescent_lasso()
        self.train_errors.extend(errors)
        train_error = self.error_function()
        self.final_train_error.append(train_error)
        print("Training error for fold number: = ", self.testing_index, ": ", train_error)

    def linear_regression_validation_lasso(self):
        self.X = self.validation_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.validation_set.iloc[:,8:9].values
        self.get_errors()
        test_error = self.error_function()
        self.final_test_error.append(test_error)
        print("Validation error for fold number: = ", self.testing_index, ": ", test_error)
    
    def linear_regression_test_lasso(self):
        self.X = self.test_set.iloc[:,0:8]
        ones = np.ones([self.X.shape[0],1])
        self.X = np.concatenate((ones, self.X),axis=1)
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.test_set.iloc[:,8:9].values
        test_error = self.error_function()
        print("Test error: = ", test_error)

    def linear_regression_lasso(self):
        self.train_errors = []
        self.test_errors = []
        self.final_train_error = []
        self.final_test_error = []
        skip = False
        
        for i in range(self.k):
            self.thetas = []
            self.testing_index = i
            self.validation_set = pd.DataFrame(columns = self.columns)
            self.training_set = pd.DataFrame(columns = self.columns)
            for data_frame_index in range(self.k):
                if data_frame_index == self.testing_index:
                    self.validation_set = pd.concat([self.validation_set, self.remaining_data[data_frame_index]])
                else:
                    self.training_set = pd.concat([self.training_set, self.remaining_data[data_frame_index]])
            if skip:
                self.theta = self.pickle_load(self.testing_index)
                self.thetas = self.pickle_load(str(self.testing_index) + "_")
            else:
                self.linear_regression_train_lasso()
            self.linear_regression_validation_lasso()
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

        self.linear_regression_test_lasso()

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

    def plot_tuning_lasso(self):
        plt.figure().set_size_inches(8, 6)
        plt.semilogx(self.alphas, self.scores)
        # plot error lines showing +/- std. errors of the self.scores
        std_error = self.scores_std / np.sqrt(self.k)

        plt.semilogx(self.alphas, self.scores + std_error, 'b--')
        plt.semilogx(self.alphas, self.scores - std_error, 'b--')

        # alpha=0.2 controls the translucency of the fill color
        plt.fill_between(self.alphas, self.scores + std_error, self.scores - std_error, alpha=0.2)

        plt.ylabel('CV score +/- std error')
        plt.xlabel('alpha')
        plt.axhline(np.max(self.scores), linestyle='--', color='.5')
        plt.xlim([self.alphas[0], self.alphas[-1]])
        plt.show()

    def __init__(self):
        self.path = paths()
        self.dl_data = download_datasets()
        self.datapath = self.path.abalone_dataset + "/Dataset.data"
        self.alpha = .1
        self.iters = 1000
        self.k = 5

        self.read_data()
        self.question_number = '1'

        # Part a
        self.question_part = 'aa'
        self.check_pre_models()
        self.linear_regression()

        # Part b
        input("Press enter for the next part")
        self.question_part = 'ab'
        self.check_pre_models()
        self.linear_regression_closed_form()

        # Part c
        input("Press enter for the next part")
        self.question_part = 'ac'
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
        # k = 1 gives the least validation error. Therefore using k = 1 as the test set, and using the rest to generate
        # training and validation set.
        input("Press enter for the next part")
        self.question_part = 'ba'
        self.lowest_val_error_index = 1 # between 0 to k-1
        self.generate_train_test_set() # generates self.remaining_data (80%) and self.test_data
        self.tune_param_ridge()
        # self.plot_tuning_ridge()
        # self.alpha_ridge = 0.7996554525892349
        # self.alpha_ridge = 1.8761746914391204
        self.linear_regression_ridge()

        input("Press enter for the next part")
        self.question_part = 'bb'
        self.lowest_val_error_index = 1
        self.generate_train_test_set()
        self.tune_param_lasso()
        # self.alpha_lasso = 0.10280447320933092
        self.linear_regression_lasso()


if __name__ == "__main__":
    main = main()
