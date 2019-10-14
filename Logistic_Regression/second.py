import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve

from paths import paths
from download_datasets import download_datasets

class second:

    def check_create_directory(self, path_given):
        if not os.path.isdir(path_given):
            os.makedirs(path_given)

    def pickle_save(self, model, model_name):
        self.check_create_directory(self.path.mnist_models)
        with open(str(self.path.mnist_models) + "/" + str(model_name), 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def pickle_load(self, model_name):
        try:
            with open(str(self.path.mnist_models) + "/" + str(model_name), 'rb') as handle:
                model = pickle.load(handle)
        except:
            print("File not found error. Exiting.")
            exit()
        return model
    
    def check_prev_models(self):
        if os.path.isdir(self.path.mnist_models):
            ls = os.listdir(self.path.mnist_models)
            if self.reg_ridge_name in ls:
                self.check_ridge = True
            if self.reg_lasso_name in ls:
                self.check_lasso = True


    def load_training_data(self):
        self.X, self.Y = loadlocal_mnist(images_path=self.path.mnist + "/train_images", labels_path=self.path.mnist + "/train_labels")
        self.X = (self.X - self.X.mean())/self.X.std()
    
    def load_testing_data(self):
        self.X_test, self.Y_test = loadlocal_mnist(images_path=self.path.mnist + "/test_images", labels_path=self.path.mnist + "/test_labels")
        self.X_test = (self.X_test - self.X_test.mean())/self.X_test.std()
    
    def logistic_regression_train_ridge(self):
        print("Training ridge model")
        self.reg_ridge = LogisticRegression(penalty='l2', random_state=0, solver='saga', multi_class='ovr').fit(self.X, self.Y)
        self.pickle_save(self.reg_ridge, self.reg_ridge_name)
    
    def logistic_regression_train_lasso(self):
        print("Training lasso model")
        self.reg_lasso = LogisticRegression(penalty='l1', random_state=0, solver='saga', multi_class='ovr').fit(self.X, self.Y)
        self.pickle_save(self.reg_lasso, self.reg_lasso_name)
        # print(reg_ridge.score(self.X, self.Y))
    
    def train_error(self):
        num_elements = self.X.shape[0]
        num_correct_classifications_ridge = []
        num_correct_classifications_lasso = []
        num_classifications = []
        for class_index in range(len(self.classes)):
            num_classifications.append(0)
            num_correct_classifications_ridge.append(0)
            num_correct_classifications_lasso.append(0)

        for element_index in range(num_elements):
            train_X = self.X[element_index].reshape(1,-1)
            train_Y = self.Y[element_index]
            pred_ridge = self.reg_ridge.predict(train_X)
            pred_lasso = self.reg_lasso.predict(train_X)
            train_Y = int(train_Y)
            if pred_ridge == train_Y:
                num_correct_classifications_ridge[train_Y] += 1
            if pred_lasso == train_Y:
                num_correct_classifications_lasso[train_Y] += 1
            num_classifications[train_Y] += 1
        self.train_ridge_avg_list = []
        self.train_lasso_avg_list = []
        for class_index in range(len(self.classes)):
            self.train_ridge_avg_list.append(num_correct_classifications_ridge[class_index]/num_classifications[class_index])
            self.train_lasso_avg_list.append(num_correct_classifications_lasso[class_index]/num_classifications[class_index])
        # print("average error:", self.train_ridge_avg_list)
        # print("average error:", self.train_lasso_avg_list)
    
    def test_error(self):
        num_elements = self.X_test.shape[0]
        num_correct_classifications_ridge = []
        num_correct_classifications_lasso = []
        num_classifications = []
        for class_index in range(len(self.classes)):
            num_classifications.append(0)
            num_correct_classifications_ridge.append(0)
            num_correct_classifications_lasso.append(0)

        for element_index in range(num_elements):
            train_X = self.X_test[element_index].reshape(1,-1)
            train_Y = self.Y_test[element_index]
            pred_ridge = self.reg_ridge.predict(train_X)
            pred_lasso = self.reg_lasso.predict(train_X)
            train_Y = int(train_Y)
            if pred_ridge == train_Y:
                num_correct_classifications_ridge[train_Y] += 1
            if pred_lasso == train_Y:
                num_correct_classifications_lasso[train_Y] += 1
            num_classifications[train_Y] += 1
        self.test_ridge_avg_list = []
        self.test_lasso_avg_list = []
        for class_index in range(len(self.classes)):
            self.test_ridge_avg_list.append(num_correct_classifications_ridge[class_index]/num_classifications[class_index])
            self.test_lasso_avg_list.append(num_correct_classifications_lasso[class_index]/num_classifications[class_index])
        # print("average error:", self.train_ridge_avg_list)
        # print("average error:", self.train_lasso_avg_list)
    
    def pretty_print_accuracy(self):
        num_elements = len(self.classes)
        for ind in range(num_elements):
            print("Training accuracy for class " + str(ind) + " for L1 regularization: ", self.train_lasso_avg_list[ind])
            print("Training accuracy for class " + str(ind) + " for L2 regularization: ", self.train_ridge_avg_list[ind])
            print("testing accuracy for class " + str(ind) + " for L1 regularization: ", self.test_lasso_avg_list[ind])
            print("testing accuracy for class " + str(ind) + " for L2 regularization: ", self.test_ridge_avg_list[ind])
            print()
    
    def plot_roc(self):
        Y = label_binarize(self.Y_test, classes = self.classes)
        
        for class_index in self.classes:
            y = []
            for element_index in range(self.X_test.shape[0]):
                a = self.reg_ridge.predict(self.X_test[element_index].reshape(1,-1))
                if a == class_index:
                    a = 1
                else:
                    a = 0
                y.append(a)
            y = np.array(y)
            fpr, tpr, thresholds = roc_curve(Y[:, class_index], y)
            lw = 2
            plt.plot(fpr, tpr, lw=lw, label = "ROC curve for class:" + str(class_index))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Graph')
        plt.legend(loc='lower right')
        plt.show()

    def __init__(self):
        self.reg_ridge_name = "reg_ridge_model"
        self.reg_lasso_name = "reg_lasso_model"
        self.check_ridge = False
        self.check_lasso = False
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.path = paths()
        self.dl_data = download_datasets()
        self.check_prev_models()

        self.load_training_data()
        self.load_testing_data()
        retrain = False

        if self.check_ridge:
            print("Using cached models from " + self.path.mnist_models + ". retraining the model would consume a lot of time")
            retrain = input("Do you want to retrain the model ('y' for yes, default is no)?: ")
            if retrain == 'y' or retrain == 'Y':
                retrain = True
            else:
                retrain = False
        if retrain:
            self.logistic_regression_train_ridge()
        else:
            self.reg_ridge = self.pickle_load(self.reg_ridge_name)
        
        if self.check_lasso:
            print("Using cached models from " + self.path.mnist_models + ". retraining the model would consume a lot of time")
            retrain = input("Do you want to retrain the model ('y' for yes, default is no)?: ")
            if retrain == 'y' or retrain == 'Y':
                retrain = True
            else:
                retrain = False
        if retrain:
            self.logistic_regression_train_lasso()
        else:
            self.reg_lasso = self.pickle_load(self.reg_lasso_name)

        # part ii
        self.train_error()
        self.test_error()
        self.pretty_print_accuracy()

        # part iii
        self.plot_roc()

if __name__ == "__main__":
    second = second()
