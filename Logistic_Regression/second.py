import os
import pickle
from mlxtend.data import loadlocal_mnist
from sklearn.linear_model import LogisticRegression

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
        print(self.X.shape, self.Y.shape)
        self.X = (self.X - self.X.mean())/self.X.std()
    
    def logistic_regression_train_ridge(self):
        print("Training ridge model")
        self.reg_ridge = LogisticRegression(penalty='l2', random_state=0, solver='saga', multi_class='ovr').fit(self.X, self.Y)
        self.pickle_save(self.reg_ridge, self.reg_ridge_name)
        # print(reg_ridge.score(self.X, self.Y))
    
    def logistic_regression_train_lasso(self):
        print("Training lasso model")
        self.reg_lasso = LogisticRegression(penalty='l1', random_state=0, solver='saga', multi_class='ovr').fit(self.X, self.Y)
        self.pickle_save(self.reg_lasso, self.reg_lasso_name)
        # print(reg_ridge.score(self.X, self.Y))

    def __init__(self):
        self.reg_ridge_name = "reg_ridge_model"
        self.reg_lasso_name = "reg_lasso_model"
        self.check_ridge = False
        self.check_lasso = False

        self.path = paths()
        self.dl_data = download_datasets()
        self.check_prev_models()

        self.load_training_data()

        if not self.check_ridge:
            self.logistic_regression_train_ridge()
        else:
            self.reg_ridge = self.pickle_load(self.reg_ridge_name)
        if not self.check_lasso:
            self.logistic_regression_train_lasso()
        else:
            self.reg_lasso = self.pickle_load(self.reg_lasso_name)

if __name__ == "__main__":
    second = second()