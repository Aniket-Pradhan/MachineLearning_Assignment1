import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import paths

class main:

    def check_create_directory(self, path_given):
        if not os.path.isdir(path_given):
            os.makedirs(path_given)
    
    def pickle_save(self, model, model_name):
        self.check_create_directory(self.path.income_models)
        with open(str(self.path.income_models) + "/" + str(self.question_number) + str(self.question_part) + str(model_name), 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def pickle_load(self, model_name):
        try:
            with open(str(self.path.income_models) + "/" + str(self.question_number) + str(self.question_part) + str(model_name), 'rb') as handle:
                model = pickle.load(handle)
        except:
            print("File not found error. Exiting.")
            exit()
        return model


    def read_data(self, datapath, datapath_test):
        self.data = pd.read_csv(datapath, delimiter="[ \s]*,[ \s]*", header=None, engine='python')
        self.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","moreless"]
        self.data.columns = self.columns
        self.features = self.columns[:len(self.columns)-1]
        
        mapping = {"Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3, "Local-gov": 4, "State-gov": 5, "Without-pay": 6, "Never-worked": 7}
        self.data = self.data.replace({'workclass': mapping})

        mapping = {"Bachelors": 0, "Some-college": 1, "11th": 2, "HS-grad": 3, "Prof-school": 4, "Assoc-acdm": 5, "Assoc-voc": 6, "9th": 7, "7th-8th": 8, "12th": 9, "Masters": 10, "1st-4th": 11, "10th": 12, "Doctorate": 13, "5th-6th": 14, "Preschool": 15}
        self.data = self.data.replace({'education': mapping})

        mapping = {"Married-civ-spouse": 0, "Divorced": 1, "Never-married": 2, "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5, "Married-AF-spouse": 6}
        self.data = self.data.replace({'marital-status': mapping})

        mapping = {"Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12, "Armed-Forces": 13}
        self.data = self.data.replace({'occupation': mapping})
        
        mapping = {"Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3, "Other-relative": 4, "Unmarried": 5}
        self.data = self.data.replace({'relationship': mapping})
        
        mapping = {"White": 0, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 2, "Other": 3, "Black": 4}
        self.data = self.data.replace({'race': mapping})
        
        mapping = {"Female": 0, "Male": 1}
        self.data = self.data.replace({'sex': mapping})
        
        mapping = {"United-States": 0, "Cambodia": 1, "England": 2, "Puerto-Rico": 3, "Canada": 4, "Germany": 5, "Outlying-US(Guam-USVI-etc)": 6, "India": 7, "Japan": 8, "Greece": 9, "South": 10, "China": 11, "Cuba": 12, "Iran": 13, "Honduras": 14, "Philippines": 15, "Italy": 16, "Poland": 17, "Jamaica": 18, "Vietnam": 19, "Mexico": 20, "Portugal": 21, "Ireland": 22, "France": 23, "Dominican-Republic": 24, "Laos": 25, "Ecuador": 26, "Taiwan": 27, "Haiti": 28, "Columbia": 29, "Hungary": 30, "Guatemala": 31, "Nicaragua": 32, "Scotland": 33, "Thailand": 34, "Yugoslavia": 35, "El-Salvador": 36, "Trinadad&Tobago": 37, "Peru": 38, "Hong": 39, "Holand-Netherlands": 40}
        self.data = self.data.replace({'native-country': mapping})
        
        mapping = {">50K": 0, "<=50K": 1}
        self.data = self.data.replace({'moreless': mapping})
        # self.data = self.data.sample(frac=1).reset_index(drop=True)
        
        # normalizing data
        # self.data = (self.data - self.data.mean())/self.data.std()
        
        # split data into multiple frames
        self.data_k_split = np.array_split(self.data, self.k)

        self.test_set = pd.read_csv(datapath_test, delimiter="[ \s]*,[ \s]*", header=None, engine='python')
        self.test_set.columns = self.columns
        self.features = self.columns[:len(self.columns)-1]
        
        mapping = {"Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3, "Local-gov": 4, "State-gov": 5, "Without-pay": 6, "Never-worked": 7}
        self.test_set = self.test_set.replace({'workclass': mapping})

        mapping = {"Bachelors": 0, "Some-college": 1, "11th": 2, "HS-grad": 3, "Prof-school": 4, "Assoc-acdm": 5, "Assoc-voc": 6, "9th": 7, "7th-8th": 8, "12th": 9, "Masters": 10, "1st-4th": 11, "10th": 12, "Doctorate": 13, "5th-6th": 14, "Preschool": 15}
        self.test_set = self.test_set.replace({'education': mapping})

        mapping = {"Married-civ-spouse": 0, "Divorced": 1, "Never-married": 2, "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5, "Married-AF-spouse": 6}
        self.test_set = self.test_set.replace({'marital-status': mapping})

        mapping = {"Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12, "Armed-Forces": 13}
        self.test_set = self.test_set.replace({'occupation': mapping})
        
        mapping = {"Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3, "Other-relative": 4, "Unmarried": 5}
        self.test_set = self.test_set.replace({'relationship': mapping})
        
        mapping = {"White": 0, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 2, "Other": 3, "Black": 4}
        self.test_set = self.test_set.replace({'race': mapping})
        
        mapping = {"Female": 0, "Male": 1}
        self.test_set = self.test_set.replace({'sex': mapping})
        
        mapping = {"United-States": 0, "Cambodia": 1, "England": 2, "Puerto-Rico": 3, "Canada": 4, "Germany": 5, "Outlying-US(Guam-USVI-etc)": 6, "India": 7, "Japan": 8, "Greece": 9, "South": 10, "China": 11, "Cuba": 12, "Iran": 13, "Honduras": 14, "Philippines": 15, "Italy": 16, "Poland": 17, "Jamaica": 18, "Vietnam": 19, "Mexico": 20, "Portugal": 21, "Ireland": 22, "France": 23, "Dominican-Republic": 24, "Laos": 25, "Ecuador": 26, "Taiwan": 27, "Haiti": 28, "Columbia": 29, "Hungary": 30, "Guatemala": 31, "Nicaragua": 32, "Scotland": 33, "Thailand": 34, "Yugoslavia": 35, "El-Salvador": 36, "Trinadad&Tobago": 37, "Peru": 38, "Hong": 39, "Holand-Netherlands": 40}
        self.test_set = self.test_set.replace({'native-country': mapping})
        
        mapping = {">50K.": 0, "<=50K.": 1}
        self.test_set = self.test_set.replace({'moreless': mapping})
        # self.test_set = self.test_set.sample(frac=1).reset_index(drop=True)

    def sigmoid(self, z):
        return (1/(1 + np.exp(-z)))
    
    def loss(self, h):
        return (-self.Y * np.log(h) - (1 - self.Y) * np.log(1 - h)).mean()

    def log_likelihood(self):
        z = np.dot(self.X, self.theta)
        ll = np.sum(self.Y * z - np.log(1 + np.exp(z)) )
        return ll
    
    def gradient_ascent(self, h):
        return np.dot(self.X.T, self.Y - h)
    
    def update_weight_mle(self, gradient):
        return self.theta + self.rate * gradient
    
    def get_errors(self):
        for theta in self.thetas:
            # z = np.dot(self.X, theta)
            # h = self.sigmoid(z)
            # final_loss = self.loss(h, self.Y)
            z = np.dot(self.X, theta)
            h = self.sigmoid(z)

            # accuracy
            pred_prob = self.sigmoid(z).round()
            final_loss = (self.Y == pred_prob).mean()
            self.validation_accuracy.append(final_loss)

            # loss
            self.validation_loss.append(self.loss(h))
    
    def logistic_regression_validate(self):
        self.X = self.validation_set.iloc[:,0:14]
        intercept = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((intercept, self.X), axis=1)
        self.X = np.array(self.X, dtype = 'float')
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.validation_set.iloc[:,14:15].values
        self.Y = np.array(self.Y, dtype = 'float')

        self.get_errors()

        #loss
        # z = np.dot(self.X, self.theta)
        # h = self.sigmoid(z)
        # final_loss = self.loss(h, self.Y)

        # accuracy
        z = np.dot(self.X, self.theta)
        pred_prob = self.sigmoid(z).round()
        final_loss = (self.Y == pred_prob).mean()
        self.final_validation_accuracy.append(final_loss)
        print("Validation accuracy for fold #" + str(self.validating_index) + ":", final_loss)
        
        if self.max_accuracy == -1:
            self.max_accuracy = final_loss
            self.max_accuracy_theta = self.theta.copy()
        elif self.max_accuracy < final_loss:
            self.max_accuracy = final_loss
            self.max_accuracy_theta = self.theta.copy()

    def logistic_regression_test(self):
        self.X = self.test_set.iloc[:,0:14]
        intercept = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((intercept, self.X), axis=1)
        self.X = np.array(self.X, dtype = 'float')
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.test_set.iloc[:,14:15].values
        self.Y = np.array(self.Y, dtype = 'float')
        
        z = np.dot(self.X, self.max_accuracy_theta)
        pred_prob = self.sigmoid(z).round()
        final_loss = (self.Y == pred_prob).mean()
        print("Test accuracy:", final_loss)
    
    def logistic_regression_train(self):
        self.X = self.training_set.iloc[:,0:14]
        intercept = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((intercept, self.X), axis=1)
        self.X = np.array(self.X, dtype = 'float')
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.training_set.iloc[:,14:15].values
        self.Y = np.array(self.Y, dtype = 'float')

        self.thetas = []
        
        # weights initialization
        self.theta = np.zeros((self.X.shape[1], 1))
        
        for i in range(self.iter):
            # h = self.sigmoid()
            # gradient = self.gradient_ascent(h)
            # self.theta = self.update_weight_mle(gradient)

            z = np.dot(self.X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(self.X.T, (h - self.Y)) / self.Y.size
            self.theta -= self.rate * gradient
            self.thetas.append(np.copy(self.theta))
            
            # accuracy
            z = np.dot(self.X, self.theta)
            pred_prob = self.sigmoid(z).round()
            final_loss = (self.Y == pred_prob).mean()
            self.train_accuracy.append(final_loss)
            # print(final_loss)

            # loss
            # z = np.dot(self.X, self.theta)
            h = self.sigmoid(z)
            self.train_loss.append(self.loss(h))
            # print(f'loss: {self.loss(h, self.Y)} \t')
        # final accuracy
        z = np.dot(self.X, self.theta)
        pred_prob = self.sigmoid(z).round()
        final_loss = (self.Y == pred_prob).mean()
        # final_loss = self.loss(h, self.Y)
        self.final_train_accuracy.append(final_loss)
        print("Training accuracy for fold #" + str(self.validating_index) + ":", final_loss)
    
    def logistic_regression(self):
        self.train_loss = []
        self.train_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.final_train_accuracy = []
        self.final_validation_accuracy = []
        skip = False
        self.max_accuracy = -1

        if self.skip_train:
            skip_ = input("previous cached models available. use them? (y for yes, default is no): ")
            if skip_ == 'y' or skip_ == 'Y':
                skip = True

        for i in range(self.k):
            self.thetas = []
            self.validating_index = i
            self.validation_set = pd.DataFrame(columns = self.columns)
            self.training_set = pd.DataFrame(columns = self.columns)

            for data_frame_index in range(self.k):
                if data_frame_index == self.validating_index:
                    self.validation_set = pd.concat([self.validation_set, self.data_k_split[data_frame_index]])
                else:
                    self.training_set = pd.concat([self.training_set, self.data_k_split[data_frame_index]])
            if skip:
                self.theta = self.pickle_load(self.validating_index)
                self.thetas = self.pickle_load(str(self.validating_index) + "_")
            else:
                self.logistic_regression_train()
            self.logistic_regression_validate()
            print()
        self.logistic_regression_test()
        if not skip:
            print("Average train accuracy: ", np.mean(np.array(self.final_train_accuracy)))
            self.pickle_save(self.final_train_accuracy, 'train_errors')
        print("Average validation accuracy: ", np.mean(np.array(self.final_validation_accuracy)))
        self.pickle_save(self.final_validation_accuracy, 'test_errors')

        train_errors = []
        validation_errors = []
        train_losses = []
        validation_losses = []
        for i in range(self.k):
            if not skip:
                train_losses.append(self.train_loss[i*self.iter:(i+1)*self.iter])
                train_errors.append(self.train_accuracy[i*self.iter:(i+1)*self.iter])
            validation_errors.append(np.array(self.validation_accuracy[i*self.iter:(i+1)*self.iter]))
            validation_losses.append(np.array(self.validation_loss[i*self.iter:(i+1)*self.iter]))

        if not skip:
            self.train_loss = np.mean(np.array(train_losses), axis=0)
            self.train_accuracy = np.mean(np.array(train_errors), axis=0)
        self.validation_accuracy = np.mean(np.array(validation_errors), axis=0)
        self.validation_loss = np.mean(np.array(validation_losses), axis=0)

        # plot error vs iterations
        if not skip:
            fig, ax = plt.subplots()
            ax.plot(np.arange(self.iter), self.train_accuracy, 'r', label = "Accuracy")
            ax.plot(np.arange(self.iter), self.train_loss, 'b', label = "Loss")
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Accuracy')
            ax.legend(loc='lower right')
            ax.set_title('Accuracy vs. Training Epoch')

        fig, ax = plt.subplots()
        ax.plot(np.arange(self.iter), self.validation_accuracy, 'r', label = "Accuracy")
        ax.plot(np.arange(self.iter), self.validation_loss, 'b', label = "Loss")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower right')
        ax.set_title('Accuracy vs. Validation Epoch')

        plt.show()

    def logistic_regression_train_ridge(self):
        self.X = self.training_set.iloc[:,0:14]
        intercept = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((intercept, self.X), axis=1)
        self.X = np.array(self.X, dtype = 'float')
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.training_set.iloc[:,14:15].values
        self.Y = np.array(self.Y, dtype = 'float')

        self.thetas = []
        
        # weights initialization
        self.theta = np.zeros((self.X.shape[1], 1))
        
        for i in range(self.iter):
            z = np.dot(self.X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(self.X.T, (h - self.Y)) / self.Y.size
            self.theta = self.theta - (self.rate * gradient) + (self.alpha_ridge/(2*len(self.X))) * self.theta
            self.thetas.append(np.copy(self.theta))
            
            # accuracy
            pred_prob = self.sigmoid(np.dot(self.X, self.theta)).round()
            final_loss = (self.Y == pred_prob).mean()
            self.train_accuracy.append(final_loss)

            # loss
            # z = np.dot(self.X, self.theta)
            h = self.sigmoid(z)
            self.train_loss.append(self.loss(h))
            # self.train_accuracy.append(self.loss(h, self.Y))
            # print(f'loss: {self.loss(h, self.Y)} \t')
        # final accuracy
        pred_prob = self.sigmoid(np.dot(self.X, self.theta)).round()
        final_loss = (self.Y == pred_prob).mean()
        # final_loss = self.loss(h, self.Y)
        self.final_train_accuracy.append(final_loss)
        print("Training accuracy for fold #" + str(self.validating_index) + ":", final_loss)
    
    def logistic_regression_ridge(self):
        self.train_loss = []
        self.train_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.final_train_accuracy = []
        self.final_validation_accuracy = []
        skip = False

        if self.skip_train:
            skip_ = input("previous cached models available. use them? (y for yes, default is no): ")
            if skip_ == 'y' or skip_ == 'Y':
                skip = True

        for i in range(self.k):
            self.thetas = []
            self.validating_index = i
            self.validation_set = pd.DataFrame(columns = self.columns)
            self.training_set = pd.DataFrame(columns = self.columns)

            for data_frame_index in range(self.k):
                if data_frame_index == self.validating_index:
                    self.validation_set = pd.concat([self.validation_set, self.data_k_split[data_frame_index]])
                else:
                    self.training_set = pd.concat([self.training_set, self.data_k_split[data_frame_index]])
            if skip:
                self.theta = self.pickle_load(self.validating_index)
                self.thetas = self.pickle_load(str(self.validating_index) + "_")
            else:
                self.logistic_regression_train_ridge()
            self.logistic_regression_validate()
            print()
        self.logistic_regression_test()
        if not skip:
            print("Average train accuracy: ", np.mean(np.array(self.final_train_accuracy)))
            self.pickle_save(self.final_train_accuracy, 'train_errors')
        print("Average validation accuracy: ", np.mean(np.array(self.final_validation_accuracy)))
        self.pickle_save(self.final_validation_accuracy, 'test_errors')

        train_errors = []
        validation_errors = []
        train_losses = []
        validation_losses = []
        for i in range(self.k):
            if not skip:
                train_losses.append(self.train_loss[i*self.iter:(i+1)*self.iter])
                train_errors.append(self.train_accuracy[i*self.iter:(i+1)*self.iter])
            validation_errors.append(np.array(self.validation_accuracy[i*self.iter:(i+1)*self.iter]))
            validation_losses.append(np.array(self.validation_loss[i*self.iter:(i+1)*self.iter]))

        if not skip:
            self.train_loss = np.mean(np.array(train_losses), axis=0)
            self.train_accuracy = np.mean(np.array(train_errors), axis=0)
        self.validation_accuracy = np.mean(np.array(validation_errors), axis=0)
        self.validation_loss = np.mean(np.array(validation_losses), axis=0)

        # plot error vs iterations
        if not skip:
            fig, ax = plt.subplots()
            ax.plot(np.arange(self.iter), self.train_accuracy, 'r', label = "Accuracy")
            ax.plot(np.arange(self.iter), self.train_loss, 'b', label = "Loss")
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Accuracy')
            ax.legend(loc='lower right')
            ax.set_title('Accuracy vs. Training Epoch')

        fig, ax = plt.subplots()
        ax.plot(np.arange(self.iter), self.validation_accuracy, 'r', label = "Accuracy")
        ax.plot(np.arange(self.iter), self.validation_loss, 'b', label = "Loss")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower right')
        ax.set_title('Accuracy vs. Validation Epoch')
        plt.show()

    def logistic_regression_train_lasso(self):
        self.X = self.training_set.iloc[:,0:14]
        intercept = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((intercept, self.X), axis=1)
        self.X = np.array(self.X, dtype = 'float')
        # normalizing data
        self.X = (self.X - self.X.mean())/self.X.std()
        self.Y = self.training_set.iloc[:,14:15].values
        self.Y = np.array(self.Y, dtype = 'float')

        self.thetas = []
        
        # weights initialization
        self.theta = np.zeros((self.X.shape[1], 1))
        
        for i in range(self.iter):
            z = np.dot(self.X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(self.X.T, (h - self.Y)) / self.Y.size
            self.theta = self.theta - (self.rate * gradient) + (self.alpha_ridge/(2*len(self.X))) * np.sign(self.theta)
            self.thetas.append(np.copy(self.theta))
            
            # accuracy
            pred_prob = self.sigmoid(np.dot(self.X, self.theta)).round()
            final_loss = (self.Y == pred_prob).mean()
            self.train_accuracy.append(final_loss)

            # loss
            # z = np.dot(self.X, self.theta)
            h = self.sigmoid(z)
            self.train_loss.append(self.loss(h))
            # self.train_accuracy.append(self.loss(h, self.Y))
            # print(f'loss: {self.loss(h, self.Y)} \t')
        # final accuracy
        pred_prob = self.sigmoid(np.dot(self.X, self.theta)).round()
        final_loss = (self.Y == pred_prob).mean()
        # final_loss = self.loss(h, self.Y)
        self.final_train_accuracy.append(final_loss)
        print("Training accuracy for fold #" + str(self.validating_index) + ":", final_loss)
    
    def logistic_regression_lasso(self):
        self.train_loss = []
        self.train_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.final_train_accuracy = []
        self.final_validation_accuracy = []
        skip = False

        if self.skip_train:
            skip_ = input("previous cached models available. use them? (y for yes, default is no): ")
            if skip_ == 'y' or skip_ == 'Y':
                skip = True

        for i in range(self.k):
            self.thetas = []
            self.validating_index = i
            self.validation_set = pd.DataFrame(columns = self.columns)
            self.training_set = pd.DataFrame(columns = self.columns)

            for data_frame_index in range(self.k):
                if data_frame_index == self.validating_index:
                    self.validation_set = pd.concat([self.validation_set, self.data_k_split[data_frame_index]])
                else:
                    self.training_set = pd.concat([self.training_set, self.data_k_split[data_frame_index]])
            if skip:
                self.theta = self.pickle_load(self.validating_index)
                self.thetas = self.pickle_load(str(self.validating_index) + "_")
            else:
                self.logistic_regression_train_lasso()
            self.logistic_regression_validate()
            print()
        self.logistic_regression_test()
        if not skip:
            print("Average train accuracy: ", np.mean(np.array(self.final_train_accuracy)))
            self.pickle_save(self.final_train_accuracy, 'train_errors')
        print("Average validation accuracy: ", np.mean(np.array(self.final_validation_accuracy)))
        self.pickle_save(self.final_validation_accuracy, 'test_errors')

        train_errors = []
        validation_errors = []
        train_losses = []
        validation_losses = []
        for i in range(self.k):
            if not skip:
                train_losses.append(self.train_loss[i*self.iter:(i+1)*self.iter])
                train_errors.append(self.train_accuracy[i*self.iter:(i+1)*self.iter])
            validation_errors.append(np.array(self.validation_accuracy[i*self.iter:(i+1)*self.iter]))
            validation_losses.append(np.array(self.validation_loss[i*self.iter:(i+1)*self.iter]))

        if not skip:
            self.train_loss = np.mean(np.array(train_losses), axis=0)
            self.train_accuracy = np.mean(np.array(train_errors), axis=0)
        self.validation_accuracy = np.mean(np.array(validation_errors), axis=0)
        self.validation_loss = np.mean(np.array(validation_losses), axis=0)

        # plot error vs iterations
        if not skip:
            fig, ax = plt.subplots()
            ax.plot(np.arange(self.iter), self.train_accuracy, 'r', label = "Accuracy")
            ax.plot(np.arange(self.iter), self.train_loss, 'b', label = "Loss")
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Accuracy')
            ax.legend(loc='lower right')
            ax.set_title('Accuracy vs. Training Epoch')

        fig, ax = plt.subplots()
        ax.plot(np.arange(self.iter), self.validation_accuracy, 'r', label = "Accuracy")
        ax.plot(np.arange(self.iter), self.validation_loss, 'b', label = "Loss")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower right')
        ax.set_title('Accuracy vs. Validation Epoch')
        plt.show()

    def __init__(self):
        self.path = paths()

        self.skip_train = False
        self.k = 5
        self.iter = 1000
        self.rate = 0.1

        datapath_train = self.path.Logistic_Regression + "/data_1/train.csv"
        datapath_test = self.path.Logistic_Regression + "/data_1/test.csv"
        self.read_data(datapath_train, datapath_test)

        self.question_number = '2'
        self.question_part = 'aa'
        # self.logistic_regression()

        self.question_part = 'ab'
        print()
        self.max_accuracy = -1
        input("Press enter to move to the next part")
        print("Ridge Logistic Regression")
        self.alpha_ridge = 1.2 #random
        print("Using a alpha_ridge value of", self.alpha_ridge)
        self.logistic_regression_ridge()

        self.question_part = 'ac'
        print()
        self.max_accuracy = -1
        input("Press enter to move to the next part")
        print("Lasso Logistic Regression")
        self.alpha_lasso = 1.2 #random
        print("Using a alpha_lasso value of", self.alpha_lasso)
        self.logistic_regression_lasso()


if __name__ == "__main__":
    main = main()