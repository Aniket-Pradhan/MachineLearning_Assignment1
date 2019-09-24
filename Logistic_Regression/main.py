import numpy as np
import pandas as pd

from paths import paths

class main:

    def read_data(self, datapath):
        self.data = pd.read_csv(datapath, delim_whitespace=True, header=None)
        self.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","moreless"]
        self.data.columns = self.columns
        self.features = self.columns[:len(self.columns)-1]
        
        mapping = {"Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3, "Local-gov": 4, "State-gov": 5, "Without-pay": 6, "Never-worked": 7}
        self.data = self.data.replace({'workclass': mapping})

        mapping = {"Bachelors": 0, "Some-college": 1, "11th": 2, "HS-grad": 3, "Prof-school": 4, "Assoc-acdm": 5, "Assoc-voc": 6, "9th": 7, "7th-8th": 8, "12th": 9, "Masters": 10, "1st-4th": 11, "10th": 12, "Doctorate": 13, "5th-6th": 14, "Preschool": 15}
        self.data = self.data.replace({'education': mapping})

        mapping = {"Married-civ-spouse": 0, "Divorced": 1, "Never-married": 2, "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5, "Married-AF-spouse": 6}
        self.data = self.data.replace({'marital-status': mapping})

        mapping = "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12, "Armed-Forces": 13}
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
        
        # normalizing data
        # self.data = (self.data - self.data.mean())/self.data.std()
        
        # split data into multiple frames
        self.data_k_split = np.array_split(self.data, self.k)

    def __init__(self):
        self.path = paths()
        datapath_train = self.path.Logistic_Regression + "/data_1/train.csv"
        self.read_data(datapath_train)
        pass

if __name__ == "__main__":
    main = main()