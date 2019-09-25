from pathlib import Path

class paths:

    def init_local_paths(self):
        self.root = str(Path(__file__).parent.parent)
        self.datasets = self.root + "/datasets"
        self.mnist = self.datasets + "/mnist"
        self.models = self.root + "/models"
        self.mnist_models = self.models + "/mnist"
        self.income_models = self.models + "/income"
        self.Linear_Regression = self.root + "/Linear_Regression"
        self.Logistic_Regression = self.root + "/Logistic_Regression"

    def init_urls(self):
        self.train_images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        self.train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
        self.test_images_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
        self.test_labels_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        pass

    def __init__(self):
        self.init_local_paths()
        self.init_urls()
