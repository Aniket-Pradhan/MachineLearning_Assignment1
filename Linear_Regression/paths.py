from pathlib import Path

class paths:

    def init_local_paths(self):
        self.root = str(Path(__file__).parent.parent)
        self.datasets = self.root + "/datasets"
        self.abalone_dataset = self.datasets + "/abalone"
        self.models = self.root + "/models"
        self.abalone_models = self.models + "/abalone"
        self.Linear_Regression = self.root + "/Linear_Regression"

    def init_urls(self):
        self.abalone_dataset_url = "ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/abalone.tar.gz"

    def __init__(self):
        self.init_local_paths()
        self.init_urls()
