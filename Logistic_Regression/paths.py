from pathlib import Path

class paths:

    def init_local_paths(self):
        self.root = str(Path(__file__).parent.parent)
        self.datasets = self.root + "/datasets"
        self.models = self.root + "/models"
        self.income_models = self.models + "/income"
        self.Linear_Regression = self.root + "/Linear_Regression"
        self.Logistic_Regression = self.root + "/Logistic_Regression"

    def init_urls(self):
        pass

    def __init__(self):
        self.init_local_paths()
        self.init_urls()
