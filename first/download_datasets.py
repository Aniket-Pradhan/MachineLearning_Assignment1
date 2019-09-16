import os
from os.path import isdir
import gzip
import shutil
import tarfile
import urllib.request

from paths import paths
import download_progressbar

class download_datasets:

    def download_abalone_dataset(self):
        dataset_root = self.path.datasets
        dataset_path = self.path.datasets + "/abalone.tar.gz"
        with download_progressbar.TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="Downloading Abalone dataset") as t:
            with urllib.request.urlopen(self.path.abalone_dataset_url) as response, open(dataset_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
        tf = tarfile.open(dataset_path)
        tf.extractall(path=dataset_root)

        with gzip.open(dataset_root + "/abalone/Dataset.data.gz", 'rb') as infile:
            with open(dataset_root + "/abalone/Dataset.data", 'wb') as outfile:
                shutil.copyfileobj(infile, outfile)
    
    def check_create_directory(self):
        dataset_root = self.path.datasets
        if not os.path.isdir(dataset_root):
            os.makedirs(dataset_root)
    
    def check_directory(self):
        dataset_root = self.path.datasets
        ls_root = os.listdir(dataset_root)
        if "abalone" in ls_root and isdir(dataset_root + "/abalone"):
            dataset_path = dataset_root + "/abalone"
            ls_datapath = os.listdir(dataset_path)
            if "Dataset.data" in ls_datapath:
                return False
        return True

    def __init__(self):
        self.path = paths()
        self.check_create_directory()
        do_download = self.check_directory()
        if do_download:
            self.download_abalone_dataset()
        else:
            do_download = input("abalone data is already present. re-download? (Enter 'y' for yes, default is no.): ")
            if do_download == 'y' or do_download == 'Y':
                self.download_abalone_dataset()
