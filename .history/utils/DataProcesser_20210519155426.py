import pandas as pd 
import numpy as np

from utils.DataLoader import DataLoader

class DataProcesser():
    def __init__(self, dataloader):
        super(DataProcesser).__init__()
        self.dataloader = dataloader
        if not isinstance(self.dataloader, DataLoader.DataLoader):
            raise TypeError("A Customized DataLoader Should be Initialized")

    def loc_process(self):
        