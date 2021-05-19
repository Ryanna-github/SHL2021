import pandas as pd 
import numpy as np

from utils.DataLoader import DataLoader, helper

class DataProcesser():
    def __init__(self, dataloader):
        super(DataProcesser).__init__()
        self.data = dataloader
        if not isinstance(self.dataloader, DataLoader.DataLoader):
            raise TypeError("A Customized DataLoader Should be Initialized")

    def process_loc(self):
        # prepare
        self.data.df['time_dlt'] = self.data.df['time'].diff().fillna(method = 'bfill')
        self.self.data.df['valid_dlt'] = self.data.df.apply(lambda x: int(x['time_dlt'] <= 10000), axis = 1)
        # utm loc
        self.data.df['east'] = self.data.df.apply(lambda x: gps2utm_east(x), axis = 1)
        self.data.df['north'] = self.data.df.apply(lambda x: gps2utm_north(x), axis = 1)
        self.data.df['east_dlt'] = self.data.df['east'].diff(1)
        self.data.df['north_dlt'] = self.data.df['north'].diff(1)
        # speed
        self.data.df['east_speed'] = self.data.df.apply(lambda x: x['east_dlt']/x['time_dlt']*1000 if x['valid_dlt'] == 1 else np.nan, axis = 1)
        self.data.df['north_speed'] = self.data.df.apply(lambda x: x['north_dlt']/x['time_dlt']*1000 if x['valid_dlt'] == 1 else np.nan, axis = 1)
        self.data.df['east_speed'] = self.data.df['east_speed'].apply(lambda x: x if np.abs(x) < 300 else np.nan)
        self.data.df['north_speed'] = self.data.df['north_speed'].apply(lambda x: x if np.abs(x) < 300 else np.nan)
        self.data.df['speed'] = self.data.df.apply(lambda x: np.sqrt(x['east_speed']**2 + x['north_speed']**2), axis = 1)
        self.data.df['speed_dif'] = self.data.df.apply(lambda x: np.abs(x['east_speed'] - x['north_speed']), axis = 1)
        # acc 
        self.data.df['acc'] = self.data.df.apply(lambda x: x['speed']/x['time_dlt'] if x['valid_dlt'] == 1 else np.nan, axis = 1)