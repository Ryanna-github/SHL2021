import pandas as pd 
import numpy as np

from utils.DataLoader import DataLoader, helper

class DataProcesser():
    def __init__(self, dataloader):
        super(DataProcesser).__init__()
        self.dataloader = dataloader
        if not isinstance(self.dataloader, DataLoader.DataLoader):
            raise TypeError("A Customized DataLoader Should be Initialized")

    def loc_process(self):
        # prepare
        data.df['time_dlt'] = data.df['time'].diff().fillna(method = 'bfill')
        data.df['valid_dlt'] = data.df.apply(lambda x: int(x['time_dlt'] <= 10000), axis = 1)
        # utm loc
        data.df['east'] = data.df.apply(lambda x: gps2utm_east(x), axis = 1)
        data.df['north'] = data.df.apply(lambda x: gps2utm_north(x), axis = 1)
        data.df['east_dlt'] = data.df['east'].diff(1)
        data.df['north_dlt'] = data.df['north'].diff(1)
        # speed
        data.df['east_speed'] = data.df.apply(lambda x: x['east_dlt']/x['time_dlt']*1000 if x['valid_dlt'] == 1 else np.nan, axis = 1)
        data.df['north_speed'] = data.df.apply(lambda x: x['north_dlt']/x['time_dlt']*1000 if x['valid_dlt'] == 1 else np.nan, axis = 1)
        data.df['east_speed'] = data.df['east_speed'].apply(lambda x: x if np.abs(x) < 300 else np.nan)
        data.df['north_speed'] = data.df['north_speed'].apply(lambda x: x if np.abs(x) < 300 else np.nan)
        data.df['speed'] = data.df.apply(lambda x: np.sqrt(x['east_speed']**2 + x['north_speed']**2), axis = 1)
        data.df['speed_dif'] = data.df.apply(lambda x: np.abs(x['east_speed'] - x['north_speed']), axis = 1)
        # acc 
        data.df['acc'] = data.df.apply(lambda x: x['speed']/x['time_dlt'] if x['valid_dlt'] == 1 else np.nan, axis = 1)