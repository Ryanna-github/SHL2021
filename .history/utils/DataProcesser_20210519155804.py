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
        print("------------------------ Features in data.loc Extracted ------------------------")

    def process_wifi(self):
        data.wifi_detail['time'] = data.wifi_detail['time'].astype("int").round(-3)
        data.wifi_detail['wifi_rssi'] = data.wifi_detail['rssi'].apply(pd.to_numeric)
        data.wifi_detail['wifi_freq'] = data.wifi_detail['freq'].apply(pd.to_numeric)
        data.wifi_detail['wifi_freq'] = data.wifi_detail['wifi_freq'].apply(lambda x: 5 if x > 3000 else 2.4)

        tmp_wifi_mode = data.wifi_detail[['time', 'wifi_rssi']].groupby(['time'], as_index = False).agg(lambda x: Counter(x).most_common()[0][0]).add_suffix("_mode")
        tmp_wifi_mean = data.wifi_detail[['time', 'wifi_rssi']].groupby(['time'], as_index = False).mean().add_suffix("_mean")
        tmp_wifi_min = data.wifi_detail[['time', 'wifi_rssi']].groupby(['time'], as_index = False).min().add_suffix("_min")
        tmp_wifi_max = data.wifi_detail[['time', 'wifi_rssi']].groupby(['time'], as_index = False).max().add_suffix("_max")
        tmp_wifi_std = data.wifi_detail[['time', 'wifi_rssi']].groupby(['time'], as_index = False).std().add_suffix("_std")

        tmp_wifi = pd.merge(tmp_wifi_mode.rename({"time_mode": "time"}, axis = 1), tmp_wifi_mean.rename({"time_mean": "time"}, axis = 1), on = ['time'])
        tmp_wifi = pd.merge(tmp_wifi, tmp_wifi_min.rename({"time_min": "time"}, axis = 1), on = ['time'])
        tmp_wifi = pd.merge(tmp_wifi, tmp_wifi_max.rename({"time_max": "time"}, axis = 1), on = ['time'])
        tmp_wifi = pd.merge(tmp_wifi, tmp_wifi_std.rename({"time_std": "time"}, axis = 1), on = ['time'])

        data.df = pd.merge(data.df, tmp_wifi, on = ['time'], how = 'left')