import pandas as pd 
import numpy as np
from collections import Counter

from utils.DataLoader import SHLDataLoader
import utils.helper as helper

class DataProcesser():
    def __init__(self, dataloader):
        # super(self.DataProcesser).__init__()
        self.data = dataloader
        if isinstance(self.data, SHLDataLoader):
            print("Customized DataLoader Passed in.")
        else:
            print("Wrong data type {}. Abort!".format(type(self.data)))

    def process_pre(self):
        # unit: 1s
        self.data.loc['time'] = self.data.loc.apply(lambda x: x['time'].astype('int').round(-3), axis = 1)
        self.data.gps['time'] = self.data.gps['time'].astype('int').round(-3)
        self.data.wifi['time'] = self.data.wifi['time'].astype('int').round(-3)
        self.data.cells['time'] = self.data.cells['time'].astype('int').round(-3)
        # data type
        self.data.gps['number'] = self.data.gps['number'].apply(pd.to_numeric)
        self.data.wifi['number'] = self.data.wifi['number'].apply(pd.to_numeric)
        self.data.cells['number'] = self.data.cells['number'].apply(pd.to_numeric)

        self.data.loc = self.data.loc.groupby(['time'], as_index = False).mean()[self.data.loc.columns.to_list()]
        self.data.gps = self.data.gps.groupby(['time'], as_index = False).mean()[self.data.gps.columns.to_list()]
        self.data.wifi = self.data.wifi.groupby(['time'], as_index = False).mean()[self.data.wifi.columns.to_list()]
        self.data.cells = self.data.cells.groupby(['time'], as_index = False).mean()[self.data.cells.columns.to_list()]

        self.data.df = pd.merge(self.data.label, self.data.loc, on = ['time'], how = 'left')
        self.data.df = pd.merge(self.data.df, self.data.gps.rename({"number": "num_gps"}, axis = 1), on = ['time'], how = 'left')
        self.data.df = pd.merge(self.data.df, self.data.wifi.rename({"number": "num_wifi"}, axis = 1), on = ['time'], how = 'left')
        self.data.df = pd.merge(self.data.df, self.data.cells.rename({"number": "num_cells"}, axis = 1), on = ['time'], how = 'left')
        print("------------------------ Basic Features Extracted (data.df) ------------------------")
        print("Feature Initialized: {}".format(list(self.data.df)))


    def process_loc(self):
        old_features = set(list(self.data.df))
        # prepare
        self.data.df['time_dlt'] = self.data.df['time'].diff().fillna(method = 'bfill')
        self.data.df['valid_dlt'] = self.data.df.apply(lambda x: int(x['time_dlt'] <= 10000), axis = 1)
        # utm loc
        self.data.df['east'] = self.data.df.apply(lambda x: helper.gps2utm_east(x), axis = 1)
        self.data.df['north'] = self.data.df.apply(lambda x: helper.gps2utm_north(x), axis = 1)
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
        self.data.df['speed_dlt'] = self.data.df['speed'].diff(1)
        self.data.df['acc'] = self.data.df.apply(lambda x: x['speed_dlt']/x['time_dlt'] if x['valid_dlt'] == 1 else np.nan, axis = 1)
        new_features = set(list(self.data.df)).difference(old_features)
        print("------------------------ Features in self.data.loc Extracted ------------------------")
        print("New Feature Added: {}".format(new_features))

    def process_wifi(self):
        old_features = set(list(self.data.df))
        self.data.wifi_detail['time'] = self.data.wifi_detail['time'].astype("int").round(-3)
        self.data.wifi_detail['wifi_rssi'] = self.data.wifi_detail['rssi'].apply(pd.to_numeric)
        self.data.wifi_detail['wifi_freq'] = self.data.wifi_detail['freq'].apply(pd.to_numeric)
        self.data.wifi_detail['wifi_freq'] = self.data.wifi_detail['wifi_freq'].apply(lambda x: 1 if x > 3000 else 0) # 1: 5Hz, 0:2.4Hz

        tmp_wifi_mode = self.data.wifi_detail[['time', 'wifi_rssi']].groupby(['time'], as_index = False).agg(lambda x: Counter(x).most_common()[0][0]).add_suffix("_mode")
        tmp_wifi_mean = self.data.wifi_detail[['time', 'wifi_rssi']].groupby(['time'], as_index = False).mean().add_suffix("_mean")
        tmp_wifi_min = self.data.wifi_detail[['time', 'wifi_rssi']].groupby(['time'], as_index = False).min().add_suffix("_min")
        tmp_wifi_max = self.data.wifi_detail[['time', 'wifi_rssi']].groupby(['time'], as_index = False).max().add_suffix("_max")
        tmp_wifi_std = self.data.wifi_detail[['time', 'wifi_rssi']].groupby(['time'], as_index = False).std().add_suffix("_std")

        tmp_wifi = pd.merge(tmp_wifi_mode.rename({"time_mode": "time"}, axis = 1), tmp_wifi_mean.rename({"time_mean": "time"}, axis = 1), on = ['time'])
        tmp_wifi = pd.merge(tmp_wifi, tmp_wifi_min.rename({"time_min": "time"}, axis = 1), on = ['time'])
        tmp_wifi = pd.merge(tmp_wifi, tmp_wifi_max.rename({"time_max": "time"}, axis = 1), on = ['time'])
        tmp_wifi = pd.merge(tmp_wifi, tmp_wifi_std.rename({"time_std": "time"}, axis = 1), on = ['time'])
        # v2 supplement
        tmp_wifi['wifi_freq_5ratio'] = self.data.wifi_detail[['time', 'wifi_freq']].groupby(['time'], as_index = False).agg(lambda x: x.sum()/len(x))['wifi_freq']

        self.data.df = pd.merge(self.data.df, tmp_wifi, on = ['time'], how = 'left')
        new_features = set(list(self.data.df)).difference(old_features)
        print("------------------------ Features in self.data.wifi(_detail) Extracted ------------------------")
        print("New Feature Added: {}".format(new_features))

    def process_cells(self):
        old_features = set(list(self.data.df))
        self.data.cells_detail['time'] = self.data.cells_detail['time'].astype("int").round(-3)
        self.data.cells_detail['isRegistered'] = self.data.cells_detail['isRegistered'].apply(pd.to_numeric)
        self.data.cells_detail['asuLevel'] = self.data.cells_detail['asuLevel'].apply(pd.to_numeric)
        self.data.cells_detail['dbm'] = self.data.cells_detail['dbm'].apply(pd.to_numeric)
        self.data.cells_detail['level'] = self.data.cells_detail['level'].apply(pd.to_numeric)

        tmp_cells_mode = self.data.cells_detail[['time', 'ctype']].groupby(['time'], as_index = False).agg(lambda x: Counter(x).most_common()[0][0]).add_prefix("cells_").add_suffix("_mode")
        tmp_cells_mean = self.data.cells_detail[['time', 'isRegistered', 'asuLevel', 'dbm', 'level']].groupby(['time'], as_index = False).mean().add_prefix("cells_").add_suffix("_mean")
        tmp_cells_min = self.data.cells_detail[['time', 'asuLevel', 'dbm', 'level']].groupby(['time'], as_index = False).min().add_prefix("cells_").add_suffix("_min")
        tmp_cells_max = self.data.cells_detail[['time', 'asuLevel', 'dbm', 'level']].groupby(['time'], as_index = False).max().add_prefix("cells_").add_suffix("_max")
        tmp_cells_std = self.data.cells_detail[['time', 'asuLevel', 'dbm']].groupby(['time'], as_index = False).std().add_prefix("cells_").add_suffix("_std")

        tmp_cells = pd.merge(tmp_cells_mode.rename({"cells_time_mode": "time"}, axis = 1), tmp_cells_mean.rename({"cells_time_mean": "time"}, axis = 1), on = ['time'])
        tmp_cells = pd.merge(tmp_cells, tmp_cells_min.rename({"cells_time_min": "time"}, axis = 1), on = ['time'])
        tmp_cells = pd.merge(tmp_cells, tmp_cells_max.rename({"cells_time_max": "time"}, axis = 1), on = ['time'])
        tmp_cells = pd.merge(tmp_cells, tmp_cells_std.rename({"cells_time_std": "time"}, axis = 1), on = ['time'])

        self.data.df = pd.merge(self.data.df, tmp_cells, on = ['time'], how = 'left')
        new_features = set(list(self.data.df)).difference(old_features)
        print("------------------------ Features in self.data.cells(_detail) Extracted ------------------------")
        print("New Feature Added: {}".format(new_features))

    def process_gps(self):
        old_features = set(list(self.data.df))
        self.data.gps_detail['time'] = self.data.gps_detail['time'].astype("int").round(-3)
        self.data.gps_detail['gps_snr'] = self.data.gps_detail['snr'].apply(pd.to_numeric)

        tmp_gps_mean = self.data.gps_detail[['time', 'gps_snr']].groupby(['time'], as_index = False).mean().add_suffix("_mean")
        tmp_gps_min = self.data.gps_detail[['time', 'gps_snr']].groupby(['time'], as_index = False).min().add_suffix("_min")
        tmp_gps_max = self.data.gps_detail[['time', 'gps_snr']].groupby(['time'], as_index = False).max().add_suffix("_max")
        tmp_gps_std = self.data.gps_detail[['time', 'gps_snr']].groupby(['time'], as_index = False).std().add_suffix("_std")

        tmp_gps = pd.merge(tmp_gps_mean.rename({"time_mean": "time"}, axis = 1), tmp_gps_min.rename({"time_min": "time"}, axis = 1), on = ['time'])
        tmp_gps = pd.merge(tmp_gps, tmp_gps_max.rename({"time_max": "time"}, axis = 1), on = ['time'])
        tmp_gps = pd.merge(tmp_gps, tmp_gps_std.rename({"time_std": "time"}, axis = 1), on = ['time'])

        self.data.df = pd.merge(self.data.df, tmp_gps, on = ['time'], how = 'left')
        new_features = set(list(self.data.df)).difference(old_features)
        print("------------------------ Features in self.data.gps(_detail) Extracted ------------------------")
        print("New Feature Added: {}".format(new_features))

    def process_pipe(self):
        try:
            self.data.df
        except:
            self.process_pre()
        self.process_loc()
        self.process_wifi()
        self.process_cells()
        self.process_gps()

    def one_hot_transform(self, col_name_list):
        self.self.data.df_hot = self.self.data.df.copy()
        # discrete v
        dis_cols = ['cells_ctype_mode']
        for col in dis_cols:
            self.self.data.df_hot[col].fillna("miss", inplace = True)
            df_hot = helper.get_one_hot(self.self.data.df_hot, col)
            self.self.data.df_hot = pd.concat([self.self.data.df_hot, df_hot]) # ========== 不对，修改！
            self.self.data.df_hot.drop(col, axis = 1, inplace = True)
            print("------------------------ Finish Encode {} (One Hot) ------------------------".format(col))