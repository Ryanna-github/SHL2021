import pandas as pd 
import numpy as np

from utils.DataLoader import SHLDataLoader
import utils.helper as helper

class DataProcesser():
    def __init__(self, dataloader):
        super(DataProcesser).__init__()
        data.dfdata = dataloader
        if not isinstance(dataloader, DataLoader.SHLDataLoader):
            raise TypeError("A Customized DataLoader Should be Initialized")

    def process_loc(self):
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
        print("------------------------ Features in data.wifi(_detail) Extracted ------------------------")

    def process_cells(self):
        data.cells_detail['time'] = data.cells_detail['time'].astype("int").round(-3)
        data.cells_detail['isRegistered'] = data.cells_detail['isRegistered'].apply(pd.to_numeric)
        data.cells_detail['asuLevel'] = data.cells_detail['asuLevel'].apply(pd.to_numeric)
        data.cells_detail['dbm'] = data.cells_detail['dbm'].apply(pd.to_numeric)
        data.cells_detail['level'] = data.cells_detail['level'].apply(pd.to_numeric)

        tmp_cells_mode = data.cells_detail[['time', 'ctype']].groupby(['time'], as_index = False).agg(lambda x: Counter(x).most_common()[0][0]).add_prefix("cells_").add_suffix("_mode")
        tmp_cells_mean = data.cells_detail[['time', 'isRegistered', 'asuLevel', 'dbm', 'level']].groupby(['time'], as_index = False).mean().add_prefix("cells_").add_suffix("_mean")
        tmp_cells_min = data.cells_detail[['time', 'asuLevel', 'dbm', 'level']].groupby(['time'], as_index = False).min().add_prefix("cells_").add_suffix("_min")
        tmp_cells_max = data.cells_detail[['time', 'asuLevel', 'dbm', 'level']].groupby(['time'], as_index = False).max().add_prefix("cells_").add_suffix("_max")
        tmp_cells_std = data.cells_detail[['time', 'asuLevel', 'dbm']].groupby(['time'], as_index = False).std().add_prefix("cells_").add_suffix("_std")

        tmp_cells = pd.merge(tmp_cells_mode.rename({"cells_time_mode": "time"}, axis = 1), tmp_cells_mean.rename({"cells_time_mean": "time"}, axis = 1), on = ['time'])
        tmp_cells = pd.merge(tmp_cells, tmp_cells_min.rename({"cells_time_min": "time"}, axis = 1), on = ['time'])
        tmp_cells = pd.merge(tmp_cells, tmp_cells_max.rename({"cells_time_max": "time"}, axis = 1), on = ['time'])
        tmp_cells = pd.merge(tmp_cells, tmp_cells_std.rename({"cells_time_std": "time"}, axis = 1), on = ['time'])

        data.df = pd.merge(data.df, tmp_cells, on = ['time'], how = 'left')
        print("------------------------ Features in data.cells(_detail) Extracted ------------------------")

    def process_gps(self):
        data.gps_detail['time'] = data.gps_detail['time'].astype("int").round(-3)
        data.gps_detail['gps_snr'] = data.gps_detail['snr'].apply(pd.to_numeric)

        tmp_gps_mean = data.gps_detail[['time', 'gps_snr']].groupby(['time'], as_index = False).mean().add_suffix("_mean")
        tmp_gps_min = data.gps_detail[['time', 'gps_snr']].groupby(['time'], as_index = False).min().add_suffix("_min")
        tmp_gps_max = data.gps_detail[['time', 'gps_snr']].groupby(['time'], as_index = False).max().add_suffix("_max")
        tmp_gps_std = data.gps_detail[['time', 'gps_snr']].groupby(['time'], as_index = False).std().add_suffix("_std")

        tmp_gps = pd.merge(tmp_gps_mean.rename({"time_mean": "time"}, axis = 1), tmp_gps_min.rename({"time_min": "time"}, axis = 1), on = ['time'])
        tmp_gps = pd.merge(tmp_gps, tmp_gps_max.rename({"time_max": "time"}, axis = 1), on = ['time'])
        tmp_gps = pd.merge(tmp_gps, tmp_gps_std.rename({"time_std": "time"}, axis = 1), on = ['time'])

        data.df = pd.merge(data.df, tmp_gps, on = ['time'], how = 'left')
        print("------------------------ Features in data.gps(_detail) Extracted ------------------------")

    def process_pipe(self):
        self.process_loc()
        self.process_wifi()
        self.process_cells()
        self.process_gps()

    def one_hot_transform(self, col_name_list):
        self.data.df_hot = self.data.df.copy()
        # discrete v
        dis_cols = ['cells_ctype_mode']
        for col in dis_cols:
            self.data.df_hot[col].fillna("miss", inplace = True)
            df_hot = helper.get_one_hot(self.data.df_hot, col)
            self.data.df_hot = pd.concat([self.data.df_hot, df_hot])
            self.data.df_hot.drop(col, axis = 1, inplace = True)
            print("------------------------ Finish Encode {} (One Hot) ------------------------".format(col))