import pandas as pd
import numpy as np
import re
import time
from utils.TimeKeeper import TimeKeeper


class SHLDataLoader():
    def __init__(self, root_path, ratio = None):
        self.load_ratio = ratio
        self.root_path = root_path

    def wifi_detail_transformer(self, wifi_list, this_time):
        """
        Description: Extract detail info of Wifi data
        return: pd.DataFrame
            - time, bssid, ssid, rssi, freq, cap
        """
        res = np.array(wifi_list).reshape(-1, 5)
        res = pd.DataFrame(res, columns = ('bssid', 'ssid', 'rssi', 'freq', 'cap'))
        res.insert(loc = 0, column = 'time', value = [this_time] * res.shape[0])
        return res

    def gps_detail_transformer(self, gps_list, this_time):
        """
        Description: Extract detail info of GPS data
        return: pd.DataFrame
            - time, id, snr, azimuth, elevation
        """
        res = np.array(gps_list).reshape(-1, 4)
        res = pd.DataFrame(res, columns = ('id', 'snr', 'azimuth', 'elevation'))
        res.insert(loc = 0, column = 'time', value = [this_time] * res.shape[0])
        return res

    def cells_detail_transformer(self, cells_list, this_time):
        """
        Description: Extract detail info of Cells data
        return: pd.DataFrame
            - "time", "ctype", "isRegistered", "cid", "lac", "mmc", "mnc", "asuLevel", "dbm", "level"
        """
        # get cell type
        cells_type = re.findall(" LTE | GSM | WCDMA ", cells_list)
        # get detail info
        cells_info = re.split(" LTE | GSM | WCDMA ", cells_list)[1:]
        cells_detail = []
        for i, ctype in enumerate(cells_type):
            if re.match('.*LTE.*', ctype):
                info = ['LTE'] + np.array(cells_info[i].split(" "))[[0, 1, 5, 2, 3, 6, 7, 8]].tolist()
            elif re.match('.*GSM.*', ctype):
                info = ['GSM'] + np.array(cells_info[i].split(" ")).tolist()
            elif re.match(".*WCDMA.*", ctype):
                info = ['WCDMA'] + np.array(cells_info[i].split(" "))[[0, 1, 2, 3, 4, 6, 7, 8]].tolist()
            else:
                raise ValueError("Unrecognized cell type {}".format(ctype))
            cells_detail.append(info)

        res = pd.DataFrame(cells_detail, columns = ("ctype", "isRegistered", "cid", "lac", "mmc", "mnc", "asuLevel", "dbm", "level"))
        res.insert(loc = 0, column = 'time', value = [this_time] * len(cells_type))
        return res

    def load_label(self):
        print("Label Loading...")
        timer = TimeKeeper()
        label_names = ['time', 'label']
        self.label = pd.read_table(self.root_path + 'Label.txt', header = None, names = label_names, sep = "\t")
        self.label = self.label.iloc[:int(self.load_ratio * self.label.shape[0]),:] if self.load_ratio else self.label
        print("Label 读取完成，共 {} 条数据，用时 {}s".format(self.label.shape[0], timer.get_update_time()))

    def load_loc(self):
        print("Location Loading...")
        timer = TimeKeeper()
        loc_names = ['time', 'ign1', 'ign2', 'accuracy', 'latitude', 'longitude', 'altitude']
        self.loc = pd.read_table(self.root_path + 'Location.txt', header = None, names = loc_names, sep = " ").drop(['ign1', 'ign2'], axis = 1)
        self.loc = self.loc.iloc[:int(self.load_ratio * self.loc.shape[0]),:] if self.load_ratio else self.loc
        print("Location 读取完成，共 {} 条数据，用时 {}s".format(self.loc.shape[0], timer.get_update_time()))

    def load_wifi(self, detail = True):
        print("Wifi Loading...")
        timer = TimeKeeper()
        wifi = pd.read_table(self.root_path + 'Wifi.txt', header = None)
        wifi['time'] = wifi.apply(lambda x: x[0].split(";")[0], axis = 1)
        wifi['number'] = wifi.apply(lambda x: x[0].split(";")[3], axis = 1)
        self.wifi = wifi.iloc[:int(self.load_ratio * wifi.shape[0]),:] if self.load_ratio else wifi
        print("Wifi 读取完成，共 {} 条数据，用时 {}s".format(wifi.shape[0], timer.get_update_time()))
        if detail:
            print("Wifi Detail Loading...")
            wifi_detail = self.wifi.apply(lambda x: self.wifi_detail_transformer(x[0].split(";")[4:], x[0].split(";")[0]), axis = 1)
            self.wifi_detail = pd.concat(wifi_detail.to_list()).reset_index(drop = True)
            print("\t-- Wifi 详细信息提取完成，共 {} 条数据，用时 {}s".format(wifi_detail.shape[0], timer.get_update_time()))
        self.wifi = self.wifi.drop([0], axis = 1)

    def load_gps(self, detail = True):
        print("GPS Loading...")
        timer = TimeKeeper()
        gps = pd.read_table(self.root_path + 'GPS.txt', header = None)
        gps['time'] = gps.apply(lambda x: x[0].split(" ")[0], axis = 1)
        gps['number'] = gps.apply(lambda x: x[0].split(" ")[-1], axis = 1)
        self.gps = gps.iloc[:int(self.load_ratio * gps.shape[0]),:] if self.load_ratio else gps
        print("GPS 读取完成，共 {} 条数据，用时 {}s".format(gps.shape[0], timer.get_update_time()))
        if detail:
            print("GPS Detail Loading...")
            gps_detail = self.gps.apply(lambda x: self.gps_detail_transformer(x[0].split(" ")[3:-1], x[0].split(";")[0]), axis = 1)
            self.gps_detail = pd.concat(gps_detail.to_list()).reset_index(drop = True)
            print("\t-- GPS 详细信息提取完成，共 {} 条数据，用时 {}s".format(gps_detail.shape[0], timer.get_update_time()))
        self.gps = self.gps.drop([0], axis = 1)
        
    def load_cells(self, detail = True):
        print("Cells Loading...")
        timer = TimeKeeper()
        cells = pd.read_table(self.root_path + 'Cells.txt', header = None)
        cells['time'] = cells.apply(lambda x: x[0].split(" ")[0], axis = 1)
        cells['number'] = cells.apply(lambda x: x[0].split(" ")[3], axis = 1)
        self.cells = gps.iloc[:int(self.load_ratio * cells.shape[0]),:] if self.load_ratio else cells
        print("Cells 读取完成，共 {} 条数据，用时 {}s".format(cells.shape[0], timer.get_update_time()))
        if detail:
            print("Cells Detail Loading...")
            cells_detail = self.cells.apply(lambda x: self.cells_detail_transformer(x[0], x[0].split(" ")[0]), axis = 1)
            self.cells_detail = pd.concat(cells_detail.to_list()).reset_index(drop = True)
            print("\t-- Cells 详细信息提取完成，共 {} 条数据，用时 {}s".format(cells_detail.shape[0], timer.get_update_time()))
        self.cells = self.cells.drop([0], axis = 1)
    
    def load_all(self):

        self.load_loc()
        self.load_wifi()
        self.load_gps()
        self.load_cells()
        self.load_label()