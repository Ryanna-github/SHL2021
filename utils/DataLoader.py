import pandas as pd
import numpy as np
import re
import time
from utils.TimeKeeper import TimeKeeper


class SHLDataLoader():
    def __init__(self, root_path):
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
            - 'ctype', 'registered', 'id', 'lac', 'mmc', 'mnc', 'dbm', 'level1', 'level2'
        """
        # get cell type
        cells_type = re.findall(" LTE | GSM | WCDMA ", cells_list)
        # get detail info
        cells_info = re.split(" LTE | GSM | WCDMA ", cells_list)[1:]
        res = []
        for i, ctype in enumerate(cells_type):
            if re.match('.*LTE.*', ctype):
                info = ['LTE'] + np.array(cells_info[i].split(" "))[[0, 1, 4, 2, 3, 7, 6, 8]].tolist()
            elif re.match('.*GSM.*', ctype):
                info = ['GSM'] + np.array(cells_info[i].split(" "))[[0, 1, 2, 3, 4, 6, 5, 7]].tolist()
            elif re.match(".*WCDMA.*", ctype):
                info = ['WCDMA'] + np.array(cells_info[i].split(" "))[[0, 1, 2, 3, 4, 7, 6, 8]].tolist()
            else:
                raise ValueError("Unrecognized cell type {}".format(ctype))
            res.append(info)

        res = pd.DataFrame(res, columns = ('ctype', 'registered', 'id', 'lac', 'mmc', 'mnc', 'dbm', 'level1', 'level2'))
        res.insert(loc = 0, column = 'time', value = [this_time] * len(cells_type))
        return res

    def load_label(self):
        timer = TimeKeeper()
        label_names = ['time', 'label']
        self.label = pd.read_table(self.root_path + 'Label.txt', header = None, names = label_names, sep = "\t")
        print("Label 读取完成，共 {} 条数据，用时 {}s".format(self.label.shape[0], timer.get_update_time()))

    def load_loc(self):
        timer = TimeKeeper()
        loc_names = ['time', 'ign1', 'ign2', 'accuracy', 'latitude', 'longitude', 'altitude']
        self.loc = pd.read_table(self.root_path + 'Location.txt', header = None, names = loc_names, sep = " ").drop(['ign1', 'ign2'], axis = 1)
        print("Location 读取完成，共 {} 条数据，用时 {}s".format(self.loc.shape[0], timer.get_update_time()))

    def load_wifi(self, detail = True):
        timer = TimeKeeper()
        wifi = pd.read_table(self.root_path + 'Wifi.txt', header = None)
        wifi['time'] = wifi.apply(lambda x: x[0].split(";")[0], axis = 1)
        wifi['number'] = wifi.apply(lambda x: x[0].split(";")[3], axis = 1)
        print("Wifi 读取完成，共 {} 条数据，用时 {}s".format(wifi.shape[0], timer.get_update_time()))
        if detail:
            wifi_detail = wifi.apply(lambda x: self.wifi_detail_transformer(x[0].split(";")[4:], x[0].split(";")[0]), axis = 1)
            self.wifi_detail = pd.concat(wifi_detail.to_list()).reset_index(drop = True)
            print("\t-- Wifi 详细信息提取完成，共 {} 条数据，用时 {}s".format(wifi_detail.shape[0], timer.get_update_time()))
        self.wifi = wifi.drop([0], axis = 1)

    def load_gps(self, detail = True):
        timer = TimeKeeper()
        gps = pd.read_table(self.root_path + 'GPS.txt', header = None)
        gps['time'] = gps.apply(lambda x: x[0].split(" ")[0], axis = 1)
        gps['number'] = gps.apply(lambda x: x[0].split(" ")[-1], axis = 1)
        print("GPS 读取完成，共 {} 条数据，用时 {}s".format(gps.shape[0], timer.get_update_time()))
        if detail:
            gps_detail = gps.apply(lambda x: self.gps_detail_transformer(x[0].split(" ")[3:-1], x[0].split(";")[0]), axis = 1)
            self.gps_detail = pd.concat(gps_detail.to_list()).reset_index(drop = True)
            print("\t-- GPS 详细信息提取完成，共 {} 条数据，用时 {}s".format(gps_detail.shape[0], timer.get_update_time()))
        self.gps = gps.drop([0], axis = 1)
        
    def load_cells(self, detail = True):
        timer = TimeKeeper()
        cells = pd.read_table(self.root_path + 'Cells.txt', header = None)
        cells['time'] = cells.apply(lambda x: x[0].split(" ")[0], axis = 1)
        cells['number'] = cells.apply(lambda x: x[0].split(" ")[3], axis = 1)
        print("Cells 读取完成，共 {} 条数据，用时 {}s".format(cells.shape[0], timer.get_update_time()))
        if detail:
            cells_detail = cells.apply(lambda x: self.cells_detail_transformer(x[0], x[0].split(" ")[0]), axis = 1)
            self.cells_detail = pd.concat(cells_detail.to_list()).reset_index(drop = True)
            print("\t-- Cells 详细信息提取完成，共 {} 条数据，用时 {}s".format(cells_detail.shape[0], timer.get_update_time()))
        self.cells = cells.drop([0], axis = 1)
    
    def load_all_small(self):
        self.load_loc()
        self.load_wifi()
        self.load_gps()
        self.load_cells()
        self.load_label()


    def load_all_small(self, detail_num = 1000):
        """
        Description: Extract all info from root path (end with "/" "data/train/" for example)
        return: dict
            - keys: label, loc, wifi, wifi_detail, gps, gps_detail, cells
        """
        timer = TimeKeeper()
        # ------------------------------------- Label -------------------------------------
        label_names = ['time', 'label']
        self.label = pd.read_table(self.root_path + 'Label.txt', header = None, names = label_names, sep = "\t")
        print("Label 读取完成，共 {} 条数据，用时 {}s".format(self.label.shape[0], timer.get_update_time()))
        # ----------------------------------- Location ------------------------------------
        loc_names = ['time', 'ign1', 'ign2', 'accuracy', 'latitude', 'longitude', 'altitude']
        self.loc = pd.read_table(self.root_path + 'Location.txt', header = None, names = loc_names, sep = " ").drop(['ign1', 'ign2'], axis = 1)
        print("Location 读取完成，共 {} 条数据，用时 {}s".format(self.loc.shape[0], timer.get_update_time()))
        # ------------------------------------- Wifi --------------------------------------
        wifi = pd.read_table(self.root_path + 'Wifi.txt', header = None)
        wifi['time'] = wifi.apply(lambda x: x[0].split(";")[0], axis = 1)
        wifi['number'] = wifi.apply(lambda x: x[0].split(";")[3], axis = 1)
        print("Wifi 读取完成，共 {} 条数据，用时 {}s".format(wifi.shape[0], timer.get_update_time()))
        # detail info
        wifi_detail = wifi.iloc[:detail_num,:].apply(lambda x: self.wifi_detail_transformer(x[0].split(";")[4:], x[0].split(";")[0]), axis = 1)
        self.wifi_detail = pd.concat(wifi_detail.to_list()).reset_index(drop = True)
        print("\t-- Wifi 详细信息提取完成，共 {} 条数据，提取了 Wifi 中前 {} 行，用时 {}s".format(wifi_detail.shape[0], detail_num, timer.get_update_time()))
        # delete raw data
        self.wifi = wifi.drop([0], axis = 1)
        # -------------------------------------- GPS --------------------------------------
        gps = pd.read_table(self.root_path + 'GPS.txt', header = None)
        gps['time'] = gps.apply(lambda x: x[0].split(" ")[0], axis = 1)
        gps['number'] = gps.apply(lambda x: x[0].split(" ")[-1], axis = 1)
        print("GPS 读取完成，共 {} 条数据，用时 {}s".format(gps.shape[0], timer.get_update_time()))
        # detail info
        gps_detail = gps.iloc[:detail_num,:].apply(lambda x: self.gps_detail_transformer(x[0].split(" ")[3:-1], x[0].split(";")[0]), axis = 1)
        self.gps_detail = pd.concat(gps_detail.to_list()).reset_index(drop = True)
        print("\t-- GPS 详细信息提取完成，共 {} 条数据，提取了 GPS 中前 {} 行，用时 {}s".format(gps_detail.shape[0], detail_num, timer.get_update_time()))
        # delete raw data
        self.gps = gps.drop([0], axis = 1)
        # -------------------------------------- Cells --------------------------------------
        cells = pd.read_table(self.root_path + 'Cells.txt', header = None)
        cells['time'] = cells.apply(lambda x: x[0].split(" ")[0], axis = 1)
        cells['number'] = cells.apply(lambda x: x[0].split(" ")[3], axis = 1)
        self.cells = cells.drop([0], axis = 1)
        print("Cells 读取完成，共 {} 条数据，用时 {}s".format(cells.shape[0], timer.get_update_time()))
        # detail info
        cells_detail = cells.iloc[:detail_num,:].apply(lambda x: self.cells_detail_transformer(x[0], x[0].split(" ")[0]), axis = 1)
        self.cells_detail = pd.concat(cells_detail.to_list()).reset_index(drop = True)
        print("\t-- Cells 详细信息提取完成，共 {} 条数据，提取了 Cells 中前 {} 行，用时 {}s".format(cells_detail.shape[0], detail_num, timer.get_update_time()))
        # delete raw data
        self.cells = cells.drop([0], axis = 1)
