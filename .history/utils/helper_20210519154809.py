import pandas as pd 
import numpy as np
import utm


# (latitude, longitude) -> east
def gps2utm_east(x):
    try:
        return utm.from_latlon(x['latitude'], x['longitude'])[0]
    except:
        return np.nan
# (latitude, longitude) -> north
def gps2utm_north(x):
    try:
        return utm.from_latlon(x['latitude'], x['longitude'])[1]
    except:
        return np.nan