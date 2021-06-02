import pandas as pd 
import numpy as np
import utm
from sklearn.metrics import precision_score, confusion_matrix, classification_report
import seaborn as sns

label_dic = {1: 'Still', 2: 'Walking', 3: 'Run', 4: 'Bike', 5: 'Car', 6: 'Bus', 7: 'Train', 8: 'Subway'}

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

# evaluate prediction result
def evaluate(y_true, y_pred, names = list(label_dic.values())):
    conf = confusion_matrix(y_true , y_pred)
    print(conf)
    sns.heatmap(conf)
    print(classification_report(y_true, y_pred, target_names = names))

# save prediction result
def save_prediction(pred_time, pred_res, file_path = 'data/RY_predictions.txt'):
    res = pd.DataFrame({'time': pred_time, 'label': pred_res})
    res.to_csv(file_path, index = False, header = False, sep = '\t')

# plot prediction labels
def plot_prediction(y_pred, y_true):
    plt.figure(figsize = [20, 8])
    plt.plot(y_pred, alpha = 0.4)
    plt.plot(y_true)

# one hot encoder
def get_one_hot(df, class_col_name):
    X = df[class_col_name].values.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown = 'ignore', sparse = False).fit(X)
    col_names = ["{}_{}".format(class_col_name, i) for i in enc.categories_[0].tolist()]
    return pd.DataFrame(enc.transform(X), columns = col_names)
