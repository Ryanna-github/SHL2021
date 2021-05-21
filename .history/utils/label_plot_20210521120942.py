import matplotlib.pyplot as plt 
import numpy as np

cmap = mpl.cm.get_cmap('viridis', 8)
label_cols = cmap(np.linspace(0, 1, 8))
label_cols = [mpl.colors.to_hex(i) for i in label_cols]

def plot_label_each(df, col_name, this_label):
    label_index = np.where(df.label == this_label)[0]
    p = plt.scatter(df.index[label_index], df[col_name][label_index], c = label_cols[this_label - 1], label = label_dic[this_label])
    return p

def plot_label(df, col_name):
    plt.figure(figsize = [12, 8])
    for this_label in np.unique(list(df.label)):
        plot_label_each(df, col_name, this_label)
    plt.legend(loc = 'best')