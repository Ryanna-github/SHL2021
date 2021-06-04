import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

from utils import helper
from utils import TimeKeeper
from utils import DataLoader, DataProcesser

class Solver():
    def __init__(self, X_train, y_train, X_val = None, y_val = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        print("X_train: {}".format(self.X_train.shape))
        print("y_train: {}".format(len(self.y_train)))
        if self.X_val:
            print("X_val  : {}".format(self.X_val.shape))
        if self.y_val:
            print("y_val  : {}".format(len(self.y_val)))

    def train(self):
        self.train_model_rf()
        self.train_model_bag()
        self.train_model_lgb()

    def train_model_rf(self):
        print("Training rf...")
        timer = TimeKeeper.TimeKeeper()
        self.model_rf = RandomForestClassifier(n_estimators = 20, random_state = 0, max_depth = 8)
        self.model_rf.fit(self.X_train, self.y_train)
        print("Time elapsed for training rf: {}".format(timer.get_update_time()))
    
    def train_model_bag(self):
        print("Training bag (tree-based)...")
        timer = TimeKeeper.TimeKeeper()
        self.model_bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 100, max_samples = 100, bootstrap=True)
        self.model_bag.fit(self.X_train, self.y_train)
        print("Time elapsed for training bag (tree-based): {}".format(timer.get_update_time()))
    
    def train_model_lgb(self):
        print("Training LightGBM...")
        timer = TimeKeeper.TimeKeeper()
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size = 0.2)
        train_data = lgb.Dataset(X_train, label = y_train - 1)
        test_data = lgb.Dataset(X_test, label = y_test - 1)
        params={
            'learning_rate':0.1,
            'lambda_l1':0.1,
            'lambda_l2':0.2,
            'max_depth':6,
            'objective':'multiclass',
            'num_class':8,  
        }
        self.model_lgb = lgb.train(params, train_data, valid_sets = [test_data])
        print("Time elapsed for training LightGBM: {}".format(timer.get_update_time()))
    
    def predict_raw(self, X_val = None, y_val = None):
        try:
            self.pred_rf = self.model_rf.predict(X_val) 
            self.pred_prob_rf = self.model_rf.predict_proba(X_val) 
            self.pred_bag = self.model_bag.predict(X_val)
            self.pred_prob_bag = self.model_bag.predict_proba(X_val)
            self.pred_prob_lgb = self.model_lgb.predict(X_val)
            self.pred_lgb = self.pred_prob_lgb.argmax(axis = 1) + 1
        except:
            print("No X for prediction detected")

    def predict_mean(self, window_size = 120):
        pred_prob = self.pred_prob_rf + self.pred_prob_bag + self.pred_prob_lgb
        pred_prob_mean = pd.DataFrame(pred_prob).rolling(window_size, center = True).mean().fillna(method = 'ffill').fillna(method = 'bfill')
        self.pred_mean = pd.Series(np.array(pred_prob_mean).argmax(axis = 1) + 1)
        
        
