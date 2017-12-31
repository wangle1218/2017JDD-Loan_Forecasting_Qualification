#encoding:utf8
import os
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import json
from gen_train_data import report
from lightgbm.sklearn import LGBMRegressor 
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from gen_test_data import make_test_set

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_
RMSE = make_scorer(fmean_squared_error, greater_is_better=False)


data_path = './data/train_set.pkl'
df = pickle.load(open(data_path,'rb'))
df.sample(frac=1)

feature_list = list(df.columns)
feature_list.remove('uid')
feature_list.remove('label')
feature_list.remove('label2')

test_ratio = int(len(df) * 0.2)
df_train = df[:-test_ratio]
df_test = df[-test_ratio:]

lgb_train = lgb.Dataset(df_train[feature_list],label=df_train['label'],feature_name=feature_list)
lgb_eval = lgb.Dataset(df_test[feature_list],label=df_test['label'],feature_name=feature_list, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'rmse'},
    # 'max_depth':6,
    # 'num_leaves':90,
    # 'min_data_in_leaf':110,
    'learning_rate': 0.02,
    # 'max_bin': 35,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': 0
}

param_test = {  
        'max_depth': range(4,15,2),  
        'num_leaves': range(10,240,20),
        }

print('Start training...')
# train

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1500,
                valid_sets=lgb_eval,
                early_stopping_rounds=50)

# cv
# gsearch = GridSearchCV(gbm , param_grid = param_test, n_jobs=5, 
#                        cv = StratifiedKFold(n_splits=5,random_state=0,shuffle=False), 
#                        scoring = RMSE,
#                        verbose =0, refit = True)

# gsearch.fit(df_train[feature_list].values, df_train['label'].values) 

# best_parameters, score, _ = max(gridsearch.grid_scores_, key=lambda x: x[1])
# print('Raw rmse score:', score)

# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))
# print(gridsearch.grid_scores_)  

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(df_test[feature_list], num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(df_test['label'], y_pred) ** 0.5)

report(np.array(df_test['label']), np.abs(y_pred))

##############submission#################
X_test = make_test_set()

sub = X_test.copy()
del X_test['uid']
X_test.columns = feature_list

pred = gbm.predict(X_test[feature_list], num_iteration=gbm.best_iteration)

pred = np.array(pred)
pred = np.where(pred< 0. ,-pred,pred)

sub['loan'] = pred
sub = sub[['uid','loan']]

sub.to_csv('./sub/sub6.csv',sep=',',header=None,index=False,encoding='utf8')
print(sub.describe())

"""
{0: 'uid', 1: 'age', 2: 'a_date', 3: 'sex_1', 4: 'sex_2', 5: 'rank_A', 6: 'rank_B', 7: 'rank_C', 8: 'rank_D',
9: 'rank_E', 10: 'buy_weights', 11: 'cost_weight', 12: 'real_price', 13: 'dis_ratio', 14: 'buy_min', 15: 'buy_mean', 
16: 'buy_max', 17: 'buy_sum', 18: 'loan_amount', 19: 'plannum', 20: 'loan_times', 21: 'loanTime_weights', 22: 'loan_weights',
23: 'repay', 24: 'month_9', 25: 'month_10', 26: 'month_11', 27: 'loan_months', 28: 'loan_12', 29: 'loan_13', 30: 'loan_23',
31: 'loan_123', 32: 'per_plannum', 33: 'per_times_loan', 34: 'per_month_loan', 35: 'loan_min', 36: 'loan_mean', 37: 'loan_max',
38: 'loan_sum', 39: 'plannum_01', 40: 'plannum_03', 41: 'plannum_06', 42: 'plannum_12', 43: 'loan_day_01_x', 44: 'loan_day_03_x',
45: 'loan_day_05_x', 46: 'loan_day_07_x', 47: 'loan_day_09_x', 48: 'loan_day_15_x', 49: 'loan_day_20_x', 50: 'loan_day_25_x', 
51: 'loan_day_30_x', 52: 'loan_day_35_x', 53: 'loan_day_40_x', 54: 'loan_day_50_x', 55: 'loan_day_60_x', 56: 'loan_day_70_x', 
57: 'loan_day_80_x', 58: 'loan_day_90_x', 59: 'loan_hours_01_x', 60: 'loan_hours_02_x', 61: 'loan_hours_03_x', 62: 'loan_hours_04_x', 
63: 'loan_hours_05_x', 64: 'loan_hours_06_x', 65: 'click_weights', 66: 'pid_1', 67: 'pid_2', 68: 'pid_3', 69: 'pid_4', 70: 'pid_5',
71: 'pid_6', 72: 'pid_7', 73: 'pid_8', 74: 'pid_9', 75: 'pid_10', 76: 'param_1', 77: 'param_2', 78: 'param_3', 79: 'param_4', 
80: 'param_5', 81: 'param_6', 82: 'param_7', 83: 'param_8', 84: 'param_9', 85: 'param_10', 86: 'param_11', 87: 'param_12',
88: 'param_13', 89: 'param_14', 90: 'param_15', 91: 'param_16', 92: 'param_17', 93: 'param_18', 94: 'param_19', 95: 'param_20',
96: 'param_21', 97: 'param_22', 98: 'param_23', 99: 'param_24', 100: 'param_25', 101: 'param_26', 102: 'param_27', 103: 'param_28',
104: 'param_29', 105: 'param_30', 106: 'param_31', 107: 'param_32', 108: 'param_33', 109: 'param_34', 110: 'param_35', 111: 'param_36', 
112: 'param_37', 113: 'param_38', 114: 'param_39', 115: 'param_40', 116: 'param_41', 117: 'param_42', 118: 'param_43', 119: 'param_44',
120: 'param_45', 121: 'param_46', 122: 'param_47', 123: 'param_48', 124: 'click_min', 125: 'click_max', 126: 'click_mean', 
127: 'click_sum', 128: 'loan_day_01_y', 129: 'loan_day_03_y', 130: 'loan_day_05_y', 131: 'loan_day_07_y', 132: 'loan_day_09_y',
133: 'loan_day_15_y', 134: 'loan_day_20_y', 135: 'loan_day_25_y', 136: 'loan_day_30_y', 137: 'loan_day_35_y', 138: 'loan_day_40_y',
139: 'loan_day_50_y', 140: 'loan_day_60_y', 141: 'loan_day_70_y', 142: 'loan_day_80_y', 143: 'loan_day_90_y', 144: 'loan_hours_01_y', 
145: 'loan_hours_02_y', 146: 'loan_hours_03_y', 147: 'loan_hours_04_y', 148: 'loan_hours_05_y', 149: 'loan_hours_06_y', 150: 'repay_cost'}

"""







