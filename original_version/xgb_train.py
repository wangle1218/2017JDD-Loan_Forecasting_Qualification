#encoding:utf8
import os
import math
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from xgboost import plot_importance
from gen_train_data import spilt_train_test,report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from gen_test_data import make_test_set

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_

RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

def xgboost_reg():
    data1_path = './data/training.pkl'
    # user_path = './tmp/train_test_user.pkl'
    # train_test_user = pickle.load(open(user_path, 'rb'))
    # model1_train_user = train_test_user['model1_train_user']
    # model1_test_user = train_test_user['model1_test_user']
    model1_training = pickle.load(open(data1_path,'rb'))
    model1_training = model1_training.sample(frac=1)
    # model1_train_data = model1_training[model1_training.uid.isin(model1_train_user)]
    # model1_test_data = model1_training[model1_training.uid.isin(model1_test_user)]
    # model1_train_data = model1_train_data.sample(frac=1)
    test_num = int(len(model1_training) * 0.2)
    model1_train_data = model1_training[:-test_num]
    model1_test_data = model1_training[-test_num:]

    feature_list = list(model1_train_data.columns)
    # feature_list.remove('uid')
    feature_list.remove('label')
    # feature_list.remove('dis_ratio')

    if os.path.exists('./model/xgb1.model'):
        model = xgb.Booster({'nthread':4})
        model.load_model('./model/xgb10.model')
        score = model.get_fscore()
        print(sorted(score.items(), key=lambda e:e[1]))
    else:
        params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'gamma': 0.1,
            'max_depth': 5,
            'lambda': 1,
            'subsample': 0.75,
            'colsample_bytree': 0.8,
            'min_child_weight':40,
            'silent': 1,
            'eta': 0.02,
            'seed': 133,
            'n_estimators':2000,
            'eval_metric':'rmse',
            'early_stopping_rounds': 65,
            'nthread': 4,
        }
        dtrain = xgb.DMatrix(model1_train_data[feature_list], model1_train_data['label'])
        num_rounds = 1000
        plst = params.items()
        model = xgb.train(plst, dtrain, num_rounds)
        model.save_model('./model/xgb11.model')

        # plot_importance(model)

    # 对测试集进行预测
    dtest = xgb.DMatrix(model1_test_data[feature_list])
    pred = model.predict(dtest)
    pred = np.array(pred)
    pred = np.where(pred <= 0.,0,pred)
    # pred = np.where(pred >= 4.,0.4*pred,pred)

    report(model1_test_data['label'],pred)

def xgboost_clf():
    X_train,X_test,_,_,y_train,y_test = spilt_train_test()
    if os.path.exists('./model/user_clf2.model'):
        model = xgb.Booster({'nthread':4})
        model.load_model('./model/user_clf.model')
    else:

        params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',
            'num_class':4,
            'gamma': 0.1,
            'max_depth': 5,
            'lambda': 1,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'min_child_weight': 2,
            'silent': 1,
            'eta': 0.025,
            'seed': 800,
            'scale_pos_weight':1,
            'early_stopping_rounds': 35,
            'nthread': 4,
        }

        dtrain = xgb.DMatrix(X_train, y_train)
        num_rounds = 1000
        plst = params.items()
        model = xgb.train(plst, dtrain, num_rounds)
        model.save_model('./model/user_clf.model')
    # plot_importance(model)

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    pred = model.predict(dtest)
    pred = np.array(pred)
    pickle.dump(pred,open('user.clf.pkl','wb'))
    precision = clf_report(y_test,pred)
    confusion_m = confusion_matrix(y_test,pred,labels=[0,1,2,3])
    print('confusion matrix:\n',confusion_m)
    print('precision:{}'.format(precision))
    y_train = list(y_train)
    y_test = list(y_test)
    print(y_train.count(0), y_test.count(0))
    print(y_train.count(1), y_test.count(1))
    print(y_train.count(2), y_test.count(2))
    print(y_train.count(3), y_test.count(3))

def xgb_submission():
    model = xgb.Booster({'nthread':4})
    model.load_model('./model/xgb10.model')

    X_test = make_test_set()
    # feat = sorted(list(X_test.columns))
    # X_test = X_test[feat]

    sub = X_test.copy()
    # del X_test['uid']

    X_test = np.array(X_test)

    dtest = xgb.DMatrix(X_test)
    pred = model.predict(dtest)

    pred = np.array(pred)
    pred = np.where(pred< 0. ,0,pred)

    sub['loan'] = pred
    sub = sub[['uid','loan']]

    sub.to_csv('./sub/sub11.csv',sep=',',header=None,index=False,encoding='utf8')
    print(sub.describe())



if __name__ == '__main__':
    import time
    start = time.time()
    # xgboost_reg()
    # xgboost_clf()
    xgb_submission()
    end = time.time()
    print('train time:',(end-start)/60.)






