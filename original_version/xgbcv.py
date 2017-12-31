#encoding:utf8
import os
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pickle
from gen_train_data import spilt_train_test,report
from gen_test_data import make_test_set

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_
RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

def xgboost_cv():
    X_train,X_test,y_train,y_test,_,_ = spilt_train_test()

    if os.path.exists('./model/mul_month2.model'):
        model = xgb.Booster({'nthread':4})
        model.load_model('./model/mul_month2.model')
    else:
        xgb_model = xgb.XGBRegressor()

        parameters = {'nthread':[4],
                      'objective':['reg:linear'],
                      'gamma': [0, 1, 5, 10, 100],
                      'learning_rate': [0.01,0.1,0.2], 
                      'max_depth': [5],
                      'min_child_weight': [3],
                      'silent': [1],
                      'subsample': [0.8],
                      'colsample_bytree': [0.7],
                      'n_estimators': [1000], 
                      'seed': [1270]}


        gridsearch = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                                   cv = StratifiedKFold(n_splits=5,random_state=0,shuffle=False), 
                                   scoring = RMSE,
                                   verbose =2, refit = True)

        gridsearch.fit(X_train, y_train)

        #trust your CV!
        best_parameters, score, _ = max(gridsearch.grid_scores_, key=lambda x: x[1])
        print('Raw rmse score:', score)

        for param_name in sorted(best_parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))
        print(gridsearch.grid_scores_)

        # gridsearch.save_model('./model/mul_month28b.model')
    # plot_importance(model)

    # # 对测试集进行预测
    # dtest = xgb.DMatrix(X_test)
    # pred = gridsearch.predict(dtest)
    # pred = np.array(pred)
    # pred = np.where(pred <= 0.,-pred,pred)

    # report(pred,y_test)

def xgb_submission():
    model = xgb.Booster({'nthread':4})
    model.load_model('./model/mul_month2.model')

    X_test = make_test_set()
    sub = X_test.copy()
    del X_test['uid']

    X_test = np.array(X_test)
    # scaler = StandardScaler()
    # X_test = scaler.fit_transform(X_test)

    dtest = xgb.DMatrix(X_test)
    pred = model.predict(dtest)

    pred = np.array(pred)
    pred = np.where(pred< 0. ,0,pred)

    sub['loan'] = pred
    sub = sub[['uid','loan']]
    df_user = pickle.load(open('df_user.pkl','rb'))
    sub = pd.merge(sub,df_user,how='left',on='uid')
    # pred = list(sub['loan'].copy())
    limit = list(sub['limit'].copy())
    true_pred = []
    for i in range(len(pred)):
        if pred[i] > limit[i]:
            true_pred.append(limit[i])
        else:
            true_pred.append(pred[i])
    sub['true_pred'] = true_pred
    sub = sub[['uid','true_pred']]
    sub.to_csv('./sub/sub28c2.csv',sep=',',header=None,index=False,encoding='utf8')
    print(sub.describe())



if __name__ == '__main__':
    import time
    start = time.time()
    xgboost_cv()
    # xgb_submission()
    end = time.time()
    print('train time:',(end-start)/60.)






