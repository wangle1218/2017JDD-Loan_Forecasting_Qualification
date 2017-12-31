#encoding:utf8
import os
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import json
import random
import xgboost as xgb
from xgboost import plot_importance
from gen_train_data import report,make_train_set
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix
from gen_test_data import make_test_set
import time
start = time.time()

data_path1 = './data/training.pkl'
user_path = './tmp/train_test_user.pkl'
train_test_user = pickle.load(open(user_path, 'rb'))
model1_train_user = train_test_user['model1_train_user']
model1_test_user = train_test_user['model1_test_user']

model1_training = pickle.load(open(data_path1,'rb'))
model1_train_data = model1_training[model1_training.uid.isin(model1_train_user)]
model1_train_data['label'][model1_train_data['label'] != 0] = 1

model1_test_data = model1_training[model1_training.uid.isin(model1_test_user)]
model1_test_data['label'][model1_test_data['label'] != 0] = 1
# df['label2'] = df['label2'].map({0:0,1:1, 2:0, 3:1})

# df.sample(frac=1)

feature_list = list(model1_training.columns)
feature_list.remove('uid')
feature_list.remove('label')
# feature_list.remove('label2')

# test_ratio = int(len(df) * 0.2)
# df_train = df[:-test_ratio]
# df_test = df[-test_ratio:]

# train_pos = df_train[df_train['label']==1]
# train_neg = df_train[df_train['label']==0]

# neg_uid = train_neg['uid'].tolist()
# samp_uid_list = random.sample(neg_uid, 18500)
# train_neg1 = train_neg[train_neg.uid.isin(samp_uid_list)]
# samp_uid_list = random.sample(neg_uid, 18500)
# train_neg2 = train_neg[train_neg.uid.isin(samp_uid_list)]
# samp_uid_list = random.sample(neg_uid, 18500)
# train_neg3 = train_neg[train_neg.uid.isin(samp_uid_list)]
# samp_uid_list = random.sample(neg_uid, 18500)
# train_neg4 = train_neg[train_neg.uid.isin(samp_uid_list)]
# samp_uid_list = random.sample(neg_uid, 18500)
# train_neg5 = train_neg[train_neg.uid.isin(samp_uid_list)]

def lgb_model(train_data,df_test,feature_list):
    # train_data = pos.append(neg)
    train_data = train_data.sample(frac=1)
    print(train_data['label'].value_counts())
    print(df_test['label'].value_counts())
    lgb_train = lgb.Dataset(train_data[feature_list],label=train_data['label'],feature_name=feature_list)
    lgb_eval = lgb.Dataset(df_test[feature_list],label=df_test['label'],feature_name=feature_list, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        # 'is_unbalance':True,
        # 'scale_pos_weight':0.2,
        'metric': {'auc'},
        'max_depth':5,
        'num_leaves':25,
        'min_data_in_leaf':300,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'bagging_seed':112,
        'verbose': -1
    }

    print('Start training...')
    # train

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1500,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100)

    print('Save model...')
    # save model to file
    # gbm.save_model('clf_model.txt')

    print('Start predicting...')
    # predict
    y_pred = gbm.predict(df_test[feature_list], num_iteration=gbm.best_iteration)
    # print(y_pred[:20])
    pred = np.where(y_pred >= 0.5,1, 0)
    # print result
    print('0\t1')
    print(df_test['label'].tolist().count(0),df_test['label'].tolist().count(1))
    print(pred.tolist().count(0), pred.tolist().count(1))

    precision = np.mean(np.array(df_test['label']).astype(np.int32) == np.array(pred).astype(np.int32))
    r = recall_score(np.array(df_test['label']).astype(np.int32), np.array(pred).astype(np.int32))
    f1 = f1_score(np.array(df_test['label']).astype(np.int32), np.array(pred).astype(np.int32))
    print('precision:{},recal:{},f1:{}'.format(precision,r,f1))

    confusion_m = confusion_matrix(df_test['label'].values,pred,labels=[0,1])
    print('confusion matrix:\n',confusion_m)
    print('-'*20)
    return pred

def xgb_model(train_data,df_test,feature_list,i):
    # train_data = pos.append(neg)
    train_data.sample(frac=1)
    if os.path.exists('./model/xgb_%s.model' % str(i)):
        model = xgb.Booster({'nthread':4})
        model.load_model('./model/xgb_%s.model' % str(i))
    else:

        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric':'error',
            'gamma': 0.1,
            'max_depth': 7,
            'lambda': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'silent': 1,
            'eta': 0.025,
            'seed': 800,
            # 'scale_pos_weight':0.2,
            'early_stopping_rounds': 35,
            'nthread': 4,
        }

        dtrain = xgb.DMatrix(train_data[feature_list],train_data['label'])
        num_rounds = 1000
        plst = params.items()
        model = xgb.train(plst, dtrain, num_rounds)
        # model.save_model('./model/xgb_%s.model' % str(i))
    # plot_importance(model)

    # 对测试集进行预测
    dtest = xgb.DMatrix(df_test[feature_list])
    pred = model.predict(dtest)
    pred = np.array(pred)
    pred = np.where(pred >= 0.5,1, 0)
    # pickle.dump(pred,open('user.clf.pkl','wb'))
    precision = precision_score(df_test['label'].values,pred)
    r = recall_score(df_test['label'].values,pred)
    f1 = f1_score(df_test['label'].values,pred)

    print('precision:{},recal:{},f1:{}'.format(precision,r,f1))
    confusion_m = confusion_matrix(df_test['label'].values,pred,labels=[0,1])
    print('confusion matrix:\n',confusion_m)

    return pred

pred1 = lgb_model(model1_train_data,model1_test_data,feature_list)
# pred2 = lgb_model(train_pos,train_neg2,df_test,feature_list)
# pred3 = lgb_model(train_pos,train_neg3,df_test,feature_list)
# pred4 = lgb_model(train_pos,train_neg4,df_test,feature_list)
# pred5 = lgb_model(train_pos,train_neg5,df_test,feature_list)

# pred1 = xgb_model(model1_train_data,model1_test_data,feature_list,8)

# pred2 = xgb_model(train_pos,train_neg2,df_test,feature_list,2)
# pred3 = xgb_model(train_pos,train_neg3,df_test,feature_list,3)
# pred4 = xgb_model(train_pos,train_neg4,df_test,feature_list,4)
# pred5 = xgb_model(train_pos,train_neg5,df_test,feature_list,5)

# df_test['class1'] = pred1.tolist()
# df_test['class2'] = pred2.tolist()
# df_test['class3'] = pred3.tolist()
# df_test['class4'] = pred4.tolist()
# df_test['class5'] = pred5.tolist()

# df_test = df_test[['uid','class1','class2','class3','class4','class5']]
# pickle.dump(df_test,open('./sub/clfpred_xgb.pkl','wb'))
end = time.time()
print('train time:',(end-start)/60.)

##############submission#################
# X_test = make_test_set()
# data_path = './data/test_data.pkl'
# X_test = pickle.load(open(data_path,'rb'))
# sub = X_test.copy()
# del X_test['uid']
# X_test.columns = feature_list

# pred = gbm.predict(X_test[feature_list], num_iteration=gbm.best_iteration)

# pred = np.array(pred)
# pred = np.where(pred< 0. ,-pred,pred)

# sub['loan'] = pred
# sub = sub[['uid','loan']]

# sub.to_csv('./sub/sub6.csv',sep=',',header=None,index=False,encoding='utf8')
# print(sub.describe())








