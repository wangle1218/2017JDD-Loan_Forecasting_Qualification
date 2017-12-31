#encoding:utf8
import os
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from gen_train_data import report,make_train_set
from sklearn.metrics import mean_squared_error,make_scorer
from gen_test_data import make_test_set
import time

params = {
    'task': 'train','boosting_type': 'gbdt','objective': 'regression',
    'metric': {'l2', 'rmse'},'max_depth':6,'num_leaves':80,
    'min_data_in_leaf':330,'learning_rate': 0.02,
    'feature_fraction': 0.8,'bagging_fraction': 0.8,'bagging_freq': 5,
    'verbose': -1}

params2 = {
    'task': 'train','boosting_type': 'gbdt','objective': 'regression',
    'metric': {'l2', 'rmse'},'max_depth':12,'num_leaves':210,
    'min_data_in_leaf':230,'learning_rate': 0.05,
    'feature_fraction': 0.8,'bagging_fraction': 0.8,'bagging_freq': 5,
    'verbose': -1}
# train
def lgb_model(df_train,df_test,params,feature_list,num):
    if os.path.exists('./model/model%s.pkl' % num):
        gbm = pickle.load(open('./model/model%s.pkl' % num,'rb'))
    else:
        lgb_train = lgb.Dataset(df_train[feature_list],label=df_train['label'],feature_name=feature_list)
        lgb_eval = lgb.Dataset(df_test[feature_list],label=df_test['label'],feature_name=feature_list, reference=lgb_train)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1500,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=50)
        # gbm.save_model('model.txt')
        # pickle.dump(gbm,open('./model/model%s.pkl' % num,'wb'))
    y_pred = gbm.predict(df_test[feature_list], num_iteration=gbm.best_iteration)
    print('The rmse of prediction is:', mean_squared_error(df_test['label'], y_pred) ** 0.5)

    return y_pred

data1_path = './data/model1_training.pkl'
user_path = './tmp/train_test_user.pkl'
train_test_user = pickle.load(open(user_path, 'rb'))
model1_train_user = train_test_user['model1_train_user']
model1_test_user = train_test_user['model1_test_user']
"""
train_test_user = {'model1_test_user':model1_test_user,'model1_train_user':model1_train_user,\
                            'model2_test_user':model2_test_user,'model2_train_user':model2_train_user,\
                            'sub_model1_user':sub_model1_user,'sub_model2_user':sub_model2_user}
"""
model1_training = pickle.load(open(data1_path,'rb'))
model1_train_data = model1_training[model1_training.uid.isin(model1_train_user)]
model1_test_data = model1_training[model1_training.uid.isin(model1_test_user)]
print(model1_test_data['label'].describe())
print(model1_train_data['label'].describe())

model1_train_data.sample(frac=1)


############################################
##########model 2 data ####################
data2_path = './data/model2_training.pkl'
user_path = './tmp/train_test_user.pkl'
train_test_user = pickle.load(open(user_path, 'rb'))
model2_train_user = train_test_user['model2_train_user']
model2_test_user = train_test_user['model2_test_user']
"""
train_test_user = {'model1_test_user':model1_test_user,'model1_train_user':model1_train_user,\
                            'model2_test_user':model2_test_user,'model2_train_user':model2_train_user,\
                            'sub_model1_user':sub_model1_user,'sub_model2_user':sub_model2_user}
"""
model2_training = pickle.load(open(data2_path,'rb'))
model2_train_data = model2_training[model2_training.uid.isin(model2_train_user)]
model2_test_data = model2_training[model2_training.uid.isin(model2_test_user)]
print(model2_test_data['label'].describe())
print(model2_train_data['label'].describe())

model2_train_data.sample(frac=1)

feature_list = list(model2_training.columns)
feature_list.remove('uid')
feature_list.remove('label')


############## predict #########################
start = time.time()
# pred1 = lgb_model(model1_train_data,model1_test_data,params,feature_list,1)
pred1 = lgb_model(model2_train_data,model2_test_data,params2,feature_list,2)
# pred1 = np.where(pred1>0.5,pred1*7.5,pred1*0.3)
report(np.array(model2_test_data['label']), np.array(pred1))

# df_test['pred_loan'] = np.abs(y_pred)
# pickle.dump(df_test[['uid','label','pred_loan']],open('lgb7.pkl','wb'))

end = time.time()
print('train time:',(end-start)/60.)

##############submission#################
# X_test = make_test_set()
# data_path = './data/test.pkl'
# X_test = pickle.load(open(data_path,'rb'))
# sub = X_test.copy()
# del X_test['uid']
# X_test.columns = feature_list

# gbm1 = pickle.load(open('./model/lgb_model1.pkl','rb'))
# pred1 = gbm1.predict(X_test[feature_list], num_iteration=gbm1.best_iteration)
# pred1 = np.where(pred1< 0. ,-pred1,pred1)

# gbm2 = pickle.load(open('./model/lgb_model2.pkl','rb'))
# pred2 = gbm2.predict(X_test[feature_list], num_iteration=gbm2.best_iteration)

# gbm3 = pickle.load(open('./model/lgb_model3.pkl','rb'))
# pred3 = gbm3.predict(X_test[feature_list], num_iteration=gbm3.best_iteration)

# gbm4 = pickle.load(open('./model/lgb_model4.pkl','rb'))
# pred4 = gbm4.predict(X_test[feature_list], num_iteration=gbm4.best_iteration)

# gbm5 = pickle.load(open('./model/lgb_model5.pkl','rb'))
# pred5 = gbm5.predict(X_test[feature_list], num_iteration=gbm5.best_iteration)
# pred5 = np.where(pred5< 0. ,-pred5,pred5)

# # pred = np.array(pred)
# # pred = np.where(pred< 0. ,-pred,pred)

# sub['pred1'] = pred1
# sub['pred2'] = pred2
# sub['pred3'] = pred3
# sub['pred4'] = pred4
# sub['pred5'] = pred5

# sub = sub[['uid','pred1','pred2','pred3','pred4','pred5']]
# sub['mean'] = sub[['pred1','pred5']].apply(lambda x : x.mean(),axis=1)
# print(sub.describe())
# sub = sub[['uid','mean']]
# sub.to_csv('./sub/sub_5.csv',sep=',',header=None,index=False,encoding='utf8')
# # print(sub.describe())








