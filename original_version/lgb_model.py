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

# train
def lgb_model(df_train,df_test,params,feature_list,num):
    if os.path.exists('./model/lgb_model%s.pkl' % num):
        gbm = pickle.load(open('./model/lgb_model%s.pkl' % num,'rb'))
    else:
        lgb_train = lgb.Dataset(df_train[feature_list],label=df_train['label'],feature_name=feature_list)
        lgb_eval = lgb.Dataset(df_test[feature_list],label=df_test['label'],feature_name=feature_list, reference=lgb_train)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1500,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=100)
        # gbm.save_model('model.txt')
        pickle.dump(gbm,open('./model/lgb_model77.pkl','wb'))
    y_pred = gbm.predict(df_test[feature_list], num_iteration=gbm.best_iteration)
    print('The rmse of prediction is:', mean_squared_error(df_test['label'], y_pred) ** 0.5)

    return y_pred

data_path = './data/training.pkl'
df = pickle.load(open(data_path,'rb'))
df = df.sample(frac=1)
# pickle.dump(df,open('./data/training.pkl','wb'))
# print(df[['order_cluster_label','loan_cluster_label','click_cluster_label','cluster_label']].describe())
feature_list = list(df.columns)
feature_list.remove('uid')
# [param_23,order_90,click_90,loan_90,plannum_mean,param_37,buy_sum,loan_sum,sex,plannum_min]
feature_list.remove('label')
# feature_list.remove('param_23')
# feature_list.remove('order_90')
# feature_list.remove('click_90')
# feature_list.remove('loan_90')
# feature_list.remove('plannum_mean')
# feature_list.remove('param_37')
# feature_list.remove('buy_sum')
# feature_list.remove('loan_sum')
# feature_list.remove('sex')
# feature_list.remove('plannum_min')


# feature_list.remove('label2')

test_num = int(len(df) * 0.2)
#############training data 1 ####################
params1 = {
    'task': 'train','boosting_type': 'gbdt','objective': 'regression',
    'metric': {'l2', 'rmse'},'max_depth':5,'num_leaves':21,
    'min_data_in_leaf':300,'learning_rate': 0.02,
    'feature_fraction': 0.75,'bagging_fraction': 0.8,'bagging_freq': 5,
    'verbose': -1}

df_train1 = df[:-test_num]
df_test1 = df[-test_num:]
#############training data 2 ####################
params2 = {
    'task': 'train','boosting_type': 'gbdt','objective': 'regression',
    'metric': {'l2', 'rmse'},'max_depth':5,'num_leaves':27,
    'min_data_in_leaf':350,'learning_rate': 0.02,
    'feature_fraction': 0.9,'bagging_fraction': 0.7,'bagging_freq': 5,
    'verbose': -1}

df_train2 = df[:test_num]
df_test2 = df[test_num:]
#############training data 3 ####################
params3 = {
    'task': 'train','boosting_type': 'gbdt','objective': 'regression',
    'metric': {'l2', 'rmse'},'max_depth':6,'num_leaves':27,
    'min_data_in_leaf':300,'learning_rate': 0.025,
    'feature_fraction': 0.8,'bagging_fraction': 0.8,'bagging_freq': 5,
    'verbose': -1}

df_train3 = df[:test_num].append(df[test_num *2:])
df_test3 = df[test_num:test_num *2]
#############training data 4 ####################
params4 = {
    'task': 'train','boosting_type': 'gbdt','objective': 'regression',
    'metric': {'l2', 'rmse'},'max_depth':6,'num_leaves':27,
    'min_data_in_leaf':300,'learning_rate': 0.02,
    'feature_fraction': 0.8,'bagging_fraction': 0.8,'bagging_freq': 5,
    'verbose': -1}
    
df_train4 = df[:test_num*2].append(df[test_num *3:])
df_test4 = df[test_num*2: test_num*3]

#############training data 5 ####################
params5 = {
    'task': 'train','boosting_type': 'gbdt','objective': 'regression',
    'metric': {'l2', 'rmse'},'max_dept':5,'num_leaves':21,
    'min_data_in_leaf':300,'learning_rate': 0.02,'num_iterations':3000,
    'feature_fraction': 0.9,'bagging_fraction': 0.9,'bagging_freq': 5,
    'verbose': -1}
    
df_train5 = df[:test_num*3].append(df[test_num *4:])
df_test5 = df[test_num*3: test_num*4]

############## predict #########################
start = time.time()
# pred1 = lgb_model(df_train1,df_test1,params1,feature_list,78)
# pred2 = lgb_model(df_train2,df_test2,params2,feature_list,9)
# pred3 = lgb_model(df_train3,df_test3,params3,feature_list,7)
# pred4 = lgb_model(df_train4,df_test4,params4,feature_list,7)
# pred5 = lgb_model(df_train5,df_test5,params5,feature_list,779)

# report(np.array(df_test1['label']), pred1)

# df_test['pred_loan'] = np.abs(y_pred)
# pickle.dump(df_test[['uid','label','pred_loan']],open('lgb7.pkl','wb'))

end = time.time()
print('train time:',(end-start)/60.)


##############submission#################
# X_test = make_test_set()
data_path = './data/test.pkl'
X_test = pickle.load(open(data_path,'rb'))
print(X_test.describe())
sub = X_test.copy()
# del X_test['uid']
X_test.columns = feature_list

gbm1 = pickle.load(open('./model/lgb_model773.pkl','rb'))
pred1 = gbm1.predict(X_test[feature_list], num_iteration=gbm1.best_iteration)
pred1 = np.where(pred1< 0. ,0,pred1)

# gbm2 = pickle.load(open('./model/lgb_model7895.pkl','rb'))
# pred2 = gbm2.predict(X_test[feature_list], num_iteration=gbm2.best_iteration)
# pred2 = np.where(pred2< 0. ,0,pred2)

gbm3 = pickle.load(open('./model/lgb_model778.pkl','rb'))
pred3 = gbm3.predict(X_test[feature_list], num_iteration=gbm3.best_iteration)
pred3 = np.where(pred3< 0. ,0,pred3)

# gbm4 = pickle.load(open('./model/lgb_model4.pkl','rb'))
# pred4 = gbm4.predict(X_test[feature_list], num_iteration=gbm4.best_iteration)

# gbm5 = pickle.load(open('./model/lgb_model5.pkl','rb'))
# pred5 = gbm5.predict(X_test[feature_list], num_iteration=gbm5.best_iteration)
# pred5 = np.where(pred5< 0. ,-pred5,pred5)

# # pred = np.array(pred)
# # pred = np.where(pred< 0. ,-pred,pred)

sub['pred1'] = pred1
# sub['pred2'] = pred2
sub['pred3'] = pred3
# sub['pred4'] = pred4
# sub['pred5'] = pred5

sub = sub[['uid','pred1','pred3']]
print(sub.describe())
sub['mean'] = sub[['pred1','pred3']].apply(lambda x : x.mean(),axis=1)

sub = sub[['uid','mean']]
sub.to_csv('./sub/sub13.csv',sep=',',header=None,index=False,encoding='utf8')
print(sub.describe())








