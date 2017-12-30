# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler  
import pandas as pd
import numpy as np
import pickle
import os
import math
from util import *

t_click_file = '../t_click.csv'
t_loan_sum_file = '../t_loan_sum.csv'
t_loan_file = '../t_loan.csv'
t_order_file = '../t_order.csv'
t_user_file = '../t_user.csv'

if  not os.path.exists('tmp'):
    os.mkdir('tmp')

def gen_user_feat():
    dump_path = './tmp/train_user_feat.pkl'
    if os.path.exists(dump_path):
        df_user = pickle.load(open(dump_path,'rb'))
    else:
        df_user = pd.read_csv(t_user_file,header=0)
        # 训练时的截止日期时11月，以11月初作为计算激活时长终止时间，预测时改为12月
        df_user['a_date'] = df_user['active_date'].map(lambda x: datetime.strptime('2016-11-1','%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
        # a_date 数据格式变成datetime格式，datetime.days , datetime.months, datetime.years,
        df_user['a_date'] = df_user['a_date'].map(lambda x : round(x.days/7))
        df_user['limit'] = df_user['limit'].map(lambda x: change_data(x))
        sex_df = pd.get_dummies(df_user['sex'], prefix="sex")
        df_user = pd.concat([df_user,sex_df],axis = 1)
        del df_user['active_date']
        del df_user['sex']
        pickle.dump(df_user, open(dump_path, 'wb'))
    return df_user

def gen_loan_feat():
    dump_path = './tmp/train_loan_feat.pkl'
    if os.path.exists(dump_path):
        df_loan = pickle.load(open(dump_path,'rb'))
    else:
        df_loan = pd.read_csv(t_loan_file,header=0)
        df_loan['month'] = df_loan['loan_time'].map(lambda x: conver_time(x))
        df_loan['loan_amount'] = df_loan['loan_amount'].map(lambda x: change_data(x))
        df_loan = df_loan[df_loan['month'] != 11]

        # 贷款时间分布特征，按比例，不要直接用次数
        loan_hour_df = df_loan.copy()
        loan_hour_df['loan_time_hours'] = loan_hour_df['loan_time'].map(lambda x: int(x.split(' ')[1].split(':')[0]))
        loan_hour_df['loan_time_hours'] = loan_hour_df['loan_time_hours'].map(lambda x : map_hours2bucket('loan',x))
        loan_hour_df = loan_hour_df.groupby(['uid','loan_time_hours'],as_index=False).count()
        loan_hour_df = loan_hour_df.pivot(index='uid', columns='loan_time_hours', values='loan_amount').reset_index()
        loan_hour_df = loan_hour_df.fillna(0)
        loan_hour_df['loan_sum_hour'] = loan_hour_df[['loan_hours_01','loan_hours_02','loan_hours_03','loan_hours_04',\
                                                    'loan_hours_05','loan_hours_05']].apply(lambda x: x.sum(),axis=1)
        loan_hour_df.loc[:,'loan_hours_01'] = loan_hour_df['loan_sum_hour']/loan_hour_df['loan_hours_01']
        loan_hour_df.loc[:,'loan_hours_02'] = loan_hour_df['loan_sum_hour']/loan_hour_df['loan_hours_02']
        loan_hour_df.loc[:,'loan_hours_03'] = loan_hour_df['loan_sum_hour']/loan_hour_df['loan_hours_03']
        loan_hour_df.loc[:,'loan_hours_04'] = loan_hour_df['loan_sum_hour']/loan_hour_df['loan_hours_04']
        loan_hour_df.loc[:,'loan_hours_05'] = loan_hour_df['loan_sum_hour']/loan_hour_df['loan_hours_05']
        loan_hour_df.loc[:,'loan_hours_06'] = loan_hour_df['loan_sum_hour']/loan_hour_df['loan_hours_06']
        del loan_hour_df['loan_sum_hour']

        # 3个月的贷款金额统计特征
        statistic_df = df_loan.copy()
        statistic_df = statistic_df.groupby(['uid','month'],as_index=False).sum()
        statistic_df = statistic_df.pivot(index='uid', columns='month', values='loan_amount').reset_index()
        statistic_df = statistic_df.fillna(0)
        statistic_df['loan_min'] = statistic_df[[8,9,10]].apply(lambda x: x.min(),axis=1)
        statistic_df['loan_max'] = statistic_df[[8,9,10]].apply(lambda x: x.max(),axis=1)
        # statistic_df['loan_sum'] = statistic_df[[8,9,10]].apply(lambda x: x.sum(),axis=1)
        statistic_df['loan_mean'] = statistic_df[[8,9,10]].apply(lambda x: x.mean(),axis=1)
        statistic_df['loan_median'] = statistic_df[[8,9,10]].apply(lambda x: x.median(),axis=1)
        statistic_df['loan_std'] = statistic_df[[8,9,10]].apply(lambda x: x.std(),axis=1)
        # statistic_df['loan_std'] = statistic_df['loan_std'].map(lambda x : 1/(1+x))

        # 每月贷款金额是否超过初始额度
        df_limit = gen_user_feat()[['uid','limit']]
        statistic_df = pd.merge(statistic_df,df_limit,how='left',on='uid')
        statistic_df['exceed_loan_1'] = statistic_df[8] - statistic_df['limit']
        statistic_df['exceed_loan_2'] = statistic_df[9] - statistic_df['limit']
        statistic_df['exceed_loan_3'] = statistic_df[10] - statistic_df['limit']
        def _map_num(x):
            if x >= 0:
                return 1
            else:
                return 0
        statistic_df['exceed_loan_1'] = statistic_df['exceed_loan'].map(lambda x : _map_num(x))
        statistic_df['exceed_loan_2'] = statistic_df['exceed_loan'].map(lambda x : _map_num(x))
        statistic_df['exceed_loan_3'] = statistic_df['exceed_loan'].map(lambda x : _map_num(x))

        # 贷款分期特征
        plannum_df = df_loan.copy()
        plannum_df = plannum_df.groupby(['uid','plannum'],as_index=False).count()
        plannum_df = plannum_df.pivot(index='uid',columns='plannum',values='loan_amount').reset_index()
        plannum_df = plannum_df.fillna(0)
        plannum_df.columns = ['uid','plannum_01','plannum_03','plannum_06','plannum_12']
        plannum_df['plannum_sum'] = plannum_df[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x: x.sum(),axis=1)
        plannum_df.loc[:,'plannum_01'] = plannum_df['plannum_sum']/plannum_df['plannum_01']
        plannum_df.loc[:,'plannum_03'] = plannum_df['plannum_sum']/plannum_df['plannum_03']
        plannum_df.loc[:,'plannum_06'] = plannum_df['plannum_sum']/plannum_df['plannum_06']
        plannum_df.loc[:,'plannum_12'] = plannum_df['plannum_sum']/plannum_df['plannum_12']

        # 最后一次贷款离11月的时长，平均每次贷款的时间间隔，

        # 每月贷款次数
        df_loan['loan_times'] = 1
        # 上次贷款距离现在的时长
        df_loan['loanTime_weights'] = df_loan['loan_time'].map(lambda x: datetime.strptime('2016-11-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_loan['loanTime_weights'] = df_loan['loanTime_weights'].map(lambda x: 1/(1+ round(x.days/7)))
        # 三个月贷款权重累计和
        df_loan['loan_weights'] = df_loan['loan_amount'] * df_loan['loanTime_weights']

        # 11月累计需要还款金额
        df_loan.loc[:,'repay'] = df_loan['loan_amount']/df_loan['plannum']
        df_loan.loc[:,'repay'][df_loan['plannum'] == 1][df_loan['month'] <= 9] = 0.
        df_loan = df_loan.groupby(['uid','month'],as_index=False).sum()

        # 是否有连续贷款情况
        month_df = pd.get_dummies(df_loan['month'], prefix="month")
        df_loan = pd.concat([df_loan,month_df],axis=1)
        df_loan = df_loan.groupby(['uid'],as_index=False).sum()
        df_loan['loan_times_months'] = df_loan['month_8']+df_loan['month_9']+df_loan['month_10']
        df_loan['loan_12'] = df_loan['month_8']+df_loan['month_9']
        df_loan['loan_12'] = df_loan['loan_12'].map({0:0,1:0,2:1})
        df_loan['loan_13'] = df_loan['month_8']+df_loan['month_10']
        df_loan['loan_13'] = df_loan['loan_13'].map({0:0,1:0,2:1})
        df_loan['loan_23'] = df_loan['month_9']+df_loan['month_10']
        df_loan['loan_23'] = df_loan['loan_23'].map({0:0,1:0,2:1})
        df_loan['loan_123'] = df_loan['month_8']+df_loan['month_9']+df_loan['month_10']
        df_loan['loan_123'] = df_loan['loan_123'].map({0:0,1:0,2:0,3:1})

        del df_loan['month']
        del df_loan['month_8']
        del df_loan['month_9']
        del df_loan['month_10']
        del df_loan['loanTime_weights']

        df_loan.loc[:,'per_plannum_loan'] = df_loan['loan_amount'] / df_loan['plannum']
        df_loan.loc[:,'per_times_loan'] = df_loan['loan_amount'] /df_loan['loan_times_months']

        df_loan = pd.merge(df_loan,statistic_df[['uid','loan_min','loan_mean','loan_median','loan_max','loan_sum','loan_std',\
                                                'exceed_loan_1','exceed_loan_2','exceed_loan_3']], how='outer',on='uid')
        df_loan = pd.merge(df_loan,plannum_df, how='left',on='uid')
        df_loan = pd.merge(df_loan,loan_hour_df, how='left',on='uid')
        # loan_cluster_label = user_cluster(df_loan,'train','loan')
        # df_loan['loan_cluster_label'] = loan_cluster_label
        pickle.dump(df_loan, open(dump_path, 'wb'))
    return df_loan


if __name__ == '__main__':
    df_user = gen_user_feat()
    print(df_user.head(10))


















