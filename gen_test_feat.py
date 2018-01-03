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
    dump_path = './tmp/test_user_feat.pkl'
    if os.path.exists(dump_path):
        df_user = pickle.load(open(dump_path,'rb'))
    else:
        df_user = pd.read_csv(t_user_file,header=0)
        # 训练时的截止日期时11月，以11月初作为计算激活时长终止时间，预测时改为12月
        df_user['a_date'] = df_user['active_date'].map(lambda x: datetime.strptime('2016-12-1','%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
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
    dump_path = './tmp/test_loan_feat.pkl'
    if os.path.exists(dump_path):
        df_loan = pickle.load(open(dump_path,'rb'))
    else:
        df_loan = pd.read_csv(t_loan_file,header=0)
        df_loan['month'] = df_loan['loan_time'].map(lambda x: conver_time(x))
        df_loan['loan_amount'] = df_loan['loan_amount'].map(lambda x: change_data(x))
        df_loan = df_loan[df_loan['month'] != 8]
        print("loan表的行列数：",df_loan.values.shape)

        # 贷款时间分布特征，按比例，不要直接用次数
        loan_hour_df = df_loan.copy()
        loan_hour_df['loan_time_hours'] = loan_hour_df['loan_time'].map(lambda x: int(x.split(' ')[1].split(':')[0]))
        loan_hour_df['loan_time_hours'] = loan_hour_df['loan_time_hours'].map(lambda x : map_hours2bucket('loan',x))
        loan_hour_df = loan_hour_df.groupby(['uid','loan_time_hours'],as_index=False).count()
        loan_hour_df = loan_hour_df.pivot(index='uid', columns='loan_time_hours', values='loan_amount').reset_index()
        loan_hour_df = loan_hour_df.fillna(0)
        loan_hour_df['loan_sum_hour'] = loan_hour_df[['loan_hours_01','loan_hours_02','loan_hours_03','loan_hours_04',\
                                                    'loan_hours_05','loan_hours_06']].apply(lambda x: x.sum(),axis=1)
        loan_hour_df.loc[:,'loan_hours_01'] = loan_hour_df['loan_hours_01']/loan_hour_df['loan_sum_hour']
        loan_hour_df.loc[:,'loan_hours_02'] = loan_hour_df['loan_hours_02']/loan_hour_df['loan_sum_hour']
        loan_hour_df.loc[:,'loan_hours_03'] = loan_hour_df['loan_hours_03']/loan_hour_df['loan_sum_hour']
        loan_hour_df.loc[:,'loan_hours_04'] = loan_hour_df['loan_hours_04']/loan_hour_df['loan_sum_hour']
        loan_hour_df.loc[:,'loan_hours_05'] = loan_hour_df['loan_hours_05']/loan_hour_df['loan_sum_hour']
        loan_hour_df.loc[:,'loan_hours_06'] = loan_hour_df['loan_hours_06']/loan_hour_df['loan_sum_hour']
        del loan_hour_df['loan_sum_hour']
        print("loan_hour_df表的行列数：",loan_hour_df.values.shape)

        # 3个月的贷款金额统计特征
        statistic_df = df_loan.copy()
        statistic_df = statistic_df.groupby(['uid','month'],as_index=False).sum().reset_index()
        stat_feat = ['min','mean','max','std','median']
        statistic_df = statistic_df.groupby('uid')['loan_amount'].agg(stat_feat).reset_index()
        statistic_df.columns = ['uid'] + ['loan_' + col for col in stat_feat]

        # 贷款分期数特征
        plannum_df = df_loan.copy()
        plannum_df = plannum_df.groupby(['uid','plannum'],as_index=False).count()
        plannum_df = plannum_df.pivot(index='uid',columns='plannum',values='loan_amount').reset_index()
        plannum_df = plannum_df.fillna(0)
        plannum_df.columns = ['uid','plannum_01','plannum_03','plannum_06','plannum_12']
        plannum_df['plannum_sum'] = plannum_df[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x: x.sum(),axis=1)
        plannum_df.loc[:,'plannum_01'] = plannum_df['plannum_01']/plannum_df['plannum_sum']
        plannum_df.loc[:,'plannum_03'] = plannum_df['plannum_03']/plannum_df['plannum_sum']
        plannum_df.loc[:,'plannum_06'] = plannum_df['plannum_06']/plannum_df['plannum_sum']
        plannum_df.loc[:,'plannum_12'] = plannum_df['plannum_12']/plannum_df['plannum_sum']
        print("plannum_df表的行列数：",plannum_df.values.shape)


        # 最后一次贷款离11月的时长，平均每次贷款的时间间隔，最后一次贷款离11月的时长是否超过平均贷款时间间隔
        time_df = df_loan.copy()
        time_df = time_df.sort_values(by=['uid','loan_time'])
        time_df['loan_time'] = time_df['loan_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        time_df['last_loan_time'] = time_df['loan_time'].map(lambda x: round(x.days/7))
        last_loan_time = time_df.copy()
        last_loan_time.drop_duplicates(subset='uid', keep='last', inplace=True)
        last_loan_time = last_loan_time[['uid','last_loan_time']]
        last_loan_time['per_loan_time_interval'] = 13.
        for idx in list(last_loan_time.index):
            uid = last_loan_time.loc[idx,'uid'] #.loc[]中括号里面的第一个参数只能是索引，不能是列名
            interval = np.array(time_df[time_df['uid']==uid]['last_loan_time'])  #将Serise格式转换成numpy ndarray元祖
            if len(interval) == 1:
                per_loan_time_interval = 13. - interval[0]
            else:
                per_loan_time_interval = (interval.max() - interval.min())/(len(interval)-1)
            last_loan_time.loc[idx,'per_loan_time_interval'] = per_loan_time_interval
        last_loan_time['is_exceed_loan_interval'] = last_loan_time['last_loan_time'] - last_loan_time['per_loan_time_interval']
        print("last_loan_time表的行列数：",last_loan_time.values.shape)

        # 每月贷款金额是否超过初始额度
        exceed_df = df_loan.copy()
        exceed_df = exceed_df.groupby(['uid','month'],as_index=False).sum().reset_index()
        exceed_df = exceed_df.pivot(index='uid', columns='month', values='loan_amount').reset_index()
        exceed_df = exceed_df.fillna(0)
        
        df_limit = gen_user_feat()[['uid','limit']]
        exceed_df = pd.merge(exceed_df,df_limit,how='left',on='uid')
        exceed_df['exceed_loan_1'] = exceed_df[8] - exceed_df['limit']
        exceed_df['exceed_loan_2'] = exceed_df[9] - exceed_df['limit']
        exceed_df['exceed_loan_3'] = exceed_df[10] - exceed_df['limit']
        print("exceed_df表的行列数：",exceed_df.values.shape)

        # 更新limit,如果某次贷款的额度超过了该用户的初始贷款额度，那么说明该用户的额度已经提高了，且提高的额度至少到该次贷款的金额
        new_limit = df_loan.copy()
        new_limit = new_limit.groupby('uid')['loan_amount'].agg('max').reset_index()
        # print(new_limit.head())
        new_limit.columns = ['uid','max']
        new_limit = pd.merge(new_limit,df_limit,how='left',on='uid')
        limit = []
        for i in range(len(new_limit['limit'])):
            if list(new_limit['limit'])[i] > list(new_limit['max'])[i]:
                limit.append(list(new_limit['limit'])[i])
            else:
                limit.append(list(new_limit['max'])[i])
        new_limit['new_limit'] = limit
        # new_limit.apply(lambda x: x["max"] if x["limit"] < x["max"] else x["limit"], axis=1)

        # 首次贷款时，贷款时间和注册时间的间隔
        df_active = gen_user_feat()[['uid','a_date']]
        time_df.drop_duplicates(subset='uid', keep='first', inplace=True)
        first_loan_df = pd.merge(time_df,df_active,how='left',on='uid')
        first_loan_df['active_loan_interval'] = first_loan_df['a_date'] - first_loan_df['last_loan_time']
        first_loan_df = first_loan_df[['uid','active_loan_interval']]

        # 贷款次数
        df_loan['loan_times'] = 1
        # 每次贷款距离现在的时长
        df_loan['loanTime_weights'] = df_loan['loan_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_loan['loanTime_weights'] = df_loan['loanTime_weights'].map(lambda x: 1/(1+ round(x.days/7)))
        # 三个月贷款权重累计和
        df_loan['loan_weights'] = df_loan['loan_amount'] * df_loan['loanTime_weights']

        # 11月累计需要还款金额
        df_loan.loc[:,'repay'] = df_loan['loan_amount']/df_loan['plannum']
        for idx in list(df_loan.index):
            if df_loan.loc[idx,'plannum'] == 1 and df_loan.loc[idx,'month'] <= 10:
                df_loan.loc[idx,'repay'] = 0
        
        df_loan = df_loan.groupby(['uid','month'],as_index=False).sum()

        # 是否有连续贷款情况
        month_df = pd.get_dummies(df_loan['month'], prefix="month")
        df_loan = pd.concat([df_loan,month_df],axis=1)
        df_loan = df_loan.groupby(['uid'],as_index=False).sum()
        df_loan['loan_times_months'] = df_loan['month_11']+df_loan['month_9']+df_loan['month_10']
        df_loan['loan_12'] = df_loan['month_10']+df_loan['month_9']
        df_loan['loan_12'] = df_loan['loan_12'].map({0:0,1:0,2:1})
        df_loan['loan_13'] = df_loan['month_11']+df_loan['month_9']
        df_loan['loan_13'] = df_loan['loan_13'].map({0:0,1:0,2:1})
        df_loan['loan_23'] = df_loan['month_11']+df_loan['month_10']
        df_loan['loan_23'] = df_loan['loan_23'].map({0:0,1:0,2:1})
        df_loan['loan_123'] = df_loan['month_11']+df_loan['month_9']+df_loan['month_10']
        df_loan['loan_123'] = df_loan['loan_123'].map({0:0,1:0,2:0,3:1})

        del df_loan['month']
        del df_loan['month_11']
        del df_loan['month_9']
        del df_loan['month_10']
        del df_loan['loanTime_weights']


        df_loan.loc[:,'per_plannum_loan'] = df_loan['loan_amount'] / df_loan['plannum']
        df_loan.loc[:,'per_times_loan'] = df_loan['loan_amount'] /df_loan['loan_times_months']
        print("loan表的行列数：",df_loan.values.shape)

        df_loan = pd.merge(df_loan,statistic_df[['uid','loan_min','loan_mean','loan_median','loan_max','loan_std',\
                                                ]], how='outer',on='uid')
        df_loan = pd.merge(df_loan,plannum_df, how='left',on='uid')
        df_loan = pd.merge(df_loan,loan_hour_df, how='left',on='uid')
        df_loan = pd.merge(df_loan,last_loan_time, how='left',on='uid')
        df_loan = pd.merge(df_loan,first_loan_df, how='left',on='uid')
        df_loan = pd.merge(df_loan,exceed_df[['uid','exceed_loan_1','exceed_loan_2','exceed_loan_3']], how='left',on='uid')
        df_loan = pd.merge(df_loan,new_limit[['uid','new_limit']], how='left',on='uid')
        pickle.dump(df_loan, open(dump_path, 'wb'))
    return df_loan


def gen_filter_loan_feat():
    dump_path = './tmp/test_filter_loan_feat.pkl'
    if os.path.exists(dump_path):
        df_filter_loan = pickle.load(open(dump_path,'rb'))
    else:
        df_filter_loan = pd.read_csv(t_loan_file,header=0)
        df_filter_loan['month'] = df_filter_loan['loan_time'].map(lambda x: conver_time(x))
        df_filter_loan['loan_amount'] = df_filter_loan['loan_amount'].map(lambda x: round(change_data(x)))
        df_filter_loan = df_filter_loan[df_filter_loan['month'] != 8]
        del df_filter_loan['month']

        # 贷款行为在滑动时间窗口内的贷款统计特征
        df_filter_loan['days'] = df_filter_loan['loan_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_filter_loan['days'] = df_filter_loan['days'].map(lambda x: int(x.days))

        uid = df_filter_loan['uid'].unique()
        exclu = [1]*len(uid) 
        days_df = pd.DataFrame({'uid':uid,'exclu':exclu})
        day_list = [0,3,7,14,21,28,35,42,49,56,63,70,77,84]
        for i in range(len(day_list)-1):
            days1 = day_list[i]
            days2 = day_list[i+1]
            df = df_filter_loan[['uid','days','loan_amount']].copy()
            day_df = get_stat_feat(df,'loan_amount', 'loan', days1,days2)
            days_df = pd.merge(days_df,day_df,how='left',on='uid')

        days_df = days_df.fillna(0.)
        del days_df['exclu']

        df_filter_loan = days_df
        pickle.dump(df_filter_loan, open(dump_path, 'wb'))
    return df_filter_loan


def gen_click_feat():
    dump_path = './tmp/test_click_feat.pkl'
    if os.path.exists(dump_path):
        df_click = pickle.load(open(dump_path,'rb'))
    else:
        df_click = pd.read_csv(t_click_file,header=0)
        df_click['month'] = df_click['click_time'].map(lambda x: conver_time(x))
        df_click = df_click[df_click['month'] != 8]

        # 点击时间特征分布
        click_hour_df = df_click.copy()
        click_hour_df['click_time_hours'] = click_hour_df['click_time'].map(lambda x: int(x.split(' ')[1].split(':')[0]))
        click_hour_df['click_time_hours'] = click_hour_df['click_time_hours'].map(lambda x : map_hours2bucket('click',x))
        click_hour_df = click_hour_df.groupby(['uid','click_time_hours'],as_index=False).count()
        click_hour_df = click_hour_df.pivot(index='uid', columns='click_time_hours', values='click_time').reset_index()
        click_hour_df = click_hour_df.fillna(0)
        click_hour_df['click_sum_hour'] = click_hour_df[['click_hours_01','click_hours_02','click_hours_03','click_hours_04',\
                                                    'click_hours_05','click_hours_06']].apply(lambda x: x.sum(),axis=1)
        click_hour_df.loc[:,'click_hours_01'] = click_hour_df['click_hours_01']/click_hour_df['click_sum_hour']
        click_hour_df.loc[:,'click_hours_02'] = click_hour_df['click_hours_02']/click_hour_df['click_sum_hour']
        click_hour_df.loc[:,'click_hours_03'] = click_hour_df['click_hours_03']/click_hour_df['click_sum_hour']
        click_hour_df.loc[:,'click_hours_04'] = click_hour_df['click_hours_04']/click_hour_df['click_sum_hour']
        click_hour_df.loc[:,'click_hours_05'] = click_hour_df['click_hours_05']/click_hour_df['click_sum_hour']
        click_hour_df.loc[:,'click_hours_06'] = click_hour_df['click_hours_06']/click_hour_df['click_sum_hour']
        del click_hour_df['click_sum_hour']

        df_click['click'] = 1

        # 三个月内的统计特征
        statistic_df = df_click.copy()
        statistic_df = statistic_df.groupby(['uid','month'],as_index=False).sum().reset_index()
        stat_feat = ['min','mean','max','std','median','count','sum']
        statistic_df = statistic_df.groupby('uid')['click'].agg(stat_feat).reset_index()
        statistic_df.columns = ['uid'] + ['click_' + col for col in stat_feat]

        # 最后一次点击离11月的时长，平均每次点击的时间间隔，最后一次点击离11月的时长是否超过平均点击时间间隔
        time_df = df_click.copy()
        time_df = time_df.sort_values(by=['uid','click_time'])
        time_df['click_time'] = time_df['click_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        time_df['last_click_time'] = time_df['click_time'].map(lambda x: round(x.days/7))
        last_click_time = time_df.copy()
        last_click_time.drop_duplicates(subset='uid', keep='last', inplace=True)
        last_click_time = last_click_time[['uid','last_click_time']]
        last_click_time['per_click_time_interval'] = 13.

        time_df = time_df.groupby(['uid','last_click_time'],as_index=False).sum()
        for idx in list(last_click_time.index):
            uid = last_click_time.loc[idx,'uid'] #.loc[]中括号里面的第一个参数只能是索引，不能是列名
            interval = np.array(time_df[time_df['uid']==uid]['last_click_time'])  #将Serise格式转换成numpy ndarray元祖
            if len(interval) == 1:
                per_click_time_interval = 13. - interval[0]
            else:
                per_click_time_interval = (interval.max() - interval.min())/(len(interval)-1)
            last_click_time.loc[idx,'per_click_time_interval'] = per_click_time_interval
        last_click_time['is_exceed_click_interval'] = last_click_time['last_click_time'] - last_click_time['per_click_time_interval']
        print("last_click_time表的行列数：",last_click_time.values.shape)

        
        df_click['click_weights'] = df_click['click_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_click['click_weights'] = df_click['click_weights'].map(lambda x: 1/(1+round(x.days/7)))
        del df_click['click_time']

        pid_df = pd.get_dummies(df_click["pid"], prefix="pid")
        param_df = pd.get_dummies(df_click["param"], prefix="param")

        del df_click['pid']
        del df_click['param']
        df_click = pd.concat([df_click,pid_df,param_df],axis=1)

        df_click = df_click.groupby(['uid'],as_index=False).sum()

        pid_feat = list(pid_df.columns) #[ i for i in range(1,51)]
        param_feat = list(param_df.columns)
        column_list = pid_feat + param_feat
        for feat in column_list:
            df_click.loc[:,feat] = df_click[feat]/(1+df_click['click'])

        del df_click['month']
        del df_click['click']
        
        df_click = pd.merge(df_click,statistic_df, how='left',on='uid')
        df_click = pd.merge(df_click,click_hour_df,how='left',on='uid')
        df_click = pd.merge(df_click,last_click_time, how='left',on='uid')

        pickle.dump(df_click, open(dump_path, 'wb'))
    return df_click


def gen_order_feat():
    dump_path = './tmp/test_order_feat.pkl'
    if os.path.exists(dump_path):
        df_order = pickle.load(open(dump_path,'rb'))
    else:
        df_order = pd.read_csv(t_order_file,header=0)
        df_order['month'] = df_order['buy_time'].map(lambda x: conver_time(x))
        df_order = df_order[df_order['month']!=8]
        df_order['order_times'] = 1
        df_order.fillna(0.,inplace=True)

        df_order['price'] = df_order['price'].map(lambda x: change_data(x))
        df_order['discount'] = df_order['discount'].map(lambda x: change_data(x))
        # 购买商品实际支付价格
        df_order['real_price'] = df_order['price']*df_order['qty'] - df_order['discount']
        df_order.loc[df_order['real_price']<0,'real_price'] = 0.

        # 三个月的消费金额统计特征
        statistic_df = df_order.copy()
        statistic_df = statistic_df.groupby(['uid','month'],as_index=False).sum()
        stat_feat = ['min','mean','max','std','median','count','sum']
        statistic_df = statistic_df.groupby('uid')['real_price'].agg(stat_feat).reset_index()
        statistic_df.columns = ['uid'] + ['price_' + col for col in stat_feat]

        # 最后一次购买离11月的时长，平均每次购买的时间间隔，最后一次购买离11月的时长是否超过平均购买时间间隔
        time_df = df_order.copy()
        time_df = time_df.sort_values(by=['uid','buy_time'])
        time_df['buy_time'] = time_df['buy_time'].map(lambda x: datetime.strptime('2016-12-1', '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
        time_df['last_order_time'] = time_df['buy_time'].map(lambda x: round(x.days/7))
        last_order_time = time_df.copy()
        last_order_time.drop_duplicates(subset='uid', keep='last', inplace=True)
        last_order_time = last_order_time[['uid','last_order_time']]
        last_order_time['per_order_time_interval'] = 13.

        time_df = time_df.groupby(['uid','last_order_time'],as_index=False).sum()
        for idx in list(last_order_time.index):
            uid = last_order_time.loc[idx,'uid'] 
            interval = np.array(time_df[time_df['uid']==uid]['last_order_time']) 
            if len(interval) == 1:
                per_order_time_interval = 13. - interval[0]
            else:
                per_order_time_interval = (interval.max() - interval.min())/(len(interval)-1)
            last_order_time.loc[idx,'per_order_time_interval'] = per_order_time_interval
        last_order_time['is_exceed_order_interval'] = last_order_time['last_order_time'] - last_order_time['per_order_time_interval']
        print("last_order_time表的行列数：",last_order_time.values.shape)

        df_order['buy_weights'] = df_order['buy_time'].map(lambda x: datetime.strptime('2016-12-1', '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
        df_order['buy_weights'] = df_order['buy_weights'].map(lambda x: 1/(1+round(x.days/7)))
        df_order['cost_weight'] = df_order['real_price'] * df_order['buy_weights']
        df_order = df_order[['uid','buy_weights','cost_weight','real_price','discount']]
        df_order = df_order.groupby(['uid'],as_index=False).sum()
        # 购买商品的折扣率
        df_order.loc[:,'dis_ratio'] = df_order['discount'] / (df_order['discount'] + df_order['real_price'])
        del df_order['discount']

        df_order = pd.merge(df_order,statistic_df, how='left',on='uid')
        df_order = pd.merge(df_order,last_order_time, how='left',on='uid')
        df_order = df_order.fillna(0)

        pickle.dump(df_order, open(dump_path, 'wb'))
    return df_order


def make_test_data():
    dump_path = './data/test.pkl'
    if os.path.exists(dump_path):
        test_set = pickle.load(open(dump_path,'rb'))
    else:
        df_user = gen_user_feat()
        df_loan = gen_loan_feat()
        df_filter_loan = gen_filter_loan_feat()
        df_click = gen_click_feat()
        df_order = gen_order_feat()

        test_set = pd.merge(df_user, df_loan, how='outer', on='uid')
        test_set = pd.merge(test_set, df_filter_loan, how='outer', on='uid')
        test_set = pd.merge(test_set, df_click, how='outer', on='uid')
        test_set = pd.merge(test_set, df_order, how='outer', on='uid')
        
        # test_set = test_set.fillna()

        pickle.dump(test_set, open(dump_path, 'wb'))
        feat_id = {i:fea for i ,fea in enumerate(list(test_set.columns))}
        print(feat_id)

    return test_set

if __name__ == '__main__':
    test_data = make_test_data()
    print(test_data.head(10))
    print("test_data表的行列数：",test_data.values.shape)



















