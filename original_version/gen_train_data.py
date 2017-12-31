# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler  
import pandas as pd
import pickle
import os
import math
import numpy as np

t_click_file = '../../t_click.csv'
t_loan_sum_file = '../../t_loan_sum.csv'
t_loan_file = '../../t_loan.csv'
t_order_file = '../../t_order.csv'
t_user_file = '../../t_user.csv'

def change_data(x):
    return math.pow(5,x)-1

def conver_time(time):
    return int(time.split('-')[1])

def map_day2period(df,values,action,days):
    def _exch(x):
        if x <= days:
            return '%s_%s' % (action, days)
        else:
            return 'exclude'
    # print(df['days'][:10])
    df['days'] = df['days'].map(lambda x:_exch(x))
    df = df.groupby(['uid','days'],as_index=False).sum()
    df = df.pivot(index='uid', columns='days', values=values).reset_index()
    df = df[['uid','%s_%s' % (action, days)]]
    # del df['uid']
    return df

def map_day2week(df,values,action):
    def _exch(x):
        if x <= 7:
            return '%s_week1' % action
        elif x <= 14:
            return '%s_week2' % action
        elif x <= 21:
            return '%s_week3' % action
        elif x <= 28:
            return '%s_week4' % action
        else:
            return 'other'
    df['weeks'] = df['days'].map(lambda x:_exch(x))
    df = df.groupby(['uid','weeks'],as_index=False).sum()
    df = df.pivot(index='uid', columns='weeks', values=values).reset_index()
    df = df[['uid','%s_week1' % action,'%s_week2' % action,'%s_week3' % action,'%s_week4' % action]]
    df.fillna(0,inplace=True)
    return df


def map_hours2bucket(action,hours):
    if hours>=8 and hours<=11:
        return '%s_hours_01' % action
    if hours >=12 and hours<=15:
        return '%s_hours_02' % action
    if hours >= 16 and hours<=19:
        return '%s_hours_03' % action
    if hours>=20 and hours<=23:
        return '%s_hours_04' % action
    if hours>=4 and hours <= 7:
        return '%s_hours_05' % action
    else:
        return '%s_hours_06' % action

def user_cluster(df,state,model_name):
    """
    使用余弦相似度/kNN计算用户的相似度
    """
    feat_list = list(df.columns)
    feat_list.remove('uid')
    data = df[feat_list].values
    data = StandardScaler().fit_transform(data)

    if state == 'train':
        clf = KMeans(init='k-means++', max_iter=1200,n_clusters=180, n_init=10, n_jobs= -2)
        clf.fit(data)
        joblib.dump(clf , './model/clu_%s.pkl' % model_name)
        cluster_label = clf.labels_ + 1
    else:
        clf = joblib.load('./model/clu_%s.pkl' % model_name)
        cluster_label = clf.predict(data) + 1
    return cluster_label

def gen_basic_user_feat():
    dump_path = './tmp/train_user_feat.pkl'
    if os.path.exists(dump_path):
        df_user = pickle.load(open(dump_path,'rb'))
    else:
        df_user = pd.read_csv(t_user_file,header=0)
        # 训练时的截止日期时11月，以11月初作为计算激活时长终止时间，预测时改为12月
        df_user['a_date'] = df_user['active_date'].map(lambda x: datetime.strptime('2016-11-1','%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
        df_user['a_date'] = df_user['a_date'].map(lambda x : x.days/7)
        df_user['limit'] = df_user['limit'].map(lambda x: change_data(x))
        del df_user['active_date']
        pickle.dump(df_user, open(dump_path, 'wb'))
    return df_user

def gen_filter_loan_feat():
    dump_path = './tmp/train_filter_loan_feat.pkl'
    if os.path.exists(dump_path):
        df_filter_loan = pickle.load(open(dump_path,'rb'))
    else:
        df_filter_loan = pd.read_csv(t_loan_file,header=0)
        df_filter_loan['month'] = df_filter_loan['loan_time'].map(lambda x: conver_time(x))
        df_filter_loan['loan_amount'] = df_filter_loan['loan_amount'].map(lambda x: round(change_data(x)))
        df_filter_loan = df_filter_loan[df_filter_loan['month'] != 11]
        del df_filter_loan['month']
        # df_filter_loan = df_filter_loan[df_filter_loan['loan_amount'] <= 90000]
        # df_filter_loan = df_filter_loan[df_filter_loan['loan_amount'] > 499]
        # 贷款行为在滑动时间窗口内的贷款总额
        df_filter_loan['days'] = df_filter_loan['loan_time'].map(lambda x: datetime.strptime('2016-11-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_filter_loan['days'] = df_filter_loan['days'].map(lambda x: int(x.days))

        uid = df_filter_loan['uid'].unique()
        exclu = [1]*len(uid) 
        days_df = pd.DataFrame({'uid':uid,'exclu':exclu})
        for day in [1,2,3,5,7,9,15,20,25,30,35,40,45,50,60,70,80,90]:
            df = df_filter_loan[['uid','days','loan_amount']].copy()
            day_df = map_day2period(df,'loan_amount', 'loan', day)
            days_df = pd.merge(days_df,day_df,how='left',on='uid')
        days_df = days_df.fillna(0)
        del days_df['exclu']
        change_list = list(days_df.columns)
        change_list.remove('uid')
        for col in change_list:
            days_df[col] = days_df[col].map(lambda x : math.log(x+1,5))

        weeks_df = map_day2week(df_filter_loan.copy(),'loan_amount','loan_filter')
        weeks_df['filter_loan_min'] = weeks_df[['loan_filter_week1','loan_filter_week2','loan_filter_week3','loan_filter_week4']].apply(lambda x: x.min(),axis=1)
        weeks_df['filter_loan_max'] = weeks_df[['loan_filter_week1','loan_filter_week2','loan_filter_week3','loan_filter_week4']].apply(lambda x: x.max(),axis=1)
        weeks_df['filter_loan_sum'] = weeks_df[['loan_filter_week1','loan_filter_week2','loan_filter_week3','loan_filter_week4']].apply(lambda x: x.sum(),axis=1)
        weeks_df['filter_loan_mean'] = weeks_df[['loan_filter_week1','loan_filter_week2','loan_filter_week3','loan_filter_week4']].apply(lambda x: x.mean(),axis=1)
        weeks_df['filter_loan_std'] = weeks_df[['loan_filter_week1','loan_filter_week2','loan_filter_week3','loan_filter_week4']].apply(lambda x: x.std(),axis=1)
        weeks_df['filter_loan_std'] = weeks_df['filter_loan_std'].map(lambda x : 1/(1+x))

        df_filter_loan = pd.merge(days_df,weeks_df,how='outer',on='uid')
        pickle.dump(df_filter_loan, open(dump_path, 'wb'))
    return df_filter_loan

def gen_basic_loan_feat():
    dump_path = './tmp/train_loan_feat.pkl'
    if os.path.exists(dump_path):
        df_loan = pickle.load(open(dump_path,'rb'))
    else:
        df_loan = pd.read_csv(t_loan_file,header=0)
        df_loan['month'] = df_loan['loan_time'].map(lambda x: conver_time(x))
        df_loan['loan_amount'] = df_loan['loan_amount'].map(lambda x: change_data(x))
        df_loan = df_loan[df_loan['month'] != 11]
        # df_loan = df_loan[df_loan['loan_amount'] <= 90000]
        # df_loan = df_loan[df_loan['loan_amount'] >= 499]

        # 贷款时间分布
        loan_hour_df = df_loan.copy()
        loan_hour_df['loan_time_hours'] = loan_hour_df['loan_time'].map(lambda x: int(x.split(' ')[1].split(':')[0]))
        loan_hour_df['loan_time_hours'] = loan_hour_df['loan_time_hours'].map(lambda x : map_hours2bucket('loan',x))
        loan_hour_df = loan_hour_df.groupby(['uid','loan_time_hours'],as_index=False).count()
        loan_hour_df = loan_hour_df.pivot(index='uid', columns='loan_time_hours', values='loan_amount').reset_index()
        loan_hour_df = loan_hour_df.fillna(0)

        statistic_df = df_loan.copy()
        statistic_df = statistic_df.groupby(['uid','month'],as_index=False).sum()
        statistic_df = statistic_df.pivot(index='uid', columns='month', values='loan_amount').reset_index()
        statistic_df = statistic_df.fillna(0)
        statistic_df['loan_min'] = statistic_df[[8,9,10]].apply(lambda x: x.min(),axis=1)
        statistic_df['loan_max'] = statistic_df[[8,9,10]].apply(lambda x: x.max(),axis=1)
        statistic_df['loan_sum'] = statistic_df[[8,9,10]].apply(lambda x: x.sum(),axis=1)
        statistic_df['loan_mean'] = statistic_df[[8,9,10]].apply(lambda x: x.mean(),axis=1)
        statistic_df['loan_std'] = statistic_df[[8,9,10]].apply(lambda x: x.std(),axis=1)
        statistic_df['loan_std'] = statistic_df['loan_std'].map(lambda x : 1/(1+x))

        # statistic_data = []
        # for uid in set(df_loan['uid']):
        #     _min = df_loan[df_loan['uid'] == uid]['loan_amount'].min()
        #     _max = df_loan[df_loan['uid'] == uid]['loan_amount'].max()
        #     _median = df_loan[df_loan['uid'] == uid]['loan_amount'].median()
        #     _sum = df_loan[df_loan['uid'] == uid]['loan_amount'].sum()
        #     statistic_data.append([uid,_min,_max,_median,_sum])
        # statistic_df = pd.DataFrame(statistic_data,columns=['uid','loan_min','loan_max','loan_median','loan_sum'])

        # 贷款分期特征
        plannum_df = df_loan.copy()
        plannum_df = plannum_df.groupby(['uid','plannum'],as_index=False).count()
        plannum_df = plannum_df.pivot(index='uid',columns='plannum',values='loan_amount').reset_index()
        plannum_df = plannum_df.fillna(0)
        plannum_df.columns = ['uid','plannum_01','plannum_03','plannum_06','plannum_12']
        plannum_df['plannum_min'] = plannum_df[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x: x.min(),axis=1)
        plannum_df['plannum_max'] = plannum_df[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x: x.max(),axis=1)
        plannum_df['plannum_sum'] = plannum_df[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x: x.sum(),axis=1)
        plannum_df['plannum_mean'] = plannum_df[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x: x.mean(),axis=1)
        plannum_df['plannum_std'] = plannum_df[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x: x.std(),axis=1)
        plannum_df['plannum_std'] = plannum_df['plannum_std'].map(lambda x : 1/(1+x))
        # 每月贷款次数
        # df_loan['loan_times'] = 1
        # 上次贷款距离现在的时长
        df_loan['loanTime_weights'] = df_loan['loan_time'].map(lambda x: datetime.strptime('2016-11-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_loan['loanTime_weights'] = df_loan['loanTime_weights'].map(lambda x: 1/(1e-6+x.days/30))
        # 贷款权重
        df_loan['loan_weights'] = df_loan['loan_amount'] * df_loan['loanTime_weights']
        # 11月累计需要还款金额
        month_8 = df_loan[df_loan['month']==8]
        month_8['repay'] = month_8['loan_amount']/month_8['plannum']
        month_8['repay'][month_8['plannum'] < 3] = 0.

        month_9 = df_loan[df_loan['month']==9]
        month_9['repay'] = month_9['loan_amount']/month_9['plannum']
        month_9['repay'][month_9['plannum'] < 2] = 0.

        month_10 = df_loan[df_loan['month']==10]
        month_10['repay'] = month_10['loan_amount']/month_10['plannum']

        df_loan = pd.concat([month_8,month_9,month_10],axis=0,ignore_index=False)
        # 每月贷款间隔特征
        month_df = pd.get_dummies(df_loan['month'], prefix="month")
        df_loan = pd.concat([df_loan,month_df],axis=1)
        df_loan = df_loan.groupby(['uid'],as_index=False).sum()
        df_loan['loan_months'] = df_loan['month_8']+df_loan['month_9']+df_loan['month_10']
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

        df_loan['per_plannum'] = df_loan['plannum'] / df_loan['loan_months']
        df_loan['per_times_loan'] = df_loan['loan_amount'] /df_loan['loan_months']

        # 每月贷款金额是否超过初始额度
        df_limit = gen_basic_user_feat()[['uid','limit']]
        df_loan = pd.merge(df_loan,df_limit,how='left',on='uid')
        df_loan['exceed_loan'] = df_loan['per_times_loan'] - df_loan['limit']
        def _map_num(x):
            if x >= 0:
                return 1
            else:
                return 0
        df_loan['exceed_loan'] = df_loan['exceed_loan'].map(lambda x : _map_num(x))
        del df_loan['limit']
        # df_loan['per_month_loan'] = df_loan['loan_amount'] /3

        df_loan = pd.merge(df_loan,statistic_df[['uid','loan_min','loan_mean','loan_max','loan_sum','loan_std']], how='outer',on='uid')
        df_loan = pd.merge(df_loan,plannum_df, how='left',on='uid')
        df_loan = pd.merge(df_loan,loan_hour_df, how='left',on='uid')
        df_loan = df_loan.fillna(0)
        loan_cluster_label = user_cluster(df_loan,'train','loan')
        df_loan['loan_cluster_label'] = loan_cluster_label
        pickle.dump(df_loan, open(dump_path, 'wb'))
    return df_loan

def gen_basic_order_feat():
    dump_path = './tmp/train_order_feat.pkl'
    if os.path.exists(dump_path):
        df_order = pickle.load(open(dump_path,'rb'))
    else:
        df_order = pd.read_csv(t_order_file,header=0)
        df_order['month'] = df_order['buy_time'].map(lambda x: conver_time(x))
        df_order = df_order[df_order['month']!=11]
        df_order['price'] = df_order['price'].map(lambda x: change_data(x))
        df_order['discount'] = df_order['discount'].map(lambda x: change_data(x))
        # 购买商品实际支付价格
        df_order['real_price'] = df_order['price']*df_order['qty'] - df_order['discount']
        df_order['real_price'][df_order['real_price']<0] = 0.

        statistic_df = df_order.copy()
        statistic_df = statistic_df.groupby(['uid','month'],as_index=False).sum()
        statistic_df = statistic_df.pivot(index='uid', columns='month', values='real_price').reset_index()
        statistic_df = statistic_df.fillna(0)
        statistic_df['buy_min'] = statistic_df[[8,9,10]].apply(lambda x: x.min(),axis=1)
        statistic_df['buy_max'] = statistic_df[[8,9,10]].apply(lambda x: x.max(),axis=1)
        statistic_df['buy_sum'] = statistic_df[[8,9,10]].apply(lambda x: x.sum(),axis=1)
        statistic_df['buy_mean'] = statistic_df[[8,9,10]].apply(lambda x: x.mean(),axis=1)
        statistic_df['buy_std'] = statistic_df[[8,9,10]].apply(lambda x: x.std(),axis=1)
        statistic_df['buy_std'] = statistic_df['buy_std'].map(lambda x : 1/(1+x))

        df_order['buy_weights'] = df_order['buy_time'].map(lambda x: datetime.strptime('2016-11-1', '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
        df_order['buy_weights'] = df_order['buy_weights'].map(lambda x: 1/(1e-6+x.days))
        df_order['cost_weight'] = df_order['real_price'] * df_order['buy_weights']
        df_order = df_order[['uid','buy_weights','cost_weight','real_price','discount']]
        df_order = df_order.groupby(['uid'],as_index=False).sum()
        # 购买商品的折扣率
        df_order['dis_ratio'] = df_order['discount'] / (df_order['discount'] + df_order['real_price'])
        del df_order['discount']

        df_order = pd.merge(df_order,statistic_df[['uid','buy_min','buy_mean','buy_max','buy_sum','buy_std']], how='left',on='uid')
        df_order = df_order.fillna(0)

        order_cluster_label = user_cluster(df_order,'train','order')
        df_order['order_cluster_label'] = order_cluster_label

        pickle.dump(df_order, open(dump_path, 'wb'))
    return df_order

def gen_filter_order_feat():
    dump_path = './tmp/train_filter_order_feat.pkl'
    if os.path.exists(dump_path):
        df_filter_order = pickle.load(open(dump_path,'rb'))
    else:
        df_filter_order = pd.read_csv(t_order_file,header=0)
        df_filter_order['month'] = df_filter_order['buy_time'].map(lambda x: conver_time(x))
        df_filter_order = df_filter_order[df_filter_order['month']!=11]
        df_filter_order['price'] = df_filter_order['price'].map(lambda x: change_data(x))
        df_filter_order['discount'] = df_filter_order['discount'].map(lambda x: change_data(x))
        # 购买商品实际支付价格
        df_filter_order['real_price'] = df_filter_order['price']*df_filter_order['qty'] - df_filter_order['discount']
        df_filter_order['real_price'][df_filter_order['real_price']<0] = 0.

        # 购买行为在滑动时间窗口内的购买总额
        df_filter_order['days'] = df_filter_order['buy_time'].map(lambda x: datetime.strptime('2016-11-1', '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
        df_filter_order['days'] = df_filter_order['days'].map(lambda x: int(x.days))
        uid = df_filter_order['uid'].unique()
        exclu = [1]*len(uid)
        days_df = pd.DataFrame({'uid':uid,'exclu':exclu})
        for day in [1,2,3,5,7,9,15,20,25,30,35,40,45,50,60,70,80,90]:
            df = df_filter_order[['uid','days','real_price']].copy()
            day_df = map_day2period(df,'real_price', 'order', day)
            days_df = pd.merge(days_df,day_df,how='left',on='uid')
        days_df = days_df.fillna(0)
        del days_df['exclu']
        change_list = list(days_df.columns)
        change_list.remove('uid')
        for col in change_list:
            days_df[col] = days_df[col].map(lambda x : math.log(x+1,5))
        df_filter_order = days_df
        pickle.dump(df_filter_order, open(dump_path, 'wb'))
    return df_filter_order

def gen_basic_click_feat():
    dump_path = './tmp/train_click_feat.pkl'
    if os.path.exists(dump_path):
        df_click = pickle.load(open(dump_path,'rb'))
    else:
        df_click = pd.read_csv(t_click_file,header=0)
        df_click['month'] = df_click['click_time'].map(lambda x: conver_time(x))
        df_click = df_click[df_click['month'] != 11]

        # 点击时间特征分布
        click_hour_df = df_click.copy()
        click_hour_df['click_time_hours'] = click_hour_df['click_time'].map(lambda x: int(x.split(' ')[1].split(':')[0]))
        click_hour_df['click_time_hours'] = click_hour_df['click_time_hours'].map(lambda x : map_hours2bucket('click',x))
        click_hour_df = click_hour_df.groupby(['uid','click_time_hours'],as_index=False).count()
        click_hour_df = click_hour_df.pivot(index='uid', columns='click_time_hours', values='click_time').reset_index()
        click_hour_df = click_hour_df.fillna(0)
        column_list = list(click_hour_df.columns)
        column_list.remove('uid')
        for fea in column_list:
            click_hour_df[fea] = click_hour_df[fea].map(lambda x:math.log(x+1,5))

        df_click['click_weights'] = df_click['click_time'].map(lambda x: datetime.strptime('2016-11-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_click['click_weights'] = df_click['click_weights'].map(lambda x: 1/(1e-6+x.days))
        del df_click['click_time']

        pid_df = pd.get_dummies(df_click["pid"], prefix="pid")
        param_df = pd.get_dummies(df_click["param"], prefix="param")
        del df_click['pid']
        del df_click['param']
        df_click = pd.concat([df_click,pid_df,param_df],axis=1)

        column_list = list(df_click.columns)
        column_list.remove('uid')
        column_list.remove('click_weights')
        column_list.remove('month')
        for fea in column_list:
            df_click[fea] = df_click[fea]*df_click['click_weights']

        df_click['click'] = 1
        statistic_df = df_click.groupby(['uid','month'],as_index=False).sum().copy()
        statistic_df = statistic_df.pivot(index='uid', columns='month', values='click').reset_index()
        statistic_df = statistic_df.fillna(0)
        statistic_df['click_min'] = statistic_df[[8,9,10]].apply(lambda x: x.min(),axis=1)
        statistic_df['click_max'] = statistic_df[[8,9,10]].apply(lambda x: x.max(),axis=1)
        statistic_df['click_sum'] = statistic_df[[8,9,10]].apply(lambda x: x.sum(),axis=1)
        statistic_df['click_mean'] = statistic_df[[8,9,10]].apply(lambda x: x.mean(),axis=1)
        statistic_df['click_std'] = statistic_df[[8,9,10]].apply(lambda x: x.std(),axis=1)
        statistic_df['click_std'] = statistic_df['click_std'].map(lambda x : 1/(1+x))

        del df_click['month']
        del df_click['click']
        df_click = df_click.groupby(['uid'],as_index=False).sum()
        df_click = pd.merge(df_click,statistic_df[['uid','click_min','click_max','click_mean','click_sum','click_std']], how='left',on='uid')
        df_click = pd.merge(df_click,click_hour_df,how='left',on='uid')

        click_cluster_label = user_cluster(df_click,'train','click')
        df_click['click_cluster_label'] = click_cluster_label

        pickle.dump(df_click, open(dump_path, 'wb'))
    return df_click

def gen_filter_click_feat():
    dump_path = './tmp/train_filter_click_feat.pkl'
    if os.path.exists(dump_path):
        df_filter_click = pickle.load(open(dump_path,'rb'))
    else:
        df_filter_click = pd.read_csv(t_click_file,header=0)

        df_filter_click['month'] = df_filter_click['click_time'].map(lambda x: conver_time(x))
        df_filter_click = df_filter_click[df_filter_click['month'] != 11]
        # 点击行为在滑动时间窗口内的点击次数
        df_filter_click['days'] = df_filter_click['click_time'].map(lambda x: datetime.strptime('2016-11-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_filter_click['days'] = df_filter_click['days'].map(lambda x: x.days)
        df_filter_click['click_num'] = 1
        uid = df_filter_click['uid'].unique()
        exclu = [1]*len(uid)
        days_df = pd.DataFrame({'uid':uid,'exclu':exclu})
        for day in [1,2,3,5,7,9,15,20,25,30,35,40,45,50,60,70,80,90]:
            df = df_filter_click[['uid','days','click_num']].copy()
            day_df = map_day2period(df,'click_num', 'click', day)
            days_df = pd.merge(days_df,day_df,how='left',on='uid')
        days_df = days_df.fillna(0)
        del days_df['exclu']

        df_filter_click = days_df

        change_list = list(df_filter_click.columns)
        change_list.remove('uid')
        for col in change_list:
            df_filter_click[col] = df_filter_click[col].map(lambda x : math.log(x+1,5))

        pickle.dump(df_filter_click,open(dump_path,'wb'))
    return df_filter_click

def gen_train_test_user():
    dump_path = './tmp/train_test_user.pkl'
    if os.path.exists(dump_path):
        train_test_user = pickle.load(open(dump_path, 'rb'))
    else:
        import random
        df_loan = pd.read_csv(t_loan_file,header=0)
        df_user = pd.read_csv(t_user_file,header=0)
        df_loan['month'] = df_loan['loan_time'].map(lambda x: conver_time(x))
        # 各月的贷款用户名单
        month_8_uid = set(df_loan[df_loan['month'] == 8]['uid'])
        month_9_uid = set(df_loan[df_loan['month'] == 9]['uid'])
        month_10_uid = set(df_loan[df_loan['month'] == 10]['uid'])
        month_11_uid = set(df_loan[df_loan['month'] == 11]['uid'])
        none_user = set(df_user['uid']) - (month_11_uid | month_10_uid | month_9_uid | month_8_uid)

        # 各月新增用户名单
        new_loan_user_9 = month_9_uid - month_8_uid
        new_loan_user_10 = month_10_uid - (month_9_uid | month_8_uid)
        new_loan_user_11 = month_11_uid - (month_10_uid | month_9_uid | month_8_uid)
        miss_user_11 = (month_10_uid | month_9_uid | month_8_uid) - month_11_uid
        keep_user_11 = month_11_uid - new_loan_user_11

        # xianxia训练集
        model1_user = month_8_uid | month_9_uid | month_10_uid
        model1_test_user1 = random.sample(list(keep_user_11), int(0.2*len(keep_user_11)))
        model1_test_user = random.sample(list(miss_user_11), int(0.2*len(miss_user_11)))
        model1_test_user.extend(model1_test_user1)
        model1_train_user = list(model1_user - set(model1_test_user))

        model2_user = set(df_user['uid']) - model1_user
        model2_test_user1 = random.sample(list(none_user), int(0.2*len(none_user)))
        model2_test_user = random.sample(list(new_loan_user_11), int(0.2*len(new_loan_user_11)))
        model2_test_user.extend(model2_test_user1)
        model2_train_user = list(model2_user - set(model2_test_user))

        # 线上提交测试集
        sub_model1_user = list(month_11_uid | month_9_uid | month_10_uid)
        sub_model2_user = list(set(df_user['uid']) - set(sub_model1_user))

        train_test_user = {'model1_test_user':model1_test_user,'model1_train_user':model1_train_user,\
                            'model2_test_user':model2_test_user,'model2_train_user':model2_train_user,\
                            'sub_model1_user':sub_model1_user,'sub_model2_user':sub_model2_user}

        pickle.dump(train_test_user, open(dump_path, 'wb'))
    return train_test_user

def gen_labels2():
    dump_path = './tmp/train_label_clf.pkl'
    if os.path.exists(dump_path):
        df_label = pickle.load(open(dump_path,'rb'))
    else:
        user_dict = gen_user_dict()
        new_loan_user_11 = list(user_dict['new_loan_user_11'])
        class1 = [1] * len(new_loan_user_11)
        miss_user_11 = list(user_dict['miss_user_11'])
        class2 = [2] * len(miss_user_11)
        keep_user_11 = list(user_dict['keep_user_11'])
        class3 = [3] * len(keep_user_11)

        new_loan_user_11.extend(miss_user_11)
        new_loan_user_11.extend(keep_user_11)

        class1.extend(class2)
        class1.extend(class3)

        data = {'uid':new_loan_user_11, 'label2':class1}
        df_label = pd.DataFrame(data)
        pickle.dump(df_label, open(dump_path, 'wb'))
        print(df_label['label2'].value_counts())
    return  df_label

def gen_labels():
    dump_path = './tmp/train_label_reg.pkl'
    if os.path.exists(dump_path):
        df_label = pickle.load(open(dump_path,'rb'))
    else:
        df_label = pd.read_csv(t_loan_sum_file,header=0)
        df_label['label'] = df_label['loan_sum']
        df_label = df_label[['uid','label']]
        pickle.dump(df_label, open(dump_path, 'wb'))
        print(df_label['label'].describe())
    return  df_label

def cal_percent(list_num):
    list_num = sorted(list_num)
    num_distant = list_num[-1] - list_num[0]
    four_1 = 0.25 * num_distant + list_num[0]
    four_2 = 0.50 * num_distant + list_num[0]
    four_3 = 0.75 * num_distant + list_num[0]
    return four_1,four_2,four_3

def map_fea(four_1,four_2,four_3,feature):
    if feature <= four_1:
        return 1
    elif feature <= four_2:
        return 2
    elif feature <= four_3:
        return 3
    else:
        return 4

def gen_union_feat():
    train_dump_path = './tmp/train_union_feat.pkl'
    test_dump_path = './tmp/test_union_feat.pkl'
    if os.path.exists(train_dump_path):
        train_union_feat = pickle.load(open(train_dump_path,'rb'))
        test_union_feat = pickle.load(open(test_dump_path,'rb'))
    else:
        df_loan = pd.read_csv(t_loan_file,header=0)
        df_loan['month'] = df_loan['loan_time'].map(lambda x: conver_time(x))
        df_loan['loan_amount'] = df_loan['loan_amount'].map(lambda x: round(change_data(x)))
        df_loan = df_loan.groupby(['uid','month'],as_index=False).sum()

        df_order = pd.read_csv(t_order_file,header=0)
        df_order['month'] = df_order['buy_time'].map(lambda x: conver_time(x))
        df_order['price'] = df_order['price'].map(lambda x: change_data(x))
        df_order['discount'] = df_order['discount'].map(lambda x: change_data(x))
        df_order['real_price'] = df_order['price']*df_order['qty'] - df_order['discount']
        df_order['real_price'][df_order['real_price']<0] = 0.
        df_order = df_order.groupby(['uid','month'],as_index=False).sum()

        df_click = pd.read_csv(t_click_file,header=0)
        df_click['month'] = df_click['click_time'].map(lambda x: conver_time(x))
        df_click = df_click.groupby(['uid','month'],as_index=False).count()

        df_loan= df_loan[['uid','month','loan_amount']]
        df_order = df_order[['uid','month','real_price']]
        df_click = df_click[['uid','month','click_time']]
        union_df = pd.merge(df_loan,df_order,how='outer',on=['uid','month'])
        union_df = pd.merge(union_df,df_click,how='outer',on=['uid','month'])
        union_df = union_df.fillna(0)

        change_list = ['loan_amount','real_price','click_time']
        for col in change_list:
            union_df[col] = union_df[col].map(lambda x : math.log(x+1,5))

        four_1,four_2,four_3 = cal_percent(union_df['loan_amount'])
        union_df['loan_amount'] = union_df['loan_amount'].map(lambda x :map_fea(four_1,four_2,four_3,x))

        four_1,four_2,four_3 = cal_percent(union_df['real_price'])
        union_df['real_price'] = union_df['real_price'].map(lambda x :map_fea(four_1,four_2,four_3,x))

        four_1,four_2,four_3 = cal_percent(union_df['click_time'])
        union_df['click_time'] = union_df['click_time'].map(lambda x :map_fea(four_1,four_2,four_3,x))

        union_df = union_df.pivot(index='uid', columns='month').reset_index()
        union_df = union_df.fillna(0)

        train_df = union_df.copy()
        train_df['lp_first'] = train_df['loan_amount'][8]+train_df['real_price'][8]
        train_df['lc_first'] = train_df['loan_amount'][8]+train_df['click_time'][8]
        train_df['pc_first'] = train_df['real_price'][8]+train_df['click_time'][8]
        train_df['lpc_first'] = train_df['loan_amount'][8]+train_df['real_price'][8]+train_df['click_time'][8]

        train_df['lp_two'] = train_df['loan_amount'][9]+train_df['real_price'][9]
        train_df['lc_two'] = train_df['loan_amount'][9]+train_df['click_time'][9]
        train_df['pc_two'] = train_df['real_price'][9]+train_df['click_time'][9]
        train_df['lpc_two'] = train_df['loan_amount'][9]+train_df['real_price'][9]+train_df['click_time'][9]

        train_df['lp_three'] = train_df['loan_amount'][10]+train_df['real_price'][10]
        train_df['lc_three'] = train_df['loan_amount'][10]+train_df['click_time'][10]
        train_df['pc_three'] = train_df['real_price'][10]+train_df['click_time'][10]
        train_df['lpc_three'] = train_df['loan_amount'][10]+train_df['real_price'][10]+train_df['click_time'][10]
        train_df = train_df[['uid','lp_first','lc_first','pc_first','lpc_first','lp_two','lc_two','pc_two','lpc_two','lp_three','lc_three','pc_three','lpc_three']]
        columns = ['uid','lp_first','lc_first','pc_first','lpc_first','lp_two','lc_two','pc_two','lpc_two','lp_three','lc_three','pc_three','lpc_three']
        train_union_feat = pd.DataFrame(train_df.values,columns = columns)
        columns.remove('uid')
        train_union_feat['union_min'] = train_union_feat[columns].apply(lambda x: x.min(),axis=1)
        train_union_feat['union_max'] = train_union_feat[columns].apply(lambda x: x.max(),axis=1)
        train_union_feat['union_sum'] = train_union_feat[columns].apply(lambda x: x.sum(),axis=1)
        train_union_feat['union_mean'] = train_union_feat[columns].apply(lambda x: x.mean(),axis=1)
        train_union_feat['union_std'] = train_union_feat[columns].apply(lambda x: x.std(),axis=1)
        train_union_feat['union_std'] = train_union_feat['union_std'].map(lambda x : 1/(1+x))
        # action_df = pd.get_dummies(train_union_feat[['lp_first','lc_first','pc_first','lpc_first','lp_two','lc_two','pc_two','lpc_two','lp_three','lc_three','pc_three','lpc_three']])
        # train_union_feat = pd.concat([train_union_feat['uid'],action_df],axis=1)

        test_df = union_df.copy()
        test_df['lp_first'] = test_df['loan_amount'][9]+test_df['real_price'][9]
        test_df['lc_first'] = test_df['loan_amount'][9]+test_df['click_time'][9]
        test_df['pc_first'] = test_df['real_price'][9]+test_df['click_time'][9]
        test_df['lpc_first'] = test_df['loan_amount'][9]+test_df['real_price'][9]+test_df['click_time'][9]

        test_df['lp_two'] = test_df['loan_amount'][10]+test_df['real_price'][10]
        test_df['lc_two'] = test_df['loan_amount'][10]+test_df['click_time'][10]
        test_df['pc_two'] = test_df['real_price'][10]+test_df['click_time'][10]
        test_df['lpc_two'] = test_df['loan_amount'][10]+test_df['real_price'][10]+test_df['click_time'][10]

        test_df['lp_three'] = test_df['loan_amount'][11]+test_df['real_price'][11]
        test_df['lc_three'] = test_df['loan_amount'][11]+test_df['click_time'][11]
        test_df['pc_three'] = test_df['real_price'][11]+test_df['click_time'][11]
        test_df['lpc_three'] = test_df['loan_amount'][11]+test_df['real_price'][11]+test_df['click_time'][11]
        test_df = test_df[['uid','lp_first','lc_first','pc_first','lpc_first','lp_two','lc_two','pc_two','lpc_two','lp_three','lc_three','pc_three','lpc_three']]
        columns = ['uid','lp_first','lc_first','pc_first','lpc_first','lp_two','lc_two','pc_two','lpc_two','lp_three','lc_three','pc_three','lpc_three']
        test_union_feat = pd.DataFrame(test_df.values,columns = columns)
        columns.remove('uid')
        test_union_feat['union_min'] = test_union_feat[columns].apply(lambda x: x.min(),axis=1)
        test_union_feat['union_max'] = test_union_feat[columns].apply(lambda x: x.max(),axis=1)
        test_union_feat['union_sum'] = test_union_feat[columns].apply(lambda x: x.sum(),axis=1)
        test_union_feat['union_mean'] = test_union_feat[columns].apply(lambda x: x.mean(),axis=1)
        test_union_feat['union_std'] = test_union_feat[columns].apply(lambda x: x.std(),axis=1)
        test_union_feat['union_std'] = test_union_feat['union_std'].map(lambda x : 1/(1+x))
        # action_df = pd.get_dummies(test_union_feat[['lp_first','lc_first','pc_first','lpc_first','lp_two','lc_two','pc_two','lpc_two','lp_three','lc_three','pc_three','lpc_three']])
        # test_union_feat = pd.concat([test_union_feat['uid'],action_df],axis=1)

        # train_feat = set(train_union_feat.columns)
        # test_feat = set(test_union_feat.columns)
        # comm_feat = train_feat & test_feat
        # comm_feat = sorted(list(comm_feat))

        # # map_comm_feat2num = {feat:i for feat,i in enumerate(comm_feat)}

        # train_union_feat = train_union_feat[comm_feat]
        # test_union_feat = test_union_feat[comm_feat]

        pickle.dump(train_union_feat, open(train_dump_path, 'wb'))
        pickle.dump(test_union_feat, open(test_dump_path, 'wb'))

    return train_union_feat,test_union_feat

def gen_interactive_feat():
    dump_path = './tmp/train_interactive_feat.pkl'
    if os.path.exists(dump_path):
        train_interactive_feat = pickle.load(open(dump_path,'rb'))
    else:
        df_loan_ori = pd.read_csv(t_loan_file,header=0)
        df_loan_ori['month'] = df_loan_ori['loan_time'].map(lambda x: conver_time(x))
        df_loan_ori = df_loan_ori[df_loan_ori['month'] != 11]
        df_loan_ori['days'] = df_loan_ori['loan_time'].map(lambda x: datetime.strptime('2016-11-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_loan_ori['days'] = df_loan_ori['days'].map(lambda x: int(x.days))
        df_loan_ori['loan_amount'] = df_loan_ori['loan_amount'].map(lambda x: round(change_data(x)))

        df_order_ori = pd.read_csv(t_order_file,header=0)
        df_order_ori['month'] = df_order_ori['buy_time'].map(lambda x: conver_time(x))
        df_order_ori = df_order_ori[df_order_ori['month'] != 11]
        df_order_ori['days'] = df_order_ori['buy_time'].map(lambda x: datetime.strptime('2016-11-1', '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
        df_order_ori['days'] = df_order_ori['days'].map(lambda x: int(x.days))
        df_order_ori['price'] = df_order_ori['price'].map(lambda x: change_data(x))
        df_order_ori['discount'] = df_order_ori['discount'].map(lambda x: change_data(x))
        df_order_ori['real_price'] = df_order_ori['price']*df_order_ori['qty'] - df_order_ori['discount']
        df_order_ori['real_price'][df_order_ori['real_price']<0] = 0.

        df_click_ori = pd.read_csv(t_click_file,header=0)
        df_click_ori['month'] = df_click_ori['click_time'].map(lambda x: conver_time(x))
        df_click_ori = df_click_ori[df_click_ori['month'] != 11]
        df_click_ori['days'] = df_click_ori['click_time'].map(lambda x: datetime.strptime('2016-11-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_click_ori['days'] = df_click_ori['days'].map(lambda x: int(x.days))

        df_loan= df_loan_ori[['uid','month','loan_amount']].copy()
        df_loan = df_loan.groupby(['uid','month'],as_index=False).sum()
        df_loan['loan_amount'] = df_loan['loan_amount'].map(lambda x : math.log(x+1,5))
        df_loan = df_loan.pivot(index='uid', columns='month').reset_index()
        df_loan.fillna(0,inplace=True)

        df_order = df_order_ori[['uid','month','real_price']].copy()
        df_order = df_order.groupby(['uid','month'],as_index=False).sum()
        df_order['real_price'] = df_order['real_price'].map(lambda x : math.log(x+1,5))
        df_order = df_order.pivot(index='uid', columns='month').reset_index()
        df_order.fillna(0,inplace=True)

        df_click = df_click_ori[['uid','month','click_time']].copy()
        df_click = df_click.groupby(['uid','month'],as_index=False).count()
        df_click['click_time'] = df_click['click_time'].map(lambda x : math.log(x+1,5))
        df_click = df_click.pivot(index='uid', columns='month').reset_index()
        df_click.fillna(0,inplace=True)

        loan_order_commom_user = sorted(list(set(df_loan['uid']) & set(df_order['uid'])))
        click_order_commom_user = sorted(list(set(df_click['uid']) & set(df_order['uid'])))
        loan_click_commom_user = sorted(list(set(df_loan['uid']) & set(df_click['uid'])))
        loan_click_order_commom_user = sorted(list(set(df_loan['uid']) & set(df_click['uid']) & set(df_order['uid'])))

        loan_ = df_loan[df_loan.uid.isin(loan_order_commom_user)].sort_values(by='uid')
        order_ = df_order[df_order.uid.isin(loan_order_commom_user)].sort_values(by='uid')
        loan_order = loan_.loc[:,[False,True,True,True]].values * order_.loc[:,[False,True,True,True]].values
        loan_order = np.c_[np.array(loan_['uid']),loan_order]
        loan_order = pd.DataFrame(loan_order,columns=['uid','lo_1','lo_2','lo_3'])

        click_ = df_click[df_click.uid.isin(click_order_commom_user)].sort_values(by='uid')
        order_ = df_order[df_order.uid.isin(click_order_commom_user)].sort_values(by='uid')
        clcik_order = click_.loc[:,[False,True,True,True]].values * order_.loc[:,[False,True,True,True]].values
        clcik_order = np.c_[np.array(click_['uid']),clcik_order]
        clcik_order = pd.DataFrame(clcik_order,columns=['uid','co_1','co_2','co_3'])

        loan_ = df_loan[df_loan.uid.isin(loan_click_commom_user)].sort_values(by='uid')
        click_ = df_click[df_click.uid.isin(loan_click_commom_user)].sort_values(by='uid')
        loan_click = loan_.loc[:,[False,True,True,True]].values * click_.loc[:,[False,True,True,True]].values
        loan_click = np.c_[np.array(loan_['uid']),loan_click]
        loan_click = pd.DataFrame(loan_click,columns=['uid','lc_1','lc_2','lc_3'])

        loan_ = df_loan[df_loan.uid.isin(loan_click_order_commom_user)].sort_values(by='uid')
        click_ = df_click[df_click.uid.isin(loan_click_order_commom_user)].sort_values(by='uid')
        order_ = df_order[df_order.uid.isin(loan_click_order_commom_user)].sort_values(by='uid')
        loan_click_order = loan_.loc[:,[False,True,True,True]].values * click_.loc[:,[False,True,True,True]].values * order_.loc[:,[False,True,True,True]].values
        loan_click_order = np.c_[np.array(loan_['uid']),loan_click_order]
        loan_click_order = pd.DataFrame(loan_click_order,columns=['uid','lco_1','lco_2','lco_3'])

        train_interactive_feat = pd.merge(loan_order,clcik_order,how='outer',on='uid')
        train_interactive_feat = pd.merge(train_interactive_feat,loan_click,how='outer',on='uid')
        train_interactive_feat = pd.merge(train_interactive_feat,loan_click_order,how='outer',on='uid')
        train_interactive_feat = train_interactive_feat.fillna(0)

        feature = list(train_interactive_feat.columns)
        feature.remove('uid')
        train_interactive_feat['interactive_min'] = train_interactive_feat[feature].apply(lambda x: x.min(),axis=1)
        train_interactive_feat['interactive_max'] = train_interactive_feat[feature].apply(lambda x: x.max(),axis=1)
        train_interactive_feat['interactive_sum'] = train_interactive_feat[feature].apply(lambda x: x.sum(),axis=1)
        train_interactive_feat['interactive_mean'] = train_interactive_feat[feature].apply(lambda x: x.mean(),axis=1)
        train_interactive_feat['interactive_std'] = train_interactive_feat[feature].apply(lambda x: x.std(),axis=1)
        train_interactive_feat['interactive_std'] = train_interactive_feat['interactive_std'].map(lambda x : 1/(1+x))


        weeks_df = map_day2week(df_loan_ori.copy(),'loan_amount','loan')
        loan_weeks_df = weeks_df.copy()
        loan_weeks_df['loan_weeks_min'] = loan_weeks_df[['loan_week1','loan_week2','loan_week3','loan_week4']].apply(lambda x: x.min(),axis=1)
        loan_weeks_df['loan_weeks_max'] = loan_weeks_df[['loan_week1','loan_week2','loan_week3','loan_week4']].apply(lambda x: x.max(),axis=1)
        loan_weeks_df['loan_weeks_sum'] = loan_weeks_df[['loan_week1','loan_week2','loan_week3','loan_week4']].apply(lambda x: x.sum(),axis=1)
        loan_weeks_df['loan_weeks_mean'] = loan_weeks_df[['loan_week1','loan_week2','loan_week3','loan_week4']].apply(lambda x: x.mean(),axis=1)
        loan_weeks_df['loan_weeks_std'] = loan_weeks_df[['loan_week1','loan_week2','loan_week3','loan_week4']].apply(lambda x: x.std(),axis=1)
        loan_weeks_df['loan_weeks_std'] = loan_weeks_df['loan_weeks_std'].map(lambda x : 1/(1+x))
        loan_weeks_df = loan_weeks_df[['uid','loan_weeks_min','loan_weeks_max','loan_weeks_sum','loan_weeks_mean','loan_weeks_std']]

        weeks_df1 = map_day2week(df_order_ori.copy(),'real_price','order')
        order_weeks_df = weeks_df1.copy()
        order_weeks_df['order_weeks_min'] = order_weeks_df[['order_week1','order_week2','order_week3','order_week4']].apply(lambda x: x.min(),axis=1)
        order_weeks_df['order_weeks_max'] = order_weeks_df[['order_week1','order_week2','order_week3','order_week4']].apply(lambda x: x.max(),axis=1)
        order_weeks_df['order_weeks_sum'] = order_weeks_df[['order_week1','order_week2','order_week3','order_week4']].apply(lambda x: x.sum(),axis=1)
        order_weeks_df['order_weeks_mean'] = order_weeks_df[['order_week1','order_week2','order_week3','order_week4']].apply(lambda x: x.mean(),axis=1)
        order_weeks_df['order_weeks_std'] = order_weeks_df[['order_week1','order_week2','order_week3','order_week4']].apply(lambda x: x.std(),axis=1)
        order_weeks_df['order_weeks_std'] = order_weeks_df['order_weeks_std'].map(lambda x : 1/(1+x))
        order_weeks_df = order_weeks_df[['uid','order_weeks_min','order_weeks_max','order_weeks_sum','order_weeks_mean','order_weeks_std']]

        df_click_ori['ctime'] = 1
        weeks_df2 = map_day2week(df_click_ori.copy(),'ctime','click')
        click_weeks_df = weeks_df2.copy()
        click_weeks_df['click_weeks_min'] = click_weeks_df[['click_week1','click_week2','click_week3','click_week4']].apply(lambda x: x.min(),axis=1)
        click_weeks_df['click_weeks_max'] = click_weeks_df[['click_week1','click_week2','click_week3','click_week4']].apply(lambda x: x.max(),axis=1)
        click_weeks_df['click_weeks_sum'] = click_weeks_df[['click_week1','click_week2','click_week3','click_week4']].apply(lambda x: x.sum(),axis=1)
        click_weeks_df['click_weeks_mean'] = click_weeks_df[['click_week1','click_week2','click_week3','click_week4']].apply(lambda x: x.mean(),axis=1)
        click_weeks_df['click_weeks_std'] = click_weeks_df[['click_week1','click_week2','click_week3','click_week4']].apply(lambda x: x.std(),axis=1)
        click_weeks_df['click_weeks_std'] = click_weeks_df['click_weeks_std'].map(lambda x : 1/(1+x))
        click_weeks_df = click_weeks_df[['uid','click_weeks_min','click_weeks_max','click_weeks_sum','click_weeks_mean','click_weeks_std']]

        action_weeks_df = pd.merge(loan_weeks_df,order_weeks_df,how='outer',on='uid')
        action_weeks_df = pd.merge(action_weeks_df,click_weeks_df,how='outer',on='uid')

        loan_ = weeks_df[weeks_df.uid.isin(loan_order_commom_user)].sort_values(by='uid')
        order_ = weeks_df1[weeks_df1.uid.isin(loan_order_commom_user)].sort_values(by='uid')
        week_loan_order = loan_.loc[:,[False,True,True,True,True]].values * order_.loc[:,[False,True,True,True,True]].values
        week_loan_order = np.c_[np.array(loan_['uid']),week_loan_order]
        week_loan_order = pd.DataFrame(week_loan_order,columns=['uid','week_lo_1','week_lo_2','week_lo_3','week_lo_4'])

        click_ = weeks_df2[weeks_df2.uid.isin(click_order_commom_user)].sort_values(by='uid')
        order_ = weeks_df1[weeks_df1.uid.isin(click_order_commom_user)].sort_values(by='uid')
        week_click_order = click_.loc[:,[False,True,True,True,True]].values * order_.loc[:,[False,True,True,True,True]].values
        week_click_order = np.c_[np.array(click_['uid']),week_click_order]
        week_click_order = pd.DataFrame(week_click_order,columns=['uid','week_co_1','week_co_2','week_co_3','week_co_4'])

        loan_ = weeks_df[weeks_df.uid.isin(loan_click_commom_user)].sort_values(by='uid')
        click_ = weeks_df2[weeks_df2.uid.isin(loan_click_commom_user)].sort_values(by='uid')
        week_loan_click = loan_.loc[:,[False,True,True,True,True]].values * click_.loc[:,[False,True,True,True,True]].values
        week_loan_click = np.c_[np.array(loan_['uid']),week_loan_click]
        week_loan_click = pd.DataFrame(week_loan_click,columns=['uid','week_lc_1','week_lc_2','week_lc_3','week_lc_4'])

        loan_ = weeks_df[weeks_df.uid.isin(loan_click_order_commom_user)].sort_values(by='uid')
        click_ = weeks_df2[weeks_df2.uid.isin(loan_click_order_commom_user)].sort_values(by='uid')
        order_ = weeks_df1[weeks_df1.uid.isin(loan_click_order_commom_user)].sort_values(by='uid')
        week_loan_click_order = loan_.loc[:,[False,True,True,True,True]].values * click_.loc[:,[False,True,True,True,True]].values * order_.loc[:,[False,True,True,True,True]].values
        week_loan_click_order = np.c_[np.array(loan_['uid']),week_loan_click_order]
        week_loan_click_order = pd.DataFrame(week_loan_click_order,columns=['uid','week_lco_1','week_lco_2','week_lco_3','week_lco_4'])

        train_interactive_week_feat = pd.merge(week_loan_order,week_click_order,how='outer',on='uid')
        train_interactive_week_feat = pd.merge(train_interactive_week_feat,week_loan_click,how='outer',on='uid')
        train_interactive_week_feat = pd.merge(train_interactive_week_feat,week_loan_click_order,how='outer',on='uid')
        train_interactive_week_feat = train_interactive_week_feat.fillna(0)

        feature = list(train_interactive_week_feat.columns)
        feature.remove('uid')
        train_interactive_week_feat['interactive_min'] = train_interactive_week_feat[feature].apply(lambda x: x.min(),axis=1)
        train_interactive_week_feat['interactive_max'] = train_interactive_week_feat[feature].apply(lambda x: x.max(),axis=1)
        train_interactive_week_feat['interactive_sum'] = train_interactive_week_feat[feature].apply(lambda x: x.sum(),axis=1)
        train_interactive_week_feat['interactive_mean'] = train_interactive_week_feat[feature].apply(lambda x: x.mean(),axis=1)
        train_interactive_week_feat['interactive_std'] = train_interactive_week_feat[feature].apply(lambda x: x.std(),axis=1)
        train_interactive_week_feat['interactive_std'] = train_interactive_week_feat['interactive_std'].map(lambda x : 1/(1+x))

        train_interactive_feat = pd.merge(train_interactive_feat,action_weeks_df,how='outer',on='uid')
        train_interactive_feat = pd.merge(train_interactive_feat,train_interactive_week_feat,how='outer',on='uid')
        train_interactive_feat = train_interactive_feat.fillna(0)
        pickle.dump(train_interactive_feat, open(dump_path, 'wb'))
    return train_interactive_feat


def make_train_set():
    dump_path = './data/training.pkl'
    if os.path.exists(dump_path):
        train_set = pickle.load(open(dump_path,'rb'))
    else:
        df_user = gen_basic_user_feat()
        df_order = gen_basic_order_feat()
        df_filter_order = gen_filter_order_feat()
        df_loan = gen_basic_loan_feat()
        df_filter_loan = gen_filter_loan_feat()
        df_click = gen_basic_click_feat()
        df_filter_click = gen_filter_click_feat()
        train_union_feat,_ = gen_union_feat()
        df_interactive_feat = gen_interactive_feat()
        df_label = gen_labels()
        # df_label2 = gen_labels2()

        train_set = pd.merge(df_user, df_order, how='outer', on='uid')
        train_set = pd.merge(train_set, df_filter_order, how='outer', on='uid')
        train_set = pd.merge(train_set, df_loan, how='outer', on='uid')
        train_set = pd.merge(train_set, df_filter_loan, how='outer', on='uid')
        train_set['repay_cost'] = train_set['repay']+train_set['cost_weight']
        train_set = pd.merge(train_set, df_click, how='outer', on='uid')
        train_set = pd.merge(train_set, df_filter_click, how='outer', on='uid')
        train_set = pd.merge(train_set, train_union_feat, how='outer', on='uid')
        train_set = pd.merge(train_set, df_interactive_feat, how='outer', on='uid')
        train_set = pd.merge(train_set, df_label, how='outer', on='uid')
        # train_set = pd.merge(train_set, df_label2, how='outer', on='uid')
        train_set = train_set.fillna(0)

        change_list = ['cost_weight','real_price','buy_weights','loan_amount','click_weights','click_min','click_max','click_mean','click_sum','click_std',\
                        'per_times_loan','repay','repay_cost','loan_min','loan_max','loan_mean','loan_sum','loan_std','limit',\
                        'buy_min','buy_mean','buy_max','buy_sum','buy_std']
        for col in change_list:
            train_set[col] = train_set[col].map(lambda x : math.log(x+1,5))

        feat = list(train_set.columns)
        feat.remove('label')
        cluster_label = user_cluster(train_set[feat],'train','all')
        train_set['cluster_label'] = cluster_label

        pickle.dump(train_set, open(dump_path, 'wb'))
    feat_id = {i:fea for i ,fea in enumerate(list(train_set.columns))}
    print(feat_id)

    return train_set

def make_training_data():
    dump_path1 = './data/model1_training.pkl'
    dump_path2 = './data/model2_training.pkl'
    if os.path.exists(dump_path1):
        model1_training = pickle.load(open(dump_path1,'rb'))
        model2_training = pickle.load(open(dump_path2,'rb'))
    else:
        df_user = gen_basic_user_feat()
        df_order = gen_basic_order_feat()
        df_filter_order = gen_filter_order_feat()
        df_loan = gen_basic_loan_feat()
        df_filter_loan = gen_filter_loan_feat()
        df_click = gen_basic_click_feat()
        df_filter_click = gen_filter_click_feat()
        train_union_feat,_ = gen_union_feat()
        df_label = gen_labels()

        model1_training = pd.merge(df_user, df_order, how='outer', on='uid')
        model1_training = pd.merge(model1_training, df_filter_order, how='outer', on='uid')
        model1_training = pd.merge(model1_training, df_loan, how='outer', on='uid')
        model1_training = pd.merge(model1_training, df_filter_loan, how='outer', on='uid')
        model1_training['repay_cost'] = model1_training['repay']+model1_training['cost_weight']
        model1_training = pd.merge(model1_training, df_click, how='outer', on='uid')
        model1_training = pd.merge(model1_training, df_filter_click, how='outer', on='uid')
        # model1_training = pd.merge(model1_training, train_union_feat, how='outer', on='uid')
        model1_training = pd.merge(model1_training, df_label, how='outer', on='uid')
        model1_training = model1_training.fillna(0)

        change_list = ['cost_weight','real_price','buy_weights','loan_amount','click_weights','click_min','click_max','click_mean','click_sum','click_std',\
                        'per_times_loan','repay','repay_cost','loan_min','loan_max','loan_mean','loan_sum','loan_std','limit',\
                        'buy_min','buy_mean','buy_max','buy_sum','buy_std']
        for col in change_list:
            model1_training[col] = model1_training[col].map(lambda x : math.log(x+1,5))

        pickle.dump(model1_training, open(dump_path1, 'wb'))
        feat_id = {i:fea for i ,fea in enumerate(list(model1_training.columns))}
        print(feat_id)

        model2_training = pd.merge(df_user, df_order, how='outer', on='uid')
        model2_training = pd.merge(model2_training, df_filter_order, how='outer', on='uid')
        model2_training = pd.merge(model2_training, df_click, how='outer', on='uid')
        model2_training = pd.merge(model2_training, df_filter_click, how='outer', on='uid')
        model2_training = pd.merge(model2_training, train_union_feat, how='outer', on='uid')
        model2_training = pd.merge(model2_training, df_label, how='outer', on='uid')
        model2_training = model2_training.fillna(0)

        change_list = ['cost_weight','real_price','buy_weights','click_weights','click_min','click_max','click_mean','click_sum','click_std',\
                        'buy_min','buy_mean','buy_max','buy_sum','buy_std']
        for col in change_list:
            model2_training[col] = model2_training[col].map(lambda x : math.log(x+1,5))

        pickle.dump(model2_training, open(dump_path2, 'wb'))
        feat_id = {i:fea for i ,fea in enumerate(list(model2_training.columns))}
        print(feat_id)

    return model1_training,model2_training

def make_clf_train_set():
    dump_path = './data/clf_train_set.pkl'
    if os.path.exists(dump_path):
        clf_train_set = pickle.load(open(dump_path,'rb'))
    else:
        user_dict = gen_user_dict()
        df_user = gen_basic_user_feat()
        df_order = gen_basic_order_feat()
        df_click = gen_basic_click_feat()
        df_filter_click = gen_filter_click_feat()
        train_union_feat,_ = gen_union_feat()
        df_label2 = gen_labels2()

        clf_train_set = pd.merge(df_user, df_order, how='outer', on='uid')
        clf_train_set = pd.merge(clf_train_set, df_click, how='outer', on='uid')
        clf_train_set = pd.merge(clf_train_set, df_filter_click, how='outer', on='uid')
        clf_train_set = pd.merge(clf_train_set, train_union_feat, how='left', on='uid')
        clf_train_set = pd.merge(clf_train_set, df_label2, how='left', on='uid')
        clf_train_set = clf_train_set.fillna(0)

        change_list = ['cost_weight','real_price','click_weights',\
                        ]
        for col in change_list:
            clf_train_set[col] = clf_train_set[col].map(lambda x : math.log(x+1,5))

        pickle.dump(clf_train_set, open(dump_path, 'wb'))
    feat_id = {i:fea for i ,fea in enumerate(list(clf_train_set.columns))}
    print(feat_id)

    return clf_train_set

def spilt_train_test():
    X_train_dump_path = './data/X_train.pkl'
    X_test_dump_path = './data/X_test.pkl'
    y_train_dump_path = './data/y_train.pkl'
    y_test_dump_path = './data/y_test.pkl'
    y_train_dump_path2 = './data/y_train2.pkl'
    y_test_dump_path2 = './data/y_test2.pkl'
    if os.path.exists(X_train_dump_path):
        X_train = pickle.load(open(X_train_dump_path,'rb'))
        X_test = pickle.load(open(X_test_dump_path,'rb'))
        y_train = pickle.load(open(y_train_dump_path,'rb'))
        y_test = pickle.load(open(y_test_dump_path,'rb'))
        y_train2 = pickle.load(open(y_train_dump_path2,'rb'))
        y_test2 = pickle.load(open(y_test_dump_path2,'rb'))
    else:
        train_set = make_train_set()

        train_set.sample(frac=1)
        labels = train_set['label'].copy()
        labels = np.array(labels)

        labels2 = train_set['label2'].copy()
        labels2 = np.array(labels2)

        user_uid = train_set[['uid','label']].copy()

        del train_set['label']
        del train_set['label2']
        del train_set['uid']

        feat = sorted(list(train_set.columns))
        train_set = train_set[feat]
        feat_id = {i:fea for i ,fea in enumerate(list(train_set.columns))}
        print(feat_id)

        train_data = np.array(train_set)

        test_ratio = 0.2
        num = int(len(labels)*test_ratio)
        test_uid = user_uid.iloc[:num,]
        test_uid.to_csv('test_uid.csv',index = False)

        X_test = train_data[:num]
        X_train = train_data[num:]

        y_test = labels[:num]
        y_train = labels[num:]

        y_test2 = labels2[:num]
        y_train2 = labels2[num:]

        pickle.dump(X_train, open(X_train_dump_path, 'wb'))
        pickle.dump(X_test, open(X_test_dump_path, 'wb'))
        pickle.dump(y_train, open(y_train_dump_path, 'wb'))
        pickle.dump(y_test, open(y_test_dump_path, 'wb'))

        pickle.dump(y_train2, open(y_train_dump_path2, 'wb'))
        pickle.dump(y_test2, open(y_test_dump_path2, 'wb'))

    return X_train,X_test,y_train,y_test,y_train2,y_test2


def report(label, pred):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.scatter(range(len(pred)), sorted(pred))
    plt.xlabel('uid to reindex', fontsize=12)
    plt.ylabel('predict', fontsize=12)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(range(len(label)), sorted(label))
    plt.xlabel('uid to reindex', fontsize=12)
    plt.ylabel('label', fontsize=12)
    plt.show()

    label = np.array(label)
    pred = np.array(pred)
    a = pred - label
    b = a * a

    rmse = np.sqrt(np.sum(b)/len(pred))
    print('RMSE:',rmse)

    plt.figure(figsize=(8,6))
    plt.scatter(range(len(a)), sorted(a))
    plt.xlabel('uid to index', fontsize=12)
    plt.ylabel('The difference between pred and label', fontsize=12)
    plt.show()

if __name__ == '__main__':
    make_train_set()
    # make_clf_train_set()
    # make_training_data()
    # gen_train_test_user()














