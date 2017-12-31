# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
from gen_train_data import gen_union_feat,change_data,conver_time,map_day2period,map_hours2bucket,user_cluster,map_day2week

t_click_file = '../../t_click.csv'
t_loan_sum_file = '../../t_loan_sum.csv'
t_loan_file = '../../t_loan.csv'
t_order_file = '../../t_order.csv'
t_user_file = '../../t_user.csv'

def gen_user_feat():
    dump_path = './tmp/test_user_feat.pkl'
    if os.path.exists(dump_path):
        df_user = pickle.load(open(dump_path,'rb'))
    else:
        df_user = pd.read_csv(t_user_file,header=0)
        # 训练时的截止日期时11月，以11月初作为计算激活时长终止时间，预测时改为12月
        df_user['a_date'] = df_user['active_date'].map(lambda x: datetime.strptime('2016-12-1','%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
        df_user['a_date'] = df_user['a_date'].map(lambda x : x.days/7)
        df_user['limit'] = df_user['limit'].map(lambda x: change_data(x))
        del df_user['active_date']
        pickle.dump(df_user, open(dump_path, 'wb'))
    return df_user

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
        # df_filter_loan = df_filter_loan[df_filter_loan['loan_amount'] <= 90000]
        # df_filter_loan = df_filter_loan[df_filter_loan['loan_amount'] > 499]
        # 贷款行为在滑动时间窗口内的贷款总额
        df_filter_loan['days'] = df_filter_loan['loan_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_filter_loan['days'] = df_filter_loan['days'].map(lambda x: x.days)
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


def gen_loan_feat():
    dump_path = './tmp/test_loan_feat.pkl'
    if os.path.exists(dump_path):
        df_loan = pickle.load(open(dump_path,'rb'))
    else:
        df_loan = pd.read_csv(t_loan_file,header=0)
        df_loan['month'] = df_loan['loan_time'].map(lambda x: conver_time(x))
        df_loan['loan_amount'] = df_loan['loan_amount'].map(lambda x: round(change_data(x)))
        df_loan = df_loan[df_loan['month'] != 8]
        # df_loan = df_loan[df_loan['loan_amount'] <= 90000]
        # df_loan = df_loan[df_loan['loan_amount'] >= 499]

        # 贷款时间分布
        loan_hour_df = df_loan.copy()
        loan_hour_df['loan_time_hours'] = loan_hour_df['loan_time'].map(lambda x: int(x.split(' ')[1].split(':')[0]))
        loan_hour_df['loan_time_hours'] = loan_hour_df['loan_time_hours'].map(lambda x : map_hours2bucket('loan',x))
        loan_hour_df = loan_hour_df.groupby(['uid','loan_time_hours'],as_index=False).count()
        loan_hour_df = loan_hour_df.pivot(index='uid', columns='loan_time_hours', values='loan_amount').reset_index()
        loan_hour_df = loan_hour_df.fillna(0)

        # 贷款统计特征
        statistic_df = df_loan.copy()
        statistic_df = statistic_df.groupby(['uid','month'],as_index=False).sum()
        statistic_df = statistic_df.pivot(index='uid', columns='month', values='loan_amount').reset_index()
        statistic_df = statistic_df.fillna(0)
        statistic_df['loan_min'] = statistic_df[[9,10,11]].apply(lambda x: x.min(),axis=1)
        statistic_df['loan_max'] = statistic_df[[9,10,11]].apply(lambda x: x.max(),axis=1)
        statistic_df['loan_sum'] = statistic_df[[9,10,11]].apply(lambda x: x.sum(),axis=1)
        statistic_df['loan_mean'] = statistic_df[[9,10,11]].apply(lambda x: x.mean(),axis=1)
        statistic_df['loan_std'] = statistic_df[[9,10,11]].apply(lambda x: x.std(),axis=1)
        statistic_df['loan_std'] = statistic_df['loan_std'].map(lambda x : 1/(1+x))

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
        df_loan['loanTime_weights'] = df_loan['loan_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_loan['loanTime_weights'] = df_loan['loanTime_weights'].map(lambda x: 1/(1e-6+x.days/30))
        # 还款权重
        df_loan['loan_weights'] = df_loan['loan_amount'] * df_loan['loanTime_weights']
        # 12月累计需要还款金额
        month_9 = df_loan[df_loan['month']==9]
        month_9['repay'] = month_9['loan_amount']/month_9['plannum']
        month_9['repay'][month_9['plannum'] < 3] = 0.

        month_10 = df_loan[df_loan['month']==10]
        month_10['repay'] = month_10['loan_amount']/month_10['plannum']
        month_10['repay'][month_10['plannum'] < 2] = 0.

        month_11 = df_loan[df_loan['month']==11]
        month_11['repay'] = month_11['loan_amount']/month_11['plannum']

        df_loan = pd.concat([month_9,month_10,month_11],axis=0,ignore_index=False)
        month_df = pd.get_dummies(df_loan['month'], prefix="month")
        df_loan = pd.concat([df_loan,month_df],axis=1)
        df_loan = df_loan.groupby(['uid'],as_index=False).sum()
        df_loan['loan_months'] = df_loan['month_11']+df_loan['month_9']+df_loan['month_10']
        df_loan['loan_12'] = df_loan['month_9']+df_loan['month_10']
        df_loan['loan_12'] = df_loan['loan_12'].map({0:0,1:0,2:1})
        df_loan['loan_13'] = df_loan['month_9']+df_loan['month_11']
        df_loan['loan_13'] = df_loan['loan_13'].map({0:0,1:0,2:1})
        df_loan['loan_23'] = df_loan['month_10']+df_loan['month_11']
        df_loan['loan_23'] = df_loan['loan_23'].map({0:0,1:0,2:1})
        df_loan['loan_123'] = df_loan['month_9']+df_loan['month_10']+df_loan['month_11']
        df_loan['loan_123'] = df_loan['loan_123'].map({0:0,1:0,2:0,3:1})

        del df_loan['month']
        del df_loan['month_9']
        del df_loan['month_10']
        del df_loan['month_11']

        df_loan['per_plannum'] = df_loan['plannum'] / df_loan['loan_months']
        df_loan['per_times_loan'] = df_loan['loan_amount'] /df_loan['loan_months']
        # 每月贷款金额是否超过初始额度
        df_limit = gen_user_feat()[['uid','limit']]
        df_loan = pd.merge(df_loan,df_limit,how='left',on='uid')
        df_loan['exceed_loan'] = df_loan['per_times_loan'] - df_loan['limit']
        def _map_num(x):
            if x >= 0:
                return 1
            else:
                return 0
        df_loan['exceed_loan'] = df_loan['exceed_loan'].map(lambda x : _map_num(x))
        del df_loan['limit']

        df_loan = pd.merge(df_loan,statistic_df[['uid','loan_min','loan_mean','loan_max','loan_sum','loan_std']], how='left',on='uid')
        df_loan = pd.merge(df_loan,plannum_df, how='left',on='uid')
        df_loan = pd.merge(df_loan,loan_hour_df, how='left',on='uid')
        df_loan = df_loan.fillna(0)

        loan_cluster_label = user_cluster(df_loan,'test','loan')
        df_loan['loan_cluster_label'] = loan_cluster_label

        pickle.dump(df_loan, open(dump_path, 'wb'))
    return df_loan

def gen_order_feat():
    dump_path = './tmp/test_order_feat.pkl'
    if os.path.exists(dump_path):
        df_order = pickle.load(open(dump_path,'rb'))
    else:
        df_order = pd.read_csv(t_order_file,header=0)
        df_order['month'] = df_order['buy_time'].map(lambda x: conver_time(x))
        df_order = df_order[df_order['month']!=8]

        df_order['price'] = df_order['price'].map(lambda x: change_data(x))
        df_order['discount'] = df_order['discount'].map(lambda x: change_data(x))
        # 购买商品实际支付价格
        df_order['real_price'] = df_order['price']*df_order['qty'] - df_order['discount']
        df_order['real_price'][df_order['real_price']<0] = 0.

        statistic_df = df_order.copy()
        statistic_df = statistic_df.groupby(['uid','month'],as_index=False).sum()
        statistic_df = statistic_df.pivot(index='uid', columns='month', values='real_price').reset_index()
        statistic_df = statistic_df.fillna(0)
        statistic_df['buy_min'] = statistic_df[[9,10,11]].apply(lambda x: x.min(),axis=1)
        statistic_df['buy_max'] = statistic_df[[9,10,11]].apply(lambda x: x.max(),axis=1)
        statistic_df['buy_sum'] = statistic_df[[9,10,11]].apply(lambda x: x.sum(),axis=1)
        statistic_df['buy_mean'] = statistic_df[[9,10,11]].apply(lambda x: x.mean(),axis=1)
        statistic_df['buy_std'] = statistic_df[[9,10,11]].apply(lambda x: x.std(),axis=1)
        statistic_df['buy_std'] = statistic_df['buy_std'].map(lambda x : 1/(1+x))

        df_order['buy_weights'] = df_order['buy_time'].map(lambda x: datetime.strptime('2016-12-1', '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
        df_order['buy_weights'] = df_order['buy_weights'].map(lambda x: 0.1/(1e-6+x.days))
        df_order['cost_weight'] = df_order['real_price'] * df_order['buy_weights']
        df_order = df_order[['uid','buy_weights','cost_weight','real_price','discount']]
        df_order = df_order.groupby(['uid'],as_index=False).sum()
        # 购买商品的折扣率
        df_order['dis_ratio'] = df_order['discount'] / (df_order['discount'] + df_order['real_price'])
        del df_order['discount']
        df_order = pd.merge(df_order,statistic_df[['uid','buy_min','buy_mean','buy_max','buy_sum','buy_std']], how='left',on='uid')
        df_order = df_order.fillna(0)

        order_cluster_label = user_cluster(df_order,'test','order')
        df_order['order_cluster_label'] = order_cluster_label

        pickle.dump(df_order, open(dump_path, 'wb'))
    return df_order

def gen_filter_order_feat():
    dump_path = './tmp/test_filter_order_feat.pkl'
    if os.path.exists(dump_path):
        df_filter_order = pickle.load(open(dump_path,'rb'))
    else:
        df_filter_order = pd.read_csv(t_order_file,header=0)
        df_filter_order['month'] = df_filter_order['buy_time'].map(lambda x: conver_time(x))
        df_filter_order = df_filter_order[df_filter_order['month']!=8]
        df_filter_order['price'] = df_filter_order['price'].map(lambda x: change_data(x))
        df_filter_order['discount'] = df_filter_order['discount'].map(lambda x: change_data(x))
        # 购买商品实际支付价格
        df_filter_order['real_price'] = df_filter_order['price']*df_filter_order['qty'] - df_filter_order['discount']
        df_filter_order['real_price'][df_filter_order['real_price']<0] = 0.

        # 购买行为在滑动时间窗口内的购买总额
        df_filter_order['days'] = df_filter_order['buy_time'].map(lambda x: datetime.strptime('2016-12-1', '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
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
        column_list = list(click_hour_df.columns)
        column_list.remove('uid')
        for fea in column_list:
            click_hour_df[fea] = click_hour_df[fea].map(lambda x:math.log(x+1,5))

        df_click['click_weights'] = df_click['click_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
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
        statistic_df['click_min'] = statistic_df[[9,10,11]].apply(lambda x: x.min(),axis=1)
        statistic_df['click_max'] = statistic_df[[9,10,11]].apply(lambda x: x.max(),axis=1)
        statistic_df['click_sum'] = statistic_df[[9,10,11]].apply(lambda x: x.sum(),axis=1)
        statistic_df['click_mean'] = statistic_df[[9,10,11]].apply(lambda x: x.mean(),axis=1)
        statistic_df['click_std'] = statistic_df[[9,10,11]].apply(lambda x: x.std(),axis=1)
        statistic_df['click_std'] = statistic_df['click_std'].map(lambda x : 1/(1+x))

        del df_click['month']
        del df_click['click']
        df_click = df_click.groupby(['uid'],as_index=False).sum()
        df_click = pd.merge(df_click,statistic_df[['uid','click_min','click_max','click_mean','click_sum','click_std']], how='left',on='uid')
        df_click = pd.merge(df_click,click_hour_df,how='left',on='uid')

        click_cluster_label = user_cluster(df_click,'test','click')
        df_click['click_cluster_label'] = click_cluster_label

        pickle.dump(df_click, open(dump_path, 'wb'))
    return df_click

def gen_filter_click_feat():
    dump_path = './tmp/test_filter_click_feat.pkl'
    if os.path.exists(dump_path):
        df_filter_click = pickle.load(open(dump_path,'rb'))
    else:
        df_filter_click = pd.read_csv(t_click_file,header=0)

        df_filter_click['month'] = df_filter_click['click_time'].map(lambda x: conver_time(x))
        df_filter_click = df_filter_click[df_filter_click['month'] != 8]
        # 点击行为在滑动时间窗口内的点击次数
        df_filter_click['days'] = df_filter_click['click_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
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

def gen_interactive_feat():
    dump_path = './tmp/test_interactive_feat.pkl'
    if os.path.exists(dump_path):
        train_interactive_feat = pickle.load(open(dump_path,'rb'))
    else:
        df_loan_ori = pd.read_csv(t_loan_file,header=0)
        df_loan_ori['month'] = df_loan_ori['loan_time'].map(lambda x: conver_time(x))
        df_loan_ori = df_loan_ori[df_loan_ori['month'] != 8]
        df_loan_ori['days'] = df_loan_ori['loan_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_loan_ori['days'] = df_loan_ori['days'].map(lambda x: int(x.days))
        df_loan_ori['loan_amount'] = df_loan_ori['loan_amount'].map(lambda x: round(change_data(x)))

        df_order_ori = pd.read_csv(t_order_file,header=0)
        df_order_ori['month'] = df_order_ori['buy_time'].map(lambda x: conver_time(x))
        df_order_ori = df_order_ori[df_order_ori['month'] != 8]
        df_order_ori['days'] = df_order_ori['buy_time'].map(lambda x: datetime.strptime('2016-12-1', '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d'))
        df_order_ori['days'] = df_order_ori['days'].map(lambda x: int(x.days))
        df_order_ori['price'] = df_order_ori['price'].map(lambda x: change_data(x))
        df_order_ori['discount'] = df_order_ori['discount'].map(lambda x: change_data(x))
        df_order_ori['real_price'] = df_order_ori['price']*df_order_ori['qty'] - df_order_ori['discount']
        df_order_ori['real_price'][df_order_ori['real_price']<0] = 0.

        df_click_ori = pd.read_csv(t_click_file,header=0)
        df_click_ori['month'] = df_click_ori['click_time'].map(lambda x: conver_time(x))
        df_click_ori = df_click_ori[df_click_ori['month'] != 8]
        df_click_ori['days'] = df_click_ori['click_time'].map(lambda x: datetime.strptime('2016-12-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
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

        test_interactive_feat = pd.merge(loan_order,clcik_order,how='outer',on='uid')
        test_interactive_feat = pd.merge(test_interactive_feat,loan_click,how='outer',on='uid')
        test_interactive_feat = pd.merge(test_interactive_feat,loan_click_order,how='outer',on='uid')
        test_interactive_feat = test_interactive_feat.fillna(0)

        feature = list(test_interactive_feat.columns)
        feature.remove('uid')
        test_interactive_feat['interactive_min'] = test_interactive_feat[feature].apply(lambda x: x.min(),axis=1)
        test_interactive_feat['interactive_max'] = test_interactive_feat[feature].apply(lambda x: x.max(),axis=1)
        test_interactive_feat['interactive_sum'] = test_interactive_feat[feature].apply(lambda x: x.sum(),axis=1)
        test_interactive_feat['interactive_mean'] = test_interactive_feat[feature].apply(lambda x: x.mean(),axis=1)
        test_interactive_feat['interactive_std'] = test_interactive_feat[feature].apply(lambda x: x.std(),axis=1)
        test_interactive_feat['interactive_std'] = test_interactive_feat['interactive_std'].map(lambda x : 1/(1+x))


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

        test_interactive_week_feat = pd.merge(week_loan_order,week_click_order,how='outer',on='uid')
        test_interactive_week_feat = pd.merge(test_interactive_week_feat,week_loan_click,how='outer',on='uid')
        test_interactive_week_feat = pd.merge(test_interactive_week_feat,week_loan_click_order,how='outer',on='uid')
        test_interactive_week_feat = test_interactive_week_feat.fillna(0)

        feature = list(test_interactive_week_feat.columns)
        feature.remove('uid')
        test_interactive_week_feat['interactive_min'] = test_interactive_week_feat[feature].apply(lambda x: x.min(),axis=1)
        test_interactive_week_feat['interactive_max'] = test_interactive_week_feat[feature].apply(lambda x: x.max(),axis=1)
        test_interactive_week_feat['interactive_sum'] = test_interactive_week_feat[feature].apply(lambda x: x.sum(),axis=1)
        test_interactive_week_feat['interactive_mean'] = test_interactive_week_feat[feature].apply(lambda x: x.mean(),axis=1)
        test_interactive_week_feat['interactive_std'] = test_interactive_week_feat[feature].apply(lambda x: x.std(),axis=1)
        test_interactive_week_feat['interactive_std'] = test_interactive_week_feat['interactive_std'].map(lambda x : 1/(1+x))

        test_interactive_feat = pd.merge(test_interactive_feat,action_weeks_df,how='outer',on='uid')
        test_interactive_feat = pd.merge(test_interactive_feat,test_interactive_week_feat,how='outer',on='uid')
        test_interactive_feat = test_interactive_feat.fillna(0)
        pickle.dump(test_interactive_feat, open(dump_path, 'wb'))
    return test_interactive_feat

def make_test_set():
    dump_path = './data/test.pkl'
    if os.path.exists(dump_path):
        test_set = pickle.load(open(dump_path,'rb'))
    else:
        df_user = gen_user_feat()
        df_order = gen_order_feat()
        df_filter_order = gen_filter_order_feat()
        df_loan = gen_loan_feat()
        df_filter_loan = gen_filter_loan_feat()
        df_click = gen_click_feat()
        df_filter_click = gen_filter_click_feat()
        _,test_union_feat = gen_union_feat()
        df_interactive_feat = gen_interactive_feat()

        test_set = pd.merge(df_user, df_order, how='outer', on='uid')
        test_set = pd.merge(test_set, df_filter_order, how='outer', on='uid')
        test_set = pd.merge(test_set, df_loan, how='outer', on='uid')
        test_set = pd.merge(test_set, df_filter_loan, how='outer', on='uid')
        test_set['repay_cost'] = test_set['repay']+test_set['cost_weight']
        test_set = pd.merge(test_set, df_click, how='outer', on='uid')
        test_set = pd.merge(test_set, df_filter_click, how='outer', on='uid')
        test_set = pd.merge(test_set, test_union_feat, how='outer', on='uid')
        test_set = pd.merge(test_set, df_interactive_feat, how='outer', on='uid')
        test_set = test_set.fillna(0)
        
        change_list = ['cost_weight','real_price','buy_weights','loan_amount','click_weights','click_min','click_max','click_mean','click_sum','click_std',\
                        'per_times_loan','repay','repay_cost','loan_min','loan_max','loan_mean','loan_sum','loan_std','limit',\
                        'buy_min','buy_mean','buy_max','buy_sum','buy_std']
        for col in change_list:
            test_set[col] = test_set[col].map(lambda x : math.log(x+1,5))

        cluster_label = user_cluster(test_set,'test','all')
        test_set['cluster_label'] = cluster_label

        pickle.dump(test_set, open(dump_path, 'wb'))

        feat_id = {i:fea for i ,fea in enumerate(list(test_set.columns))}
        print(feat_id)
        # print(test_set.describe())

    return test_set

def make_clf_test_set():
    dump_path = './data/clf_test_data.pkl'
    if os.path.exists(dump_path):
        clf_test_data = pickle.load(open(dump_path,'rb'))
    else:
        df_user = gen_user_feat()
        df_order = gen_order_feat()
        df_click = gen_click_feat()
        df_filter_click = gen_filter_click_feat()
        _,test_union_feat = gen_union_feat()

        clf_test_data = pd.merge(df_user, df_order, how='outer', on='uid')
        clf_test_data = pd.merge(clf_test_data, df_click, how='outer', on='uid')
        clf_test_data = pd.merge(clf_test_data, df_filter_click, how='outer', on='uid')
        clf_test_data = pd.merge(clf_test_data, test_union_feat, how='outer', on='uid')
        clf_test_data = clf_test_data.fillna(0)

        change_list = ['cost_weight','real_price','click_weights',
                        ]
        for col in change_list:
            clf_test_data[col] = clf_test_data[col].map(lambda x : math.log(x+1,5))

        pickle.dump(clf_test_data, open(dump_path, 'wb'))

        feat_id = {i:fea for i ,fea in enumerate(list(clf_test_data.columns))}
        print(feat_id)
        # print(clf_test_data.describe())

    return clf_test_data

if __name__ == '__main__':
    # X_train,X_test,y_train,y_test = spilt_train_test()
    # print(X_train.shape)
    # print(X_train.columns)
    make_test_set()
    # make_clf_test_set()














