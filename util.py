#encoding:utf-8
import math
import pandas as pd


def change_data(x):
    return round(math.pow(5,x)-1)

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

def conver_time(time):
    return int(time.split('-')[1])

def get_stat_feat(df,values,action,days1,days2):
    df = df[df['days'] > days1]
    df = df[df['days'] <= days2]
    stat_feat = ['min','mean','max','median','count','sum','std']
    df = df.groupby('uid')[values].agg(stat_feat).reset_index()
    df.columns = ['uid'] + ['%s_%s_' % (action,days2) + col for col in stat_feat] #loan_7_min,loan_7_max

    return df










