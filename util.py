#encoding:utf-8
import math


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