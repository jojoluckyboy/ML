# !/usr/bin/python
# -*- coding:utf-8 -*-

import urllib2
import re

import MySQLdb
import pandas_profiling

import pymysql
import sqlalchemy
from urllib import quote_plus as urlquote

from sqlalchemy import create_engine
import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib.pyplot import GridSpec, figure
from pandas.plotting import scatter_matrix
import seaborn as sns
import numpy as np
import pandas as pd
import missingno as msno


#######################################################################
# Description：
# 三D魔图状态与其余指标值之间的数据EDA
# Created with Python 2.7.
# User：wang wencong
# Date：2018/7/3
# Copyright© 2003-2016 Zhejiang huixin technology company
#######################################################################
# 1. 用sqlalchemy构建数据库链接engine
connect_info = "mysql+mysqldb://root:%s@192.168.5.63:3306/sandp?charset=utf8" % urlquote('wwc@icinfo')

engine = create_engine(connect_info, max_overflow=5)
# sql 命令
sql_cmd = "SELECT * FROM sandp.sandhistory"

sql_cmd1 = "SELECT * FROM sandp.sanddankill WHERE issue>='2018001' "

# 获取SandHistory表数据
resdf = pd.read_sql(sql=sql_cmd, con=engine)
sanDkilldf = pd.read_sql(sql=sql_cmd1, con=engine)


if __name__ == '__main__':
    pfr = pandas_profiling.ProfileReport(sanDkilldf)
    pfr.to_file("E:/example.html")


