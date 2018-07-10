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
sql_cmd = "SELECT mtstat FROM sandp.sandhistory"

sql_cmd1 = "SELECT baihund,shidecade,gesingle,sjbaihund,sjshidecade,sjgesingle,sanDV,shijiDV,sanDgc,sanDsum," \
           "sanDkd,flagContain,flagBai,shijiDsum,shijiDkd FROM sandp.sandhistory "

# 获取SandHistory表数据
df1 = pd.read_sql(sql=sql_cmd, con=engine)
df2 = pd.read_sql(sql=sql_cmd1, con=engine)


# 获取SandHistory表数据

#%获取mtstat从第二期到最后，相应的其他状态值从第一到倒数第二，每次以前一天的值预测后一天的mtstat %

df1 = df1.iloc[1:].reset_index(drop=True)
df2 = df2.iloc[:-1].reset_index(drop=True)
df = pd.concat([df1, df2], axis=1)


#对数据的各个指标的类型做标准化转换
df['baihund'] = df['baihund'].astype('int64')
df['shidecade'] = df['shidecade'].astype('int64')
df['gesingle'] = df['gesingle'].astype('int64')
df['sjbaihund'] = df['sjbaihund'].astype('int64')
df['sjshidecade'] = df['sjshidecade'].astype('int64')
df['sjgesingle'] = df['sjgesingle'].astype('int64')

df['sanDV'] = df['sanDV'].astype('int64')
df['shijiDV'] = df['shijiDV'].astype('int64')
df['flagContain'] = df['flagContain'].astype('int64')
df['flagBai'] = df['flagBai'].astype('int64')

#%获取所有column的list %
all_columns = df.columns.tolist()
#print all_columns
#print ", ".join(all_columns)

#统计多变量，多特征的图%
fig, ax = plt.subplots(figsize=(10, 10))
scatter_matrix(df[['mtstat', 'baihund', 'shidecade', 'gesingle', 'sjbaihund', 'sjshidecade', 'sjgesingle', 'sanDV',
                   'shijiDV', 'sanDgc', 'sanDsum', 'sanDkd', 'flagContain', 'flagBai', 'shijiDsum', 'shijiDkd']],
                alpha=0.2, diagonal='hist', ax=ax)
plt.show()

"""
#%%

#对数据的各个指标的类型做标准化转换
#print resdf.dtypes

resdf['baihund'] = resdf['baihund'].astype('int64')
resdf['shidecade'] = resdf['shidecade'].astype('int64')
resdf['gesingle'] = resdf['gesingle'].astype('int64')
resdf['sjbaihund'] = resdf['sjbaihund'].astype('int64')
resdf['sjshidecade'] = resdf['sjshidecade'].astype('int64')
resdf['sjgesingle'] = resdf['sjgesingle'].astype('int64')
resdf['xingtai3D'] = resdf['xingtai3D'].astype('int64')
resdf['xingtaisj'] = resdf['xingtaisj'].astype('int64')

resdf['sanDV'] = resdf['sanDV'].astype('int64')
resdf['shijiDV'] = resdf['shijiDV'].astype('int64')
resdf['flagContain'] = resdf['flagContain'].astype('int64')
resdf['flagBai'] = resdf['flagBai'].astype('int64')



#%对数据的各个指标做初略统计 %
print resdf.describe().T


#%获取所有column的list %
all_columns = resdf.columns.tolist()
print all_columns
print ", ".join(all_columns)

#%看单个变量的统计值%

print "--------------------单变量均值-----------"
print resdf['sanDV'].mean()

#%如果issue有重复,平均值会略有不同，groupby去掉重复值%
print np.mean(resdf.groupby(['issue']).sanDV.mean())

print "--------------------单变量均值-----------"
grouped = resdf.groupby(['flagContain'])['sanDV'].mean()
print grouped


#%数据分块切片%
#魔图状态与试机号信息
shiji_index = 'issue'
shiji_cols = {
    'issue',
    'sjbaihund',
    'sjshidecade',
    'sjgesingle',
    'xingtaisj',
    'shijiDV',
    'mtstat',
    'flagContain',
    'flagBai',
    'shijiDsum',
    'shijiDkd'
    }

#%校验是否数据清洗正确，正确的话每个指标数是1，大于1代表有重复数据%
all_columns_unique_shiji = resdf.groupby('issue').agg({col:'nunique' for col in shiji_cols})
print all_columns_unique_shiji.head()
print all_columns_unique_shiji[all_columns_unique_shiji > 1].dropna().shape[0] == 0

#%定制检测数据清洗结果的功能函数%
def get_dataclean(dataframe, g_index, g_colums):
    g = dataframe.groupby(g_index).agg({col:'nunique' for col in g_colums})
    if g[g>1].dropna().shape[0] !=0:
        print ("Warning: you probably assumed this had all unique values but it doesn't. ")
    return dataframe.groupby(g_index).agg({col:'max' for col in g_colums})

shiji = get_dataclean(resdf,shiji_index,shiji_cols)
print ("通过维度切分后数据验证没问题，输出：")
print shiji.head()

#%定制存储，切分校验过清洗的功能函数%
def save_subgroup(dataframe, g_index, subgroup_name, prefix = 'raw_'):
    save_subgroup_filename = "".join([prefix, subgroup_name, ".csv.gz"])
    dataframe.to_csv('E:/'+save_subgroup_filename, compression='gzip', encoding='UTF-8')
    test_df = pd.read_csv('E:/'+save_subgroup_filename, compression='gzip', index_col=g_index, encoding='UTF-8')
    # Test that we recover what we send in
    if dataframe.equals(test_df):
        print ("Test-passed: we recover the equivalent subgroup dataframe.")
    else:
        print ("Warning -- equivalence test!!! Double-check.")

save_subgroup(shiji, shiji_index, "shijihao")

#%定制双变量对应关系的数据集%

dyad_index = ['issue', 'expertname']
dyad_cols = {
    'sanDV',
    'singDan',
    'singDanhit',
    'spicerock',
    'singDankd',
    'fiveDanV',
    'fiveDanhit',
    'fiveDandiv',
    'fiveDanskewn',
    'fiveDanacV',
    'sumDanV',
    'sumDanhit',
    'sumDandiv',
    'sumDanskewn',
    'sumDanacV',
    'kuaDanV',
    'kuaDanhit',
    'kuaDandiv',
    'kuaDanskewn',
    'kuaDanacV',
    'mtstat',
    }

dyads = get_dataclean(sanDkilldf, g_index=dyad_index,g_colums=dyad_cols)
#print dyads.head()
save_subgroup(dyads, dyad_index, "sanDkill")
"""
"""
#%定制从外部压缩文件读取数据%
def load_subgroup(filename, index_col = [0]):
    return pd.read_csv(filename, compression='gzip', index_col=index_col)

sanDkill = load_subgroup('E:/raw_sanDkill.csv.gz')
print sanDkill.shape
#print sanDkill.head

#%校验缺失值%
msno.matrix(sanDkill.sample(1000), figsize=(12, 5), width_ratios=(10, 1),)
plt.show()

msno.bar(sanDkill.sample(1000), figsize=(12, 5),)
plt.show()

#msno.heatmap(sanDkill.sample(1000))
#plt.show()

#%统计缺失值%
sanDkill_lack = sanDkill[sanDkill.sanDV.notnull()]
print sanDkill_lack.shape[0]

#%定制观测两个特征之间的关系%
print pd.crosstab(sanDkill.singDan, sanDkill.spicerock)


#%seaborn画热度图%
plt.figure(figsize=(12, 8))
ax = sns.heatmap(pd.crosstab(sanDkill.singDan, sanDkill.spicerock), cmap='Blues', annot=True, fmt='d', linewidths=.5)
ax.set_title("Correlation between singDan and spicerock")
plt.show()

#%dataframe添加一个字段%
sanDkill['sumadd'] = sanDkill[['sumDanskewn', 'kuaDanacV']].mean(axis=1)
print sanDkill.head()

#%单个指标分布的柱形图%
sns.distplot(sanDkill.spicerock, kde=False)
plt.show()

#%指标值进行值域分类%
singDan_type = sanDkill.singDan.unique()
print singDan_type

low_value = ['0', '1', '2', '3']
mid_value = ['4', '5', '6', '7']
high_value = ['8', '9', '10']
null_value = '--'

#%修改dataframe%
sanDkill.loc[sanDkill.singDan.isin(low_value), 'singDan_agg'] = 'low_value'
sanDkill.loc[sanDkill.singDan.isin(mid_value), 'singDan_agg'] = 'mid_value'
sanDkill.loc[sanDkill.singDan.isin(high_value), 'singDan_agg'] = 'high_value'
sanDkill.loc[sanDkill.singDan.eq(null_value), 'singDan_agg'] = 'null_value'

#%按照值分类画plt图%
fig, ax = plt.subplots(figsize=(12, 8))
sanDkill.singDan_agg.value_counts(dropna=False, ascending=True).plot(kind='barh', ax=ax)
ax.set_ylabel('singDan_agg')
ax.set_xlabel('Counts')
fig.tight_layout()
plt.show()


#按照bins分块数据%
mtstat_categories = ["vlow_value", "low_value", "mid_value"
                     , "high_value", "vhigh_value"]

sanDkill['mtstateclass'] = pd.qcut(sanDkill['mtstat'], len(mtstat_categories), mtstat_categories)
print sanDkill.head

#统计多变量，多特征的图%
fig, ax = plt.subplots(figsize=(10, 10))
scatter_matrix(sanDkill[['singDan', 'spicerock', 'singDanhit', 'singDankd']], alpha=0.2, diagonal='hist', ax=ax)
plt.show()

if __name__ == '__main__':
    pfr = pandas_profiling.ProfileReport(sanDkill)
    pfr.to_file("./example.html")
"""

