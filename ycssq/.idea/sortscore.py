# -*- coding:utf-8 -*-

import itertools

import numpy as np
import MySQLdb
from multiprocessing import Process,Lock
from multiprocessing import Pool, Manager, Lock
from multiprocessing.dummy import Pool as ThreadPool
import time
import sys
from functools import partial
from collections import Counter




# 列举2、3、4、5、6匹配模式
import pandas as pd

# if __name__ == '__main__':
#
#     dic_list = {}
#     dic_result = {}
#     reader = pd.read_csv('E:\\eryuan.txt',header=None,sep='：',dtype=str)
#     dic_list = dict(np.array(reader).tolist())
#
#     dic_result=sorted(dic_list.items(), key=lambda x: x[1], reverse=True)
#     origin = sys.stdout
#     f = open('E:/二元位置匹配结果.txt', 'a')
#     sys.stdout = f
#     for item in range(0,21):
#
#         print (str(dic_result[item][0])+"："+str(dic_result[item][1]))
#
#     sys.stdout = origin
#     f.flush()
#     f.close()

if __name__ == '__main__':

    dic_list = {}
    dic_result = {}
    reader = pd.read_csv('E:\\eryuanred.txt',header=None,sep='：',dtype=str)
    b = reader.sort_values(by=1,ascending=False).head(20)
    print b

    # dic_list = dict(np.array(reader).tolist())
    #
    # dic_result=sorted(dic_list.items(), key=lambda x: x[1], reverse=True)
    # origin = sys.stdout
    # f = open('E:/二元位置匹配结果.txt', 'a')
    # sys.stdout = f
    # for item in range(0,21):
    #
    #     print (str(dic_result[item][0])+"："+str(dic_result[item][1]))
    #
    # sys.stdout = origin
    # f.flush()
    # f.close()