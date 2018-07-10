# -*- coding:utf-8 -*-

import itertools

import multiprocessing
import numpy as np
import MySQLdb
import threading
from multiprocessing import Process,Lock
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import time
import os
from functools import partial
from Queue import Queue

#######################################################################
# Description：
# 无多线程运行匹配循环
# Created with Python 2.7.
# User：wang wencong
# Date：2017/7/17
# Copyright© 2003-2016 Zhejiang huixin technology company
#######################################################################

# 列举2、3、4、5、6匹配模式
balllist = np.arange(1, 34)
balllist1 = np.arange(1, 10)
#528
#listb2 = list(itertools.combinations(balllist, 2))
# print len(listb2)
# #5456
listb3 = list(itertools.combinations(balllist, 3))
# print len(listb3)
# #40920
#listb4 = list(itertools.combinations(balllist, 4))
# print len(listb4)
#237336
#listb5 = list(itertools.combinations(balllist, 5))
# print len(listb5)
# #1107568
# listb6 = list(itertools.combinations(balllist, 6))
# print len(listb6)
# listb7 = list(itertools.combinations(balllist, 7))
# print len(listb7)
# listb8 = list(itertools.combinations(balllist, 8))
# print len(listb8)
# listb9 = list(itertools.combinations(balllist, 9))
# print len(listb9)
# listb10 = list(itertools.combinations(balllist, 10))
# print len(listb10)

try:
    conn = MySQLdb.connect(host='192.168.5.63', user='root', passwd='wwc@icinfo', port=3306)
    cur = conn.cursor()

    conn.select_db('ssq')

    count = cur.execute('select locateComp from ssqrealtime')
    print 'there has %s rows record' % count

    results = cur.fetchall()
    results = list(results)
    conn.commit()
    cur.close()
    conn.close()

except MySQLdb.Error, e:
    print "Mysql Error %d: %s" % (e.args[0], e.args[1])



# def main10(n):
#         poolb = ThreadPool(4)
#         listresult = []
#         listresult = list(poolb.map(listresult.append(list(listresult)), listb2))
#         poolb.close()
#         poolb.join()
#         def func(listresult):
#             l = listresult
#             def count_match(l):
#                     def list_match(i):
#                         count = 0
#                         listMatch = set(map(eval, results[i][0].split(" ")))
#                         resultMatch = set(l) & listMatch
#                         if len(resultMatch) == 2:
#                                 count = count + 1
#                         return count
#                     dictSort["三元匹配模式" + str(listresult) + "结果正确为："] = count
#                     return count_match
#             pool0 = ThreadPool(10)
#             count_match = count_match(l)
#             pool0.map(count_match, range(0, len(results)))
#             pool0.close()
#             pool0.join()
#
#
#         pool10 = ThreadPool(10)
#         pool10.map(func(listresult), listresult)
#         pool10.close()
#         pool10.join()

# def main2(n):
#     # 注意tp的三个位置，看结果放到哪里都一样
#     # 但是放到这里，两个子进程调用的tp对象是不同的
#     pool = ThreadPool(2)
#     for i in range(10):
#         res = pool.apply(sleep, (3, i))    #非阻塞
#         pool.close()
#         pool.join()
#
# def main5(n):
#     # 注意tp的三个位置，看结果放到哪里都一样
#     # 但是放到这里，两个子进程调用的tp对象是不同的
#     pool = ThreadPool(5)
#     for i in range(10):
#         res = pool.apply(sleep, (3, i))    #非阻塞
#         pool.close()
#         pool.join()
# start = time.clock()
# pool = ThreadPool(5)
# # for i in range(10):
# #     # res = pool.apply(sleep, (3, i))    #非阻塞
# #     # sleep(3, i)
# res = []
# res = pool.map(sleep, range(10))
# pool.close()
# pool.join()
# print "%s %s" %("2线程测试耗时", time.clock() - start), "second"
dictSort = {}


listresult = []
global listduplic
listduplic = []

global list_duplic
list_duplic = []

# lock = Lock()
# lockOne = threading.Lock()

def list_result(listresult):
    poolb = ThreadPool(4)
    listresult = list(poolb.map(listresult.append(list(listresult)), listb3))
    poolb.close()
    poolb.join()
    return listresult

# def list_match(l):
#     count = 0
#     for i in range(0, len(results)):
#         listMatch = set(map(eval, results[i][0].split(" ")))
#         resultMatch = set(l) & listMatch
#         if len(resultMatch) == 2:
#             count = count + 1
#             continue
#         else:
#             continue
#     else:
#         dictSort["五元匹配模式" + str(l) + "结果正确为："] = count
#     return list_match
#
#
#
# count1 = 0
# def count_num():
#     global count1
#     count1 = count1 + 1
#     print count1


def match_lr(n, lr, qresult):
    global listduplic
    qmatch = qresult
    resultMatch = set(lr[n]) & qmatch
    if len(resultMatch) == 2:
        listduplic.append(lr[n])
    return listduplic



def match_query(n, lr):
    listresult = lr
    pool_lr = ThreadPool(10)
    listMatch = set(map(eval, results[n][0].split(" ")))
    partial_match_lr = partial(match_lr, lr=listresult, qresult=listMatch)

    r= pool_lr.map(partial_match_lr,  range(0, len(lr)))

    pool_lr.close()
    pool_lr.join()
    return r


def test_match():

    pool = ThreadPool(processes=8)
    partial_match_query = partial(match_query, lr=listresult)
    pool.map(partial_match_query,  range(0, len(results)))
    pool.close()
    pool.join()


#
# def main10(n):
#     list_result()
#     pool0 = Pool(processes=2)
#     pool0.map(list_match(listresult), listresult)
#     pool0.close()
#     pool0.join()
if __name__ == '__main__':
        # tp = ThreadPoolExecutor(max_workers=10)

    listresult = list_result(listresult)
    start = time.clock()
    pool = ThreadPool(8)
    partial_match_query = partial(match_query, lr=listresult)

    pool.map(partial_match_query,  range(0, len(results)))

    pool.close()
    pool.join()

    print listduplic

    print "%s %s" %("2线程测试耗时", time.clock() - start), "second"
    print 1
        #time.sleep(5)    #这个时间比3大就行



