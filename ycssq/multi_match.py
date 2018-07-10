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


#######################################################################
# Description：
# 多进程多线程大数据数组
# Created with Python 2.7.
# User：wang wencong
# Date：2017/8/17
# Copyright© 2003-2016 Zhejiang huixin technology company
#######################################################################

# 列举2、3、4、5、6匹配模式
balllist = np.arange(1, 34)
balllist1 = np.arange(1, 10)
#528
listb2 = list(itertools.combinations(balllist, 2))
# print len(listb2)
# #5456
# listb3 = list(itertools.combinations(balllist, 3))
# print len(listb3)
# # #40920
# listb4 = list(itertools.combinations(balllist, 4))
# # print len(listb4)
# #237336
#listb5 = list(itertools.combinations(balllist, 5))
# # print len(listb5)
# # #1107568
#listb6 = list(itertools.combinations(balllist, 6))
# # print len(listb6)
listb7 = list(itertools.combinations(balllist, 7))
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

    count = cur.execute('select redComp from ssqrealtime')
    print 'there has %s rows record' % count

    results = cur.fetchall()
    results = list(results)
    conn.commit()
    cur.close()
    conn.close()

except MySQLdb.Error, e:
    print "Mysql Error %d: %s" % (e.args[0], e.args[1])

dictSort = {}

listresult = []

global listduplic
listduplic = []

counter = []
lock = Lock()
# lockOne = threading.Lock()

def list_result(listresult):
    poolb = ThreadPool(8)
    listresult = list(poolb.map(listresult.append(list(listresult)), listb2))
    poolb.close()
    poolb.join()
    return listresult




def match_lr(n, lr, qresult,listduplic):

    qmatch = qresult
    resultMatch = set(lr[n]) & qmatch
    with lock:
        if len(resultMatch) == 6:
            listduplic.append(lr[n])
    return listduplic



def match_query(n, lr, queue):
    global lock, counter
    listresult_param = lr
    pool_lr = ThreadPool(10)
    listMatch = set(map(eval, results[n][0].split(" ")))
    partial_match_lr = partial(match_lr, lr=listresult_param, qresult=listMatch, listduplic=queue)

    counter= pool_lr.map(partial_match_lr,  range(0, len(lr)))

    pool_lr.close()
    pool_lr.join()

    return queue

if __name__ == '__main__':
        # tp = ThreadPoolExecutor(max_workers=10)
    manager = Manager()
    q_list = manager.list()
    listresult = list_result(listresult)


    start = time.clock()
    pool = Pool(processes=8)
    partial_match_query = partial(match_query, lr=listresult, queue=q_list)

    pool.map(partial_match_query,  range(0, len(results)))

    pool.close()
    pool.join()
    print "%s %s" %("8进程测试耗时", time.clock() - start), "second"
    dic_list = list(q_list)
    dic_result = {}
    dic_result = sorted(dict(Counter(dic_list)).items(), key=lambda x: x[1], reverse=True)

    origin = sys.stdout
    f = open('E:/file.txt', 'w')
    sys.stdout = f
    for item in range(0, len(dic_result)):

        print ("六元匹配模式"+str(dic_result[item][0])+"结果正确次数为："+str(dic_result[item][1]))

    sys.stdout = origin
    f.close()
