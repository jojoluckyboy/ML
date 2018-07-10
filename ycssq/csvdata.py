# -*- coding:utf-8 -*-

import itertools
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
import time
#######################################################################
# Description：
# csv大数据文件分块读取写出
# Created with Python 2.7.
# User：wang wencong
# Date：2017/8/17
# Copyright© 2003-2016 Zhejiang huixin technology company
#######################################################################
# balllist = np.arange(1, 34)
# listb9 = list(itertools.combinations(balllist, 9))
# head = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
# data = pd.DataFrame(listb9, columns=head)
# start = time.clock()
# poolb = ThreadPool(8)
# poolb.map(data.to_csv('E:\\data\\nine.csv', sep=' ', index=False, header=True, mode='a'), listb9)
# poolb.close()
# poolb.join()
# print "%s %s" %("8进程测试耗时", time.clock() - start), "second"


# balllist = np.arange(1, 34)
# listb10 = list(itertools.combinations(balllist, 10))
# head = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
# data = pd.DataFrame(listb10, columns=head)
# start = time.clock()
# poolb = ThreadPool(8)
# poolb.map(data.to_csv('E:\\data\\ten.csv', sep=' ', index=False, header=True, mode='a'), listb10)
# poolb.close()
# poolb.join()
# print "%s %s" %("8进程测试耗时", time.clock() - start), "second"

#11位大数据写出
listb = itertools.combinations(xrange(1, 34), 18)

head = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18"]
# head = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"]
#head = ["C1", "C2"]
listappend = []
i = 0
data = pd.DataFrame(listappend, columns=head)
data.to_csv('E:\\data\\eighteen1.csv', sep=' ', header=True, mode='a')
try:
    for item in listb:

        listappend.append(item)
        i = i+1
        if i == 20000000:
            data = pd.DataFrame(listappend, columns=head)
            start = time.clock()
            poolb = ThreadPool(8)
            poolb.map(data.to_csv('E:\\data\\eighteen1.csv', sep=' ', index=False, header=False, mode='a'), listappend)
            poolb.close()
            poolb.join()
            i = 0
            print "%s %s" %("8进程测试耗时", time.clock() - start), "second"
            del listappend[:]
        if item.__str__() == '(12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)':
            data = pd.DataFrame(listappend, columns=head)
            start = time.clock()
            poolb = ThreadPool(8)
            poolb.map(data.to_csv('E:\\data\\eighteen1.csv', sep=' ', index=False, header=False, mode='a'), listappend)
            poolb.close()
            poolb.join()

            print "%s %s" %("8进程测试耗时", time.clock() - start), "second"
            break
except StopIteration, e:
       print "StopIteration %d: %s" % (e.args[0], e.args[1])
