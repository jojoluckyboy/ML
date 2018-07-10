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
balllist = np.arange(1, 50)
balllist1 = np.arange(1, 10)
#528
listb2 = list(itertools.combinations(balllist, 3))
print len(listb2)
# #5456
# listb3 = list(itertools.combinations(balllist, 3))
# print len(listb3)
# # #40920
# listb4 = list(itertools.combinations(balllist, 4))
# # print len(listb4)
# #237336
# listb5 = list(itertools.combinations(balllist, 5))
# # print len(listb5)
# # #1107568
# listb6 = list(itertools.combinations(balllist, 6))
# # print len(listb6)
# listb7 = list(itertools.combinations(balllist, 7))
# print len(listb7)
listb8 = list(itertools.combinations(balllist, 8))
# print len(listb8)
# listb9 = list(itertools.combinations(balllist, 9))
# print len(listb9)
# listb10 = list(itertools.combinations(balllist, 10))
# print len(listb10)

listresult = []
def list_result(listresult):
    poolb = ThreadPool(2)
    listresult = list(poolb.map(listresult.append(list(listresult)), listb2))
    poolb.close()
    poolb.join()
    return listresult



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

if __name__ == '__main__':
        # tp = ThreadPoolExecutor(max_workers=10)
    start = time.clock()
    listresult = list_result(listresult)
    print len(listresult)
    for i in range(1, 11):
        x = 1
        print x


    print "%s %s" %("2线程测试耗时", time.clock() - start), "second"
    #time.sleep(5)    #这个时间比3大就行



# if __name__ == '__main__':
#     # tp = ThreadPoolExecutor(max_workers=10)
#     p2 = Pool(processes=2)
#     start = time.clock()
#     p2.map(main2, (1, 2))    #阻塞，要等待所有子进程都执行完毕
#     print "%s %s" %("5线程测试耗时", time.clock() - start), "second"
#
# if __name__ == '__main__':
#     # tp = ThreadPoolExecutor(max_workers=10)
#     p3 = Pool(processes=2)
#     start = time.clock()
#     p3.map(main2, (1, 2))    #阻塞，要等待所有子进程都执行完毕
#     print "%s %s" %("10线程测试耗时", time.clock() - start), "second"