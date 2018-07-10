# -*- coding:utf-8 -*-
import itertools

import numpy as np
import pandas as pd
import MySQLdb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score

#######################################################################
# Description：
# 数据读库加逻辑回归验证
# Created with Python 2.7.
# User：wang wencong
# Date：2017/8/17
# Copyright© 2003-2016 Zhejiang huixin technology company
#######################################################################

# a = [[2,2,1],[3,2,1],[0,3,1]]
# b = [0, 1, 1]
# for i in range(0, 128):
#     c = bin(i)
#     d =c.split('b', 1)[1]
#     if len(d) == 1:
#         d = '000000' + str(d)
#     if len(d) == 2:
#         d = '00000' + str(d)
#     if len(d) == 3:
#         d = '0000' + str(d)
#     if len(d) == 4:
#         d = '000' + str(d)
#     if len(d) == 5:
#         d = '00' + str(d)
#     if len(d) == 6:
#         d = '0' + str(d)
#
#     print d
# logreg = LogisticRegression(C=1e5)
# b_hat = logreg.predict(a)
# b_hat_prob = logreg.predict_proba(a)
# logreg.fit(a, b)
#
# b_hat = logreg.predict(a)
# b_hat_prob = logreg.predict_proba(a)
# np.set_printoptions(suppress=True)
#
# print 'y_hat = \n', b_hat
# print 'y_hat_prob = \n', b_hat_prob
#
# print '准确率： ', accuracy_score(b, b_hat)

# data = np.loadtxt('hmmodletest.data', dtype=str, delimiter=',')
# print data.size/2
# QQ = ['0000111', '0001111', '0010111', '0011011', '0011111', '0100111', '0101011', '0101111', '0110111', '0111011',
#       '0111101', '0111111', '1001111', '1010111', '1011011', '1011101', '1011111', '1100111', '1101101', '1101111',
#       '1110111', '1111011', '1111101', '1111111']
# QR = ['0001101', '0001110', '0011110', '0101110', '0110110', '0111010', '0111100', '0111110', '1001110', '1010110',
#       '1011110', '1101110', '1110011', '1111001', '1111100', '1111110']
# QP = ['0001011', '0011101', '0100011', '0100101', '0101101', '0110011', '0110101', '0111001', '1000111', '1001011',
#       '1001101', '1010011', '1100011', '1100101', '1101001', '1101011', '1110101', '1110110', '1111010']
# RP = ['0000110', '0001010', '0010011', '0010110', '0011010', '0011100', '0100110', '0101010', '0101100', '0110010',
#       '0110100', '1000110', '1001010', '1001100', '1010010', '1010101', '1011010', '1011100', '1100110', '1101010',
#       '1101100', '1110010', '1110100']
# RQ = ['0000001', '0000011', '0000101', '0001001', '0010001', '0010101', '0011001', '0100001', '0101001', '0110001',
#       '1000001', '1000011', '1000101', '1001001', '1010001', '1011001', '1100001', '1110001']
# RR = ['0000000', '0000010', '0000100', '0001000', '0001100', '0010000', '0010010', '0010100', '0011000', '0100000',
#       '0100010', '0100100', '0101000', '0110000', '0111000', '1000000', '1000010', '1000100', '1001000', '1010000',
#       '1010100', '1011000', '1100000', '1100010', '1100100', '1101000', '1110000', '1111000']
#
# for i in range(7, data.size/2):
#     datact = ""
#     for j in range(i-7, i):
#          datact = datact + str(data[j][0])
#
#     if datact in QQ:
#         print "QQ"
#     if datact in QR:
#         print "QR"
#     if datact in QP:
#         print "QP"
#     if datact in RP:
#         print "RP"
#     if datact in RQ:
#         print "RQ"
#     if datact in RR:
#         print "RR"

Str1 = "0000000,0000100,0001100,0101100,0101000,1000100,1001000,1010100,1011000,1101000,1101100,1110000,1110100"
Str1.split(',')
Str2 = ""
s = "0"
for i in range(0, len(Str1.split(','))):

    Str2 =Str2+"\""+s+Str1.split(',')[i]+"\""+","

print Str2


