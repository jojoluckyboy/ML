# -*- coding:utf-8 -*-
import itertools
import sys
import numpy as np
import pandas as pd
import MySQLdb
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

#######################################################################
# Description：
# 数据读库加逻辑回归验证
# Created with Python 2.7.
# User：wang wencong
# Date：2017/8/17
# Copyright© 2003-2016 Zhejiang huixin technology company
#######################################################################


reload(sys)
sys.setdefaultencoding('utf8')

def time_trans(issuedate):
    timedata = np.loadtxt('timearray.data', dtype=str, delimiter=',')
    time_trans = issuedate
    for i in range(0,timedata.size):
        if issuedate == timedata[i:(i+1), [0]].tostring():
            time_trans = timedata[i:(i+1), [1]].tostring()
            break
    return time_trans

data = np.loadtxt('expert.data', dtype=str, delimiter=',')
expertlist = ""
for i in range(0,data.size):

    if i == data.size-1:
        expertlist += "\""+data[i].tostring()+"\""
    else:
        expertlist += "\""+data[i].tostring()+"\"" +", "

try:
        conn = MySQLdb.connect(host='192.168.5.63', user='root', passwd='wwc@icinfo', port=3306, charset='utf8')
        cur = conn.cursor()

        conn.select_db('ssq')

        # count = cur.execute('select issue,expertname,redthree,lasthit,lastscore,formax,oddeven,smallbig,primecom,p51,p52,p53,'
        #                     'p101,p102,p103,p201,p202 ,p203,p301,p302,p303,p501,p502,p503,p1001,p1002,p1003,nowhit,nowscore '
        #                     'from ssqthred WHERE expertname = "一码当先"')
        specialdate = ['2014001', '2015001', '2016001', '2017001']
        for i in range(0, data.size):


                sqlqry= "select issue from ssqthred WHERE expertname =\""+data[i].tostring()+"\""
                count = cur.execute(sqlqry)
                issuers = cur.fetchall()
                issuers = list(issuers)

                sqlqryY = "select nowhit from ssqthred WHERE expertname =\""+data[i].tostring()+"\" order by expertname desc"
                count = cur.execute(sqlqryY)
                yresult = cur.fetchall()
                yresult = list(yresult)
                pddYR = pd.DataFrame(yresult, columns=["nowhit"], dtype=str)
                pddYR = pddYR.iloc[1:]
                pddYR.to_csv ("E:/LGY.csv", encoding="utf-8")
                pddR = pd.DataFrame(data, columns=["expertname"], dtype=str)


                for elemtime in issuers:
                    if elemtime[0] in specialdate:
                        timeres = time_trans(elemtime[0])
                    else:
                        timeres = str(int(elemtime[0])-1)

                    sqlT = "SELECT t1.expertname,t2.nowhit FROM (SELECT expertname FROM ssqthred WHERE expertname IN ("\
                           +expertlist+") GROUP BY expertname order by expertname desc ) AS t1 LEFT JOIN (SELECT expertname AS ep,nowhit  FROM" \
                           " ssqthred WHERE expertname IN ("+expertlist+") AND issue = " + timeres+") AS t2 ON t1.expertname = t2.ep"

                    count = cur.execute(sqlT)
                    resT = cur.fetchall()
                    pdd = pd.DataFrame(list(resT), columns=["expertname", timeres], dtype=str)


                # print pdd1
                    pddR = pd.merge(pddR, pdd, on='expertname',how='right')
                    timend = str(int(elemtime[0]))


                # sqlPlus = "SELECT t1.expertname,t2.nowhit FROM (SELECT expertname FROM ssqthred WHERE expertname IN ("\
                #            +expertlist+") GROUP BY expertname) AS t1 LEFT JOIN (SELECT expertname AS ep,nowhit  FROM" \
                #            " ssqthred WHERE expertname IN ("+expertlist+") AND issue = " + timend+") AS t2 ON t1.expertname = t2.ep"
                #
                # count = cur.execute(sqlPlus)
                # resPlus = cur.fetchall()
                # pdd = pd.DataFrame(list(resPlus), columns=["expertname", timend], dtype=str)
                # # print pdd1
                # pddR = pd.merge(pddR, pdd)
                pddX = pddR.T.iloc[2:]
                pddX = pddX.fillna(str(3))

                pddX.to_csv ("E:/LGXV.csv", encoding="utf-8")

                x = np.array(pddX)
                y = np.array(pddYR)
                print x
                print y
                x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

                 # Logistic回归
                lr = LogisticRegression(penalty='l2')
                lr.fit(x_train, y_train.ravel())
                y_hat = lr.predict(x_test)

                origin = sys.stdout
                f = open('E:/MLResult.txt', 'w')
                sys.stdout = f
                print data[i].tostring(), 'Logistic回归正确率：', accuracy_score(y_test, y_hat)
                sys.stdout = origin
                f.close()


                 #随机森林调最优参数
                model = RandomForestClassifier(n_estimators=60, criterion='entropy', max_depth=7, min_samples_leaf=1)
                # model = GridSearchCV(rf, param_grid={'max_depth': np.arange(2, 10), 'min_samples_leaf': (1, 10)})
                model.fit(x_train, y_train.ravel())
                # print '最优参数： ', model.best_params_
                y_train_pred = model.predict(x_train)   # 训练数据
                y_test_hat = model.predict(x_test)      # 测试数据

                origin = sys.stdout
                f = open('E:/MLResult.txt', 'w')
                sys.stdout = f
                print data[i].tostring(), '随机森林训练集准确率：', accuracy_score(y_train, y_train_pred)
                print data[i].tostring(), '随机森林测试集准确率:', accuracy_score(y_test, y_test_hat)
                sys.stdout = origin
                f.close()


                # XGBoost
                # y_train[y_train == 3] = 0
                # y_test[y_test == 3] = 0
                data_train = xgb.DMatrix(x_train, label=y_train)
                data_test = xgb.DMatrix(x_test, label=y_test)
                watch_list = [(data_test, 'eval'), (data_train, 'train')]
                params = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 2}
                bst = xgb.train(params, data_train, num_boost_round=10, evals=watch_list)
                y_hat = bst.predict(data_test)

                origin = sys.stdout
                f = open('E:/MLResult.txt', 'w')
                sys.stdout = f
                print data[i].tostring(), 'XGBoost正确率：', accuracy_score(y_test, y_hat)
                sys.stdout = origin
                f.close()



        conn.commit()
        cur.close()
        conn.close()
except MySQLdb.Error, e:
        print "Mysql Error %d: %s" % (e.args[0], e.args[1])

