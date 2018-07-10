# -*- coding:utf-8 -*-
import itertools

import numpy as np
import pandas as pd
import MySQLdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


#######################################################################
# Description：
# 数据读库加逻辑回归验证
# Created with Python 2.7.
# User：wang wencong
# Date：2017/8/17
# Copyright© 2003-2016 Zhejiang huixin technology company
#######################################################################
def find_result():
    try:
        conn = MySQLdb.connect(host='192.168.5.63', user='root', passwd='wwc@icinfo', port=3306, charset='utf8')
        cur = conn.cursor()

        conn.select_db('ssq')

        # count = cur.execute('select issue,expertname,redthree,lasthit,lastscore,formax,oddeven,smallbig,primecom,p51,p52,p53,'
        #                     'p101,p102,p103,p201,p202 ,p203,p301,p302,p303,p501,p502,p503,p1001,p1002,p1003,nowhit,nowscore '
        #                     'from ssqthred WHERE expertname = "一码当先"')

        sqlqry= "select issue,expertname,redthree,lasthit,lastscore,formax,ssqthred.distinct,oddeven,smallbig,primecom,p51,p52,p53,"\
                        "p101,p102,p103,p201,p202 ,p203,p301,p302,p303,p501,p502,p503,p1001,p1002,p1003,"\
                        "threesum,div30,div31,div32,divergence,skewness,acV,nowhit"\
                        " from ssqthred WHERE expertname ='一码当先'";

        count = cur.execute(sqlqry)
        print 'there has %s rows record' % count

        results = cur.fetchall()
        results = list(results)
        conn.commit()
        cur.close()
        conn.close()
        return results
    except MySQLdb.Error, e:
        print "Mysql Error %d: %s" % (e.args[0], e.args[1])

if __name__ == '__main__':
    results = find_result()
    # print str(results).decode(encoding='unicode_escape')
    # # print str(results).decode(encoding='string_escape')


   #通过切割合并dataframe形成回归数据

    # head = ["issue", "expertname", "redthree", "lasthit", "lastscore", "formax", "distinct", "oddeven", "smallbig"
    #         , "primecom", "p51", "p52", "p53", "p101", "p102", "p103", "p201", "p202" , "p203", "p301", "p302", "p303"
    #         , "p501", "p502", "p503", "p1001", "p1002", "p1003", "nowscore", "nowhit"]
    # headT = ["formax", "p51", "p52", "p53", "p101", "p102", "p103", "p201", "p202", "p203", "p301", "p302", "p303"
    #          , "p501", "p502", "p503", "p1001", "p1002", "p1003"]
    # data = pd.DataFrame(results, columns=head, dtype=np.int32)
    #
    # dataT = pd.DataFrame(np.random.rand(1, 19), columns=headT, dtype=np.int32)
    #
    # dataTrans = data.loc[:(data.iloc[:, 0].size-2), ["formax", "p51", "p52", "p53", "p101", "p102", "p103", "p201", "p202", "p203", "p301", "p302", "p303"
    #                                                  , "p501", "p502", "p503", "p1001", "p1002", "p1003"]]
    #
    # frames = [dataT, dataTrans]
    # datalandscape = pd.concat(frames, keys=['x', 'y'], ignore_index=True)
    #
    #
    # # dataTrans = dataTrans.drop dataTrans.iloc[-1:]
    # # print dataTrans
    # dataS = data.loc[:, [ "lasthit", "lastscore", "formax", "distinct", "oddeven", "smallbig"
    #         , "primecom", "p51", "p52", "p53", "p101", "p102", "p103", "p201", "p202" , "p203", "p301", "p302", "p303"
    #         , "p501", "p502", "p503", "p1001", "p1002", "p1003", "nowscore", "nowhit"]]
    #
    # dataS1 = data.loc[:, [ "lasthit", "lastscore", "formax", "distinct", "oddeven", "smallbig"
    #         , "primecom",  "nowhit"]]
    #
    # dataS2 = data.loc[:, ["lasthit", "lastscore", "distinct", "oddeven", "smallbig"
    #                   , "primecom",  "nowhit"]]
    #
    # dataS3 = data.loc[:, ["formax", "distinct", "nowhit"]]
    #
    # datamerge =pd.concat([datalandscape, dataS2], axis=1)
    # print datamerge
    ############################################################################################################

    head = ["issue", "expertname", "redthree", "lasthit", "lastscore", "formax", "distinct", "oddeven", "smallbig"
            , "primecom", "p51", "p52", "p53", "p101", "p102", "p103", "p201", "p202", "p203", "p301", "p302", "p303"
            ,"p501","p502","p503","p1001","p1002","p1003", "threesum", "div30", "div31", "div32", "divergence", "skewness", "acV", "nowhit"]

    data = pd.DataFrame(results, columns=head, dtype=np.int32)
    print data

    dataT = data.loc[:, ["threesum", "div30", "div31", "div32", "divergence", "skewness", "acV", "lasthit", "lastscore",
                         "formax", "distinct", "oddeven", "smallbig", "primecom", "p51", "p52", "p53", "p101", "p102",
                         "p103", "p201", "p202", "p203", "nowhit"]]

    dataT1 = data.loc[:, ["threesum", "div30", "div31", "div32",  "divergence", "skewness", "acV", "lasthit", "formax", "distinct", "oddeven",
                          "smallbig", "primecom", "nowhit"]]

    dataT2 = data.loc[:, ["divergence", "threesum", "formax", "skewness", "acV", "nowhit"]]
    #
    # x, y = np.split(dataS.values, (26,), axis=1)
    #
    # # 仅使用前两列特征
    # x = x[:, :26]

    x, y = np.split(dataT2.values, (5,), axis=1)
#    x, y = np.split(datamerge.values, (25,), axis=1)

    # 仅使用前两列特征
    x = x[:, :5]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.5)
    #
    # lr = Pipeline([('sc', StandardScaler()),
    #                ('poly', PolynomialFeatures(degree=10)),
    #                ('clf', LogisticRegression()) ])
    #lr =LogisticRegressionCV(Cs=np.logspace(-3, 3, 7))
#使用logistic回归
    # lr =LogisticRegression()
    #
    # lr.fit(x_train, y_train.ravel())
    #
    #
    # # y_hat = lr.predict(x)
    # # y_hat_prob = lr.predict_proba(x)
    # y_hat = lr.predict(x_test)
    # y_hat_prob = lr.predict_proba(x_test)
    # np.set_printoptions(suppress=True)
    #
    # print 'y_hat = \n', y_hat
    # print 'y_hat_prob = \n', y_hat_prob
    # print u'准确度：%.2f%%' % (100*np.mean(y_hat == y_test.ravel()))
    # print '准确率： ', accuracy_score(y_test, y_hat)


    rf = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=0, min_samples_leaf=0)
    model = GridSearchCV(rf, param_grid={'max_depth': np.arange(2, 10), 'min_samples_leaf': (1, 10)})
    model.fit(x_train, y_train)
    print '最优参数： ', model.best_params_
    y_train_pred = model.predict(x_train)   # 训练数据
    y_test_hat = model.predict(x_test)      # 测试数据
    print y_test_hat
    print y_test
    print '随机森林训练集准确率：', accuracy_score(y_train, y_train_pred)
    print '随机森林测试集准确率:', accuracy_score(y_test, y_test_hat)


     # 画图
    N, M = 500, 500     # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()   # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()   # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
    x_test1 = np.stack((x1.flat, x2.flat), axis=1)   # 测试点

    # # 无意义，只是为了凑另外两个维度
    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    # y_hat = lr.predict(x_test1)                  # 预测值
    y_hat = model.predict(x_test1)                  # 预测值
    y_hat = y_hat.reshape(x1.shape)                 # 使之与输入的形状相同
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)     # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(x[:, 0].shape), edgecolors='k', s=50, cmap=cm_dark)    # 样本的显示
    plt.xlabel(u'预测区域', fontsize=14)
    plt.ylabel(u'与最大连错的差距', fontsize=14)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
              mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
              mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]
    plt.legend(handles=patchs, fancybox=True, framealpha=0.8)
    plt.title(u'预测回归分类效果 - 标准化', fontsize=17)
    plt.show()
