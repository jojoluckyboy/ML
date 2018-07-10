# -*- coding:utf-8 -*-
##############################################################
# 开发人员: Jerry
# 代码编码：utf-8
# 代码主题:练习爬虫定义功能模块
# 代码参数: None
# 创建日期:  2016-06-20
# 修改记录： None
#############################################################
import urllib
import urllib2
import re

#78500彩票网预测爬取
class SPIDEYC:


    #初始化，传入基地址
    def __init__(self,baseUrl):
        self.baseUrl = baseUrl

    #传入页码，获取该类帖子所有页的代码
    def getpage(self,initlink):
        try:
            url = self.baseUrl+initlink
            request = urllib2.Request(url)
            response = urllib2.urlopen(request)
            html = response.read()
            pattern = re.findall("下页</a><a href.*?末页",html)
            pageNum = re.split('_|\.',str(pattern).decode('string_escape'))
            getpage = int(pageNum[1])
            return getpage

            if getpage <1:
                print u"该类帖子不存在,请查找原因"
                return None
        except urllib2.URLError, e:
            if hasattr(e,"reason"):
                print u"连接网站失败,错误原因",e.reason
                return None

    #传入页码，读取当前页内容
    def currentpage(self,connectstr,page):
        try:
            url = self.baseUrl+connectstr+str(page+1)+'.html'
            request = urllib2.Request(url)
            response = urllib2.urlopen(request)
            currentpage = response.read()
            return currentpage
        except urllib2.URLError, e:
            if hasattr(e,"reason"):
                print u"跳转当前页失败,错误原因",e.reason
                return None

    #传入专家名字列表，获取期数和每期的链接
    def findexpert(self,expert,compile1,compile2,websitecontent):
        try:
            pattern = re.compile(compile1+expert+compile2)
            links = re.findall(pattern,websitecontent)
            findexpert = re.split('"',str(links).decode('string_escape'))
            #print findexpert
            #print len(findexpert)
            #判断当前页是否取到了该专家的链接
            if len(findexpert)>1:
                #print u"成功获取信息"
                return findexpert
            else:
                #print u"未匹配到信息"
                return None
        except urllib2.URLError, e:
            if hasattr(e,"reason"):
                print u"跳转当前页失败,错误原因",e.reason
                return None

    #获取当期专家页面内容
    def expertpage(self,headstr,expertlink):
        try:
            url = headstr + expertlink
            request = urllib2.Request(url)
            response = urllib2.urlopen(request)
            expertpage = response.read()
            return expertpage
            #print expertpage
        except urllib2.URLError, e:
            if hasattr(e,"reason"):
                print u"跳转当前页失败,错误原因",e.reason
                return None

    #传入参数，获取专家推荐号
    def getball(self,n,consign,ballpattern1,ballpattern2,ballpattern3,websitecontent):
        try:
            ballpattern = ballpattern1 + n + ballpattern2 + consign + ballpattern3
            getball = re.findall(ballpattern,websitecontent)

            #判断是否取到了推荐号
            if getball:
                print u"成功获取推荐号"
                return getball
            else:
                print u"未匹配到信息"
                return 0
        except urllib2.URLError, e:
            if hasattr(e,"reason"):
                print u"跳转当前页失败,错误原因",e.reason
                return None

baseUrl = 'http://www.caiu8.com'
test = SPIDEYC(baseUrl)
#pNum = test.getpage("/ssqyc/list.html")
pNum =15
connectstr = '/ssqyc/list_'

for page in range(pNum):
    currentpage = test.currentpage(connectstr,page)
    expert = '高手的姐'
    compile1 = ".*?<a href=(.*?)target=.*?"
    compile2 = "(.*?)期.*?</a></li>"

    if test.findexpert(expert,compile1,compile2,currentpage):
        expertcontent = test.findexpert(expert,compile1,compile2,currentpage)
        print str(expertcontent).decode('string_escape')
        # headstr ='http://www.caiu8.com/ssqyc'
        # resultpage = test.expertpage(headstr,expertcontent[1])
        #
        # ballpattern1 = "<p>\n.*?"
        # ballpattern2 = ".*?：(.*?)"
        # ballpattern3 = "(.*?)\n</p>"
        # getball = test.getball("12","+",ballpattern1,ballpattern2,ballpattern3,resultpage)
        # print str(getball).decode('string_escape')


