_author__='Jerry'
# -*- coding:utf-8 -*-

import urllib2
import re

#彩票网预测爬虫类

class  CPYC:

    #初始化，传入基地址
    def __init__(self,baseUrl):
        self.baseUrl = baseUrl

    #获取该页面所有超链接

baseUrl = 'http://www.caiu8.com/ssqyc/list.html'
#request = CPYC(baseUrl)

#print request
website = urllib2.urlopen( 'http://www.caiu8.com/ssqyc/list.html')
html = website.read()
#links = re.findall('<ul class="lists">(.+?)</ul>', html,re.S)

#pattern = re.findall('<a href="/ssqyc/list_675.html" style="margin-right:5px;">末页</a> ', html,re.S)
#pattern = re.findall('.*?<a href=".*?/ssqyc/list_.*?.html"> ',html,re.S)
#pattern = re.findall('.*?>下页</a> ',html,re.S)
#print str(links).decode('string_escape')
#print str(pattern).decode('string_escape')
#print str(html).decode('string_escape')

pattern = re.findall("下页</a><a href.*?末页",html)
pageNum = re.split('_|\.',str(pattern).decode('string_escape'))
pNum = int(pageNum[1])

#print str(pattern).decode('string_escape')
#print pNum
#pNum-670
for pages in range(20):

        weblist =  urllib2.urlopen( 'http://www.caiu8.com/ssqyc/list_'+str(pages+1)+'.html')
        ball = weblist.read()
#正则表达式插入变量
        #print str(ball).decode('string_escape')
        #name = "寒天"
        #print(name)
        #links = re.findall("<li.*?"+str(name)+".*?</a></li>", ball)

#获取页面链接
        # links = re.compile("<li><a href=(.*?)target=.*?"+str(name)+".*?</a></li>")
        # links1 = re.findall(links,ball)
        # linksp = re.split('"',str(links1).decode('string_escape'))
        # print(linksp[1])
        name = "军辉"

        #links = re.findall("<li><a href=.*? target=(.*?)"+str(name)+".*?</a></li>", ball)
        #links = re.findall(".*?<a href=(.*?) target=(.*?)"+str(name)+"(.*?)期(.*?)</a></li>",ball)

        links = re.compile(".*?<a href=(.*?)target=.*?"+name+"(.*?)期.*?</a></li>")
        links1 = re.findall(links,ball)

        linksp = re.split('"',str(links1).decode('string_escape'))
        print str(links1).decode('string_escape')
        #print str('http://www.caiu8.com/ssqyc/list_'+str(pages+1)+'.html').decode('string_escape')
        #print str(ball).decode('string_escape')
        #print str(links1).decode('string_escape')
        findmessage = urllib2.urlopen( 'http://www.caiu8.com/ssqyc/'+str(linksp[1]))
        resultpage = findmessage.read()
        print str(resultpage).decode('string_escape')

        #getBall = re.findall('<div class="context">.*?<p>.*?12|8.*?推荐.*?</p>',resultpage)
        getBall = re.findall("<p>\n.*?12.*?：(.*?)(.*?)\n</p>",resultpage)
        getBallS1 = re.split(',|：',str(getBall).decode('string_escape'))


        print str(getBall).decode('string_escape')

