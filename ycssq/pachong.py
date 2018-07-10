# -*- coding: utf-8 -*-
import urllib
import urllib2
import re
from lxml import etree
print "开始爬取..."
# values = {"username":"1016903103@qq.com","password":"XXXX"}
# data = urllib.urlencode(values)
user_agent = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36"
headers = { 'User-Agent' : user_agent }
data = ""
url = "http://blog.csdn.net/"


# #设置保存cookie的文件，同级目录下的cookie.txt
# filename = 'cookie.txt'
# # #声明一个MozillaCookieJar对象实例来保存cookie，之后写入文件
# # cookie = cookielib.MozillaCookieJar(filename)
# # #利用urllib2库的HTTPCookieProcessor对象来创建cookie处理器
# handler = urllib2.HTTPCookieProcessor(cookie)
# # #通过handler来构建opener
# opener = urllib2.build_opener(handler)
# # #创建一个请求，原理同urllib2的urlopen
# # response = opener.open(url)
# # #保存cookie到文件
# # cookie.save(ignore_discard=True, ignore_expires=True)

request = urllib2.Request(url,data,headers)
response = urllib2.urlopen(request)
html = response.read()
#print html

selector = etree.HTML(html)
#这里使用id属性来定位哪个div和ul被匹配 使用text()获取文本内容
content1 = selector.xpath('//*[@class="csdn-tracking-statistics"]/a/text()')
str_content = " ".join(content1)
print str_content

print ("=========================================")
for i in range(1, 20):
#content = selector.xpath('//*[@class="list_con"]')
    content = selector.xpath('//*[@id="feedlist_id"]/li['+str(i)+']/div/h2/a/text()')
    print content[0]
print ("=========================================")
res = r'<ul class="feedlist_mod" id="feedlist_id".*?>(.*?)</ul>'
mm =re.findall(res, html, re.S|re.M)

for value in mm:
    print value
# readnum = selector.xpath('//*[@class="read_num"]/text()')
#str_readnum = " ".join(readnum)
#str_content = " ".join(content)

#print str_readnum

