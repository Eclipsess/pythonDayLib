#coding:utf-8
# in Ubuntu16.04 locale('en-US','UTF-8')
# ubuntu print 默认自动encode utf-8 (如果需要decode 也是默认decode(utf-8)，然后encode('utf-8'))
#　终端运行此脚本
import os
wo = '\xe6\x88\x91' #wo utf-8
wogbk = wo.decode('utf-8').encode('gbk')
print wogbk
#print wo.decode('gbk') #error
print wo
#乱码 
a = 'ww'.encode('gbk')
print a
#ww
k1 = os.listdir('/home/sy/finetuneVGGFACE/demoImage_crop/王光伟')[0]
print type(k1)
#类型为str
print k1
k = os.listdir('/home/sy/finetuneVGGFACE/demoImage_crop/王光伟'.decode('utf-8'))[0]
print type(k)
#类型为unicode
print k.encode('utf-8')

#print k.encode('ascii') 
#assert k.decode('utf-8') == u'王光伟'
