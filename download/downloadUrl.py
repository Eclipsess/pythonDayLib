#coding:utf-8
import urllib2
import pandas as pd
import os
data = pd.read_excel('/Users/eclipsycn/Documents/xx.xlsx')
fileCount = 0
imgCount = 0
for i in range(data.shape[0]):
    try:
        fileName = data.iloc[i][0]
        imgUrl = data.iloc[i][1]
        f = urllib2.urlopen(imgUrl)
        da = f.read()
        if not os.path.exists('./SKU/' + fileName):
            os.mkdir('./SKU/' + fileName)
            fileCount += 1
        with open('./SKU/' + fileName + '/' + imgUrl.split('/')[-1], 'wb') as w:
            w.write(da)
            imgCount += 1
        if imgCount % 100 == 0:
            print imgCount
    except:
        print imgUrl
        pass
    
