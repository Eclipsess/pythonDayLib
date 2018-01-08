#coding:utf-8

'''
计算余弦距离。
cos()计算一维数组余弦距离
cos_sim()利用numpy库计算一维数组余弦距离
sklearn包中cosine_similarity()可以计算二维余弦距离
'''

import numpy as np
import sklearn.metrics.pairwise as pw


def cos(vector1,vector2):  
    '''适用于一维数组'''
    dot_product = 0.0;  
    normA = 0.0;  
    normB = 0.0;  
    for a,b in zip(vector1,vector2):  
        dot_product += a*b  
        normA += a**2  
        normB += b**2  
    if normA == 0.0 or normB==0.0:  
        return None  
    else:  
        return dot_product / ((normA*normB)**0.5)

def cos_sim(x1, x2):
    '''适用于一维数组'''
    a = np.sum(np.multiply(x1, x2))
    b = np.sum(np.square(x1))
    c = np.sum(np.square(x2))
    if b == 0.0 or c == 0.0:
        return None
    else:
        return a / ((b * c) ** 0.5)

if __name__ == '__main__':
    x1 = np.array([1,2,3])
    x2 = np.array([2,3,4])

    x3 = np.array([[1,2,3],[1,3,5]])
    x4 = np.array([[2,3,4],[4,5,6]])

    print cos(x1, x2)
    print cos_sim(x1, x2)
    print pw.cosine_similarity(x3, x4)
