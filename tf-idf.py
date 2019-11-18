# -*- coding: utf-8 -*-
import sys
import jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
"""
词频统计工具
"""
def max_value_key(dictionary,num):
    res = []
    for i in range(num+1):
        res.append(max(dictionary, key=dictionary.get))
        del dictionary[res[i]]
    return res

'读入文档'
def read_text(filepath):
    text=[]
    with open (filepath,'r',encoding='UTF-8') as f:
        for line in f:
            text.append(line.strip())
    return text

'获取评论词汇'
def get_word(text,stop_word):
    #print(stop_word)
    words= pd.DataFrame()
    line_num = 0
    for line in text:
        temp_words = {'review_id':line_num, 'content':[]}
        for temp in jieba.cut(line):
            if temp not in stop_word:
                temp_words['content'].append(temp)
        words = words.append(pd.DataFrame(temp_words),ignore_index=True)
        line_num += 1
    return words,line_num

def tf_idf_matrix(words_frame,review_num):
    #构造矩阵
    words=list(set(words_frame['content']))  #获取所有词
    word_dict=dict(zip(words,range(len(words))))  #每个词对应一个ID的词典
    #print(word_dict)
    matrix = np.zeros((len(words),review_num))      #行数=词数；列数=评论数
    for i in range (len(words_frame)):
        col = words_frame.loc[i,'review_id']
        lin = word_dict[words_frame.loc[i,'content']]
        matrix[lin,col] += 1
    sm=np.sum(matrix,axis=0)
    tf=matrix/sm
    #idf
    D=matrix.shape[1]
    j=np.sum(matrix>0,axis=1)
    idf=np.log(D/j)
    #tf-idf
    tf_idf=tf*(idf.reshape(matrix.shape[0],1))
    
    #特征集
    feature_num = np.sum(tf_idf,axis=1)
    for i in range(len(words)):
        print("各词语的tf-idf总值：{} = {}".format(words[i],feature_num[i]))
    feature_dict = dict(zip(words,feature_num))
    feature = max_value_key(feature_dict,30)
    print("tf-idf值最高的前30词：",feature)
    return tf_idf 

def visualize(tf_idf):
    
    #聚类
    Kmeans=KMeans(n_clusters=1000)
    Kmeans.fit(tf_idf)
    #print(Kmeans.cluster_centers_)
    #for index, label in enumerate(Kmeans.labels_, 1):
        #print("index: {}, label: {}".format(index, label))
    print("inertia: {}".format(Kmeans.inertia_))
    
    #可视化
    '''
    使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(tf_idf)
    '''
    pca=PCA(n_components= 2 )
    decomposition_data = pca.fit_transform(tf_idf)
    #print(decomposition_data)
    x = []
    y = []
    
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=Kmeans.labels_, marker="x")
    plt.show()
    plt.savefig('./sample.png', aspect=1)
    #文本相关性（余弦计算）
    n = tf_idf.shape[1]
    relations = np.zeros((n, n))
    for i in range(n):
        vec1 = tf_idf[:, i]
        for j in range(i, n):
            vec2 = tf_idf[:, j]
            relations[i, j] = cosine(vec1, vec2)
    #reverse = dict(zip(word_dict.values(), word_dict.keys()))
    print(relations)
    plt.matshow(relations)
    return 

def main():
    stop_word = read_text(sys.argv[1])
    print("已获取stop_word")
   #stop_word = read_text(input('stop_word_path:'))
    review_word,line_num = get_word(read_text(sys.argv[2]),stop_word)
    print("已获取review_word")
   #review_text = read_text(input('review_text_path:'))
    tf_idf = tf_idf_matrix(review_word,line_num)
    print("已获取tf-idf矩阵")
    visualize(tf_idf)
    #feature = 
    
if __name__=='__main__':main()