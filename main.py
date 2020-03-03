# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:09:25 2019

@author: gby
attention model
"""
import os
import keras
from keras.models import Model
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
import cv2

from utils import fit_corpus,text2dix,create_embedding_matrix,pos_neg_split,img2array,test_results


# =================训练样本的生成：====================
img_h = 96*2
img_w = 64*2
# 正负样本的划分：
pos_list,neg_list,book_dict = pos_neg_split(0.25,selected_books=['外国books/傲慢与偏见.xls','外国books/悲惨世界.xls','外国books/呼啸山庄.xls'])


#  把所有的表都合在一起，方便查询其他信息：
chinese_books = ['中国books/'+name for name in os.listdir('../books/中国books')]
foreign_books = ['外国books/'+name for name in os.listdir('../books/外国books')]
all_books = chinese_books+foreign_books
tables = []
for book in all_books:
    table = pd.read_excel('../books/%s'%book,header=1)
    tables.append(table)
big_table = pd.concat(tables)
#big_table = pd.read_excel('../books/中国books/红楼梦.xls',header=1)


def getColorHist(img_path):
	image = cv2.imread(img_path)
	c0 =  cv2.calcHist([image], [0], None, [256], [0.0,255.0])
	c1 =  cv2.calcHist([image], [1], None, [256], [0.0,255.0])
	c2 =  cv2.calcHist([image], [2], None, [256], [0.0,255.0])
	return np.concatenate((c0,c1,c2)).reshape(768,)


# 把图片转化成数值：
X_img = []
X_colors = []
X_isbns = []
Y = []
for isbn in pos_list:
    try:
        img_path = '../covers/%s.jpg'%isbn
        X_img.append(img2array(img_path,img_h,img_w))
        X_colors.append(getColorHist(img_path))
        Y.append(1)
        X_isbns.append(isbn)
    except Exception as e:
        print('Error for picture: %s.jpg'%isbn,e)
for isbn in neg_list:
    try:
        img_path = '../covers/%s.jpg'%isbn
        X_img.append(img2array(img_path,img_h,img_w))
        X_colors.append(getColorHist(img_path))
        Y.append(0)
        X_isbns.append(isbn)
    except Exception as e:
        print('Error for picture: %s.jpg'%isbn,e)

# 文本的处理：
corpus = []
for v in list(book_dict.values()):
    corpus += v
vocab_size = 5000
maxlen = 15
tokenizer, word_index, freq_word_index = fit_corpus(corpus,vocab_size=vocab_size)
rev_freq_word_index = {}
for k in freq_word_index:
    v = freq_word_index[k]
    rev_freq_word_index[v] = k
rev_freq_word_index[0] = '*'
X_title_orig = [book_dict[isbn] for isbn in X_isbns]
X_title = text2dix(tokenizer,X_title_orig,maxlen=maxlen)


##
#from shutil import copy
#for isbn in X_isbns:
#    source_path = 'D:/PythonOK/图书封面影响/covers/%s.jpg'%isbn
#    target_path = 'D:/PythonOK/图书封面影响/covers_subset'
#    copy(source_path,target_path)
##

# 加载词向量模型，创建embedding matrix：
wv_path = '../../wv/wikibaikeWV250/wikibaikewv250'
print("Loading word2vec model, may take a few minutes......")
if ('wvmodel' not in vars()): # 避免重复加载  
    wvmodel = Word2Vec.load(wv_path)
wvdim = 250
embedding_matrix = create_embedding_matrix(wvmodel,vocab_size,wvdim,freq_word_index)


# ==============================
# 将顺序打乱
indexs = np.random.permutation(range(len(X_isbns)))
X_isbns = np.array(X_isbns)[indexs]
X_colors = np.array(X_colors)[indexs]
X_img = np.array(X_img)[indexs]
X_title = X_title[indexs]
Y = np.array(Y)[indexs]

# 获取其他的信息，例如price、publisher：
X_price = []
X_pub = []
for isbn in X_isbns:
    price = list(big_table[big_table.ISBN == isbn]['定价'])[0]
    publisher = list(big_table[big_table.ISBN == isbn]['出版社'])[0]
    X_price.append(price)
    X_pub.append(publisher)
# 把publisher转化成编号的形式
unique_publisher = set(X_pub)
pub_idx = {}
for i,each in enumerate(list(unique_publisher)):
    pub_idx[each] = i
X_pub = [pub_idx[pub] for pub in X_pub]

# 划分训练、测试集：
train_size = int(0.75*len(Y))

X_img_train = X_img[:train_size]
X_colors_train = X_colors[:train_size]
X_title_train = X_title[:train_size]
y_train = Y[:train_size]

X_img_test = X_img[train_size:]
X_colors_test = X_colors[train_size:]
X_title_test = X_title[train_size:]
y_test = Y[train_size:]




# ========= 训练传统模型 ==========
#from scipy import sparse
#from sklearn.linear_model import LogisticRegression as LR
#from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
## 连续特征进行离散化，离散特征进行独热编码，最终吧所有特征都转化成one-hot
#from sklearn.preprocessing import KBinsDiscretizer,OneHotEncoder
#dis_encoder = KBinsDiscretizer(n_bins=6).fit(np.array(X_price).reshape(-1,1))
#X_price_oh = dis_encoder.transform(np.array(X_price).reshape(-1,1))
#oh_encoder = OneHotEncoder().fit(np.array(X_pub).reshape(-1,1))
#X_pub_oh = oh_encoder.transform(np.array(X_pub).reshape(-1,1))
#X_price_train = X_price_oh[:train_size]
#X_pub_train = X_pub_oh[:train_size]
#X_price_test = X_price_oh[train_size:]
#X_pub_test = X_pub_oh[train_size:]
## 二者拼接起来：
#X_pp_oh = sparse.csc_matrix(np.concatenate((X_price_oh.toarray(),X_pub_oh.toarray()),axis=1))
#X_pp_train = X_pp_oh[:train_size]
#X_pp_test = X_pp_oh[train_size:]
#
#svm = SVC(kernel='linear',probability=True)
#mlp = MLPClassifier((32,16))
#lr = LR()
#ml_model = lr
#ml_model.fit(X_price_train,y_train)
#test_results(ml_model,y_test,X_price_test,'sklearn')


from models import mv_att_mm,simple_cat_m,all_cat_model
#=======
vgg_layer = 'block3_conv1'
shrink_rate = 16
filters = 256
"""
block5_conv1   shrink_rate = 256  filters = 512
block4_conv1   shrink_rate = 64  filters = 256
block3_conv1   shrink_rate = 16  filters = 256
block2_conv1   shrink_rate = 4  filters = 128
"""
# my_model = simple_cat_m(vocab_size,maxlen,wvdim,embedding_matrix,img_h,img_w,vgg_layer,shrink_rate,filters) # (192,128),(96,64)
my_model = all_cat_model(vocab_size,maxlen,wvdim,embedding_matrix)
for i in range(20):
    my_model.fit([X_title_train,X_colors_train],y_train,batch_size=32,epochs=1)
    test_results(my_model,y_test,[X_title_test,X_colors_test],'keras')

 

 

