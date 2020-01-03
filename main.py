# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:09:25 2019

@author: gby
attention model
"""

import keras
from keras.models import Model
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from utils import fit_corpus,text2dix,create_embedding_matrix,pos_neg_split,img2array,test_results


# =================训练样本的生成：====================
img_h = 96*2
img_w = 64*2
# 正负样本的划分：
pos_list,neg_list,book_dict = pos_neg_split(0.25)

# 把图片转化成数值：
X_img = []
X_isbns = []
Y = []
for isbn in pos_list:
    try:
        img_path = '../covers/%s.jpg'%isbn
        X_img.append(img2array(img_path,img_h,img_w))
        Y.append(1)
        X_isbns.append(isbn)
    except Exception as e:
        print('Error for picture: %s.jpg'%isbn,e)
for isbn in neg_list:
    try:
        img_path = '../covers/%s.jpg'%isbn
        X_img.append(img2array(img_path,img_h,img_w))
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




# 加载词向量模型，创建embedding matrix：
wv_path = '../../wv/wikibaikeWV250/wikibaikewv250'
print("Loading word2vec model, may take a few minutes......")
if ('wvmodel' not in vars()): # 避免重复加载  
    wvmodel = Word2Vec.load(wv_path)
wvdim = 250
embedding_matrix = create_embedding_matrix(wvmodel,vocab_size,wvdim,freq_word_index)


# ==============================

indexs = np.random.permutation(range(len(X_isbns)))
X_img = np.array(X_img)[indexs]
X_title = X_title[indexs]
Y = np.array(Y)[indexs]

train_size = int(0.75*len(Y))
X_img_train = X_img[:train_size]
X_title_train = X_title[:train_size]
y_train = Y[:train_size]
X_img_test = X_img[train_size:]
X_title_test = X_title[train_size:]
y_test = Y[train_size:]




# ========= 训练模型 ==========
from models import mp_att_mm,mp_att_mm_2,mv_att_mm
mp_model = mp_att_mm_2(vocab_size,maxlen,wvdim,embedding_matrix,img_h,img_w) # (192,128),(96,64)

vgg_layer = 'block3_conv1'
shrink_rate = 16
filters = 256
"""
block5_conv1   shrink_rate = 256  filters = 512
block4_conv1   shrink_rate = 64  filters = 256
block3_conv1   shrink_rate = 16  filters = 256
block2_conv1   shrink_rate = 4  filters = 128
"""
mv_model = mv_att_mm(vocab_size,maxlen,wvdim,embedding_matrix,img_h,img_w,vgg_layer,shrink_rate,filters) # (192,128),(96,64)
for i in range(8):
    mv_model.fit([X_img_train,X_title_train],y_train,batch_size=32,epochs=1)
    test_results(mv_model,y_test,[X_img_test,X_title_test])
# 提取attention weights的模型：
att_scores = mv_model.get_layer('att_scores').output
filter_acts = mv_model.get_layer(vgg_layer).output
att_model = Model(inputs=mv_model.input,outputs=[att_scores,filter_acts])

"""
pos: 9787547721452,9787519005917,9787511725554,9787544768504
neg: 9787550009455,9787200048964,9787807365532,9787807534679
"""

example_isbns = [9787547721452,9787519005917,9787511725554,9787544768504]
X_title_example_words = [book_dict[isbn] for isbn in example_isbns]
X_title_example = text2dix(tokenizer,X_title_example_words,maxlen=maxlen)
X_img_example = [img2array('../covers/%s.jpg'%isbn,img_h,img_w) for isbn in example_isbns]
y_example = mp_model.predict([X_img_example,X_title_example])
atns,acts = att_model.predict([X_img_example,X_title_example])[0],\
            att_model.predict([X_img_example,X_title_example])[1]

n = 10
for x_img,atn, act in zip(X_img_example[:5],atns[:5],acts[:5]):
    plt.imshow(x_img)
    plt.axis('off')
    plt.show()
    scores = [x[0] for x in atn]
    sorted_scores = sorted(scores,reverse=True)
    best_scores = sorted_scores[:n]
    best_idxs = []
#    plt.figure(figsize=(8,4))
    for i,each in enumerate(best_scores):
        idx = scores.index(each)+i
        scores.remove(each)
        plt.subplot(2,n//2,i+1)
        plt.imshow(act[:,:,idx])
        plt.axis('off')
    plt.show()


