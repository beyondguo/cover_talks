# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:54:21 2019
@author: gby
"""
import os
from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import jieba
import re
import time
import copy
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve, auc 
# ===================preprocessing:==============================
def remove_punctuations(text):
    return re.sub('[，。：；’‘“”？！、,.!?\'\"\n\t]','',text)


def fit_corpus(corpus,vocab_size=None):
    """
    corpus 为分好词的语料库
    """
    print("Start fitting the corpus......")
    t = Tokenizer(vocab_size) # 要使得文本向量化时省略掉低频词，就要设置这个参数
    tik = time.time()
    t.fit_on_texts(corpus) # 在所有的评论数据集上训练，得到统计信息
    tok = time.time()
    word_index = t.word_index # 不受vocab_size的影响
    print('all_vocab_size',len(word_index))
    print("Fitting time: ",(tok-tik),'s')
    freq_word_index = {}
    if vocab_size is not None:
        print("Creating freq-word_index...")
        x = list(t.word_counts.items())
        s = sorted(x,key=lambda p:p[1],reverse=True)
        freq_word_index = copy.deepcopy(word_index) # 防止原来的字典也被改变了
        for item in s[vocab_size:]:
            freq_word_index.pop(item[0])
        print("Finished!")
    return t,word_index,freq_word_index


def text2dix(tokenizer,text,maxlen):
    """
    text 是一个列表，每个元素为一个文档的分词
    """
    print("Start vectorizing the sentences.......")
    X = tokenizer.texts_to_sequences(text) # 受vocab_size的影响
    print("Start padding......")
    pad_X = pad_sequences(X,maxlen=maxlen,padding='post')
    print("Finished!")
    return pad_X


def create_embedding_matrix(wvmodel,vocab_size,emb_dim,word_index):
    """
    vocab_size 为词汇表大小，一般为词向量的词汇量
    emb_dim 为词向量维度
    word_index 为词和其index对应的查询词典
    """
    embedding_matrix = np.random.uniform(size=(vocab_size+1,emb_dim)) # +1是要留一个给index=0
    print("Transfering to the embedding matrix......")
    # sorted_small_index = sorted(list(small_word_index.items()),key=lambda x:x[1])
    for word,index in word_index.items():
        try:
            word_vector = wvmodel[word]
            embedding_matrix[index] = word_vector
        except Exception as e:
            print(e,"Use random embedding instead.")
    print("Finished!")
    print("Embedding matrix shape:\n",embedding_matrix.shape)
    return embedding_matrix


def label2idx(label_list):
    label_dict = {}
    unique_labels = list(set(label_list))
    for i,each in enumerate(unique_labels):
        label_dict[each] = i
    new_label_list = []
    for label in label_list:
        new_label_list.append(label_dict[label])
    return new_label_list,label_dict


# 去除图片周围空白的算法，去除之后再resize，更准确:
def trim_blank(img_arr):
    matrix = img_arr[:,:,1]
    w = img_arr.shape[0]
    h = img_arr.shape[1]
    left,right,top,bottom = 0,w,h,0
    # 无论哪个通道，其空白处的像素都是250~255
    blank = 250
    # 把图片的矩阵，分别横向、纵向求和，找出边界点，从而切片
    horizon = np.sum(matrix,axis=0)
    for i in range(w//2):
        if horizon[i] < blank*h:
            left = i
            break
    for i in range(w//2,w):
        if horizon[i] >= blank*h:
            right = i-1
            break
    vertical = np.sum(matrix,axis=1)
    for i in range(h//2):
        if vertical[i] < blank*w:
            bottom = i
            break
    for i in range(h//2,h):
        if vertical[i] >= blank*w:
            top = i-1
            break
    return img_arr[bottom:top,left:right,:]


def img2array(img_path,img_h,img_w):
    img = Image.open(img_path).convert('RGB').resize((img_h,img_h))
    arr = trim_blank(np.array(img))
    # 把书的侧面多余部分给去掉：
    try:
        arr = arr[:,-img_w:,:]
    except Exception as e:
        print('Error for',img_path,e)
    new_img = Image.fromarray(arr)
    new_img = new_img.resize((img_w,img_h)) # 长，宽
    return np.array(new_img)

def pos_neg_split(pos_size,selected_books=None):
    # 根据正样本比例来划分正负样本。
    covers = os.listdir('../covers')
    chinese_books = ['中国books/'+name for name in os.listdir('../books/中国books')]
    foreign_books = ['外国books/'+name for name in os.listdir('../books/外国books')]
    books = foreign_books+chinese_books
    pos_list = []
    neg_list = []
    book_dict = {}
    if selected_books:
        books = selected_books
    for book in books:
        book_detail = pd.read_excel('../books/%s'%book,header=1)
        book_detail = book_detail.sort_values(by='年销',ascending=False)
        sorted_isbns = list(book_detail.ISBN)
        sorted_titles = list(book_detail['书名'])
        cut = int(len(sorted_isbns)*pos_size)
        pos_list += sorted_isbns[:cut]
        neg_list += sorted_isbns[cut:]
        for i,isbn in enumerate(sorted_isbns):
            if isbn not in book_dict:
                book_dict[isbn] = jieba.lcut(sorted_titles[i])
    # filter books without covers:
    pos_list = [x for x in pos_list if '%s.jpg'%x in covers]
    neg_list = [x for x in neg_list if '%s.jpg'%x in covers]
    return list(set(pos_list)),list(set(neg_list)),book_dict


def test_results(model,y_test,inputs_test,framework,plot_roc=False):
    y_true = y_test
    if framework == 'keras':
        y_prob = model.predict(inputs_test)
        y_predict = [1 if y>0.5 else 0 for y in y_prob]
    elif framework == 'sklearn':
        y_prob = model.predict_proba(inputs_test)[:,1]
        y_predict = model.predict(inputs_test)
    else:
        print('Wrong framework! Must be keras or sklearn.')
        return 0
    
    # accuracy,precision,recall,f1
    print('accuracy:',accuracy_score(y_true,y_predict))
    print('precision:',precision_score(y_true,y_predict))
    print('recall:',recall_score(y_true,y_predict))
    print('F1:',f1_score(y_true,y_predict))
    
    # ROC:  
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print('AUC:',roc_auc)
    if plot_roc:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

