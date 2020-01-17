# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:00:14 2019

@author: gby
"""

import keras
from keras.models import Sequential,Model
from keras.applications.vgg19 import VGG19
from keras.layers import Dense,LSTM,GRU,Embedding,Input,Conv1D,MaxPooling1D,GlobalMaxPooling2D
from keras.layers import Dropout,Concatenate,concatenate,Lambda,Subtract,Multiply,BatchNormalization
from keras.layers import Flatten,Reshape,TimeDistributed,RepeatVector
import keras.backend as K
import tensorflow as tf
from keras.utils import to_categorical

# ============== 模型搭建：=======

def mp_att_mm(vocab_size,maxlen,wvdim,embedding_matrix,img_h,img_w):# 192,128
    """
	mp，即multi-position，这里指对不同position进行attention。
    """
    # 标题部分
    title_input = Input((maxlen,))
    title_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix])(title_input)
    x = LSTM(64,return_sequences=True)(title_emb)
    title_vec = LSTM(32)(x)
    
    # 图像部分：
    img_base_model = VGG19(weights='imagenet',include_top=False,input_shape=(img_h,img_w,3))
    for layer in img_base_model.layers:
        layer.trainable = False
    
    '''
    将VGG的中间某层的各通道提出来，而不是拿最后的FC层出来的向量做attention操作
    '''
    
    ffs = img_base_model.get_layer('block5_conv3').output
    flat_ffs = Reshape(target_shape=[img_h*img_w//256,512])(ffs)  # [None, img_h*img_w//256, 512]
    # 这里可以使用keras的TimeDistributed层，把18当做是不同的时间维度来处理，6的一批
    img_features = TimeDistributed(Dense(32,activation='relu'))(flat_ffs) #[None, img_h*img_w//256, 32]
    
    # multi-position attention:
    repeated_title_vec = RepeatVector(img_h*img_w//256)(title_vec) #[None, img_h*img_w//256, 32]
    #repeated_title_vec = Reshape(target_shape=[16,512])(repeated_title_vec)
    multidim_concat = Concatenate()([img_features,repeated_title_vec]) #[None, img_h*img_w//256, 64]
    att_base = TimeDistributed(Dense(1,activation='sigmoid'))(multidim_concat)
    att_scores = Lambda(lambda b:K.softmax(b,axis=1),name='att_scores')(att_base)  # [img_h*img_w//256,1]
    # keras的Multiply层只能接受相同shape的Tensor进行相乘，好像tf就不收这个限制
    attended_img_features = Lambda(lambda pair:tf.multiply(pair[0],pair[1]))([att_scores,img_features])#[None, img_h*img_w//256, 32]
    img_vec = Lambda(lambda s:K.sum(s,axis=1),name='weighted_img_vec')(attended_img_features)
#    img_vec = Flatten()(attended_img_features)
#    img_vec = Dense(32,activation='relu')(img_vec)
    
    # 二者融合,拼接法：
    concat = Concatenate()([title_vec,img_vec])
    
    fusion = Dense(32,activation='relu')(concat)
    prediction = Dense(1,activation='sigmoid')(fusion)
    
    model = Model(inputs=[img_base_model.input,title_input],outputs=[prediction])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


def mp_att_mm_2(vocab_size,maxlen,wvdim,embedding_matrix,img_h,img_w):# 192,128
    """
	mp，即multi-position，这里指对不同position进行attention。
	"""
    # 标题部分：
    title_input = Input((maxlen,))
    title_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix])(title_input)
    x = LSTM(64,return_sequences=True)(title_emb)
    title_vec = LSTM(32)(x)
    
    # 图像部分：
    img_base_model = VGG19(weights='imagenet',include_top=False,input_shape=(img_h,img_w,3))
    for layer in img_base_model.layers:
        layer.trainable = False

    ffs = img_base_model.get_layer('block5_conv3').output
    flat_ffs = Reshape(target_shape=[img_h*img_w//256,512])(ffs)  # [None, img_h*img_w//256, 512]
    # 这里可以使用keras的TimeDistributed层，把各个位置当做是不同的时间维度来处理，6的一批
    img_features = TimeDistributed(Dense(32,activation='relu'))(flat_ffs) #[None, img_h*img_w//256, 32]
    
    """
    得到了各个位置的图像向量之后，对每个位置都把文本向量拼接上去
    即每个位置都得到一个32+32维的向量。
    """
    # multi-position attention:
    repeated_title_vec = RepeatVector(img_h*img_w//256)(title_vec) #[None, img_h*img_w//256, 32]
    #repeated_title_vec = Reshape(target_shape=[16,512])(repeated_title_vec)
    mp_cat = Concatenate()([img_features,repeated_title_vec]) #[None, img_h*img_w//256, 64]
    
    att_base = TimeDistributed(Dense(32,activation='relu'))(mp_cat)
    att_base = TimeDistributed(Dense(1,activation='sigmoid'))(att_base)
    att_scores = Lambda(lambda b:K.softmax(b,axis=1),name='att_scores')(att_base) #[img_h*img_w//256,1]
    # keras的Multiply层只能接受相同shape的Tensor进行相乘，好像tf就不收这个限制
    att_mp_cat = Lambda(lambda pair:tf.multiply(pair[0],pair[1]))([att_scores,mp_cat])#[None, img_h*img_w//256, 32]
    att_mp_vec = Lambda(lambda s:K.sum(s,axis=1),name='weighted_vec')(att_mp_cat)
    
    prediction = Dense(1,activation='sigmoid')(att_mp_vec)
    
    model = Model(inputs=[img_base_model.input,title_input],outputs=[prediction])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def mv_att_mm(vocab_size,maxlen,wvdim,embedding_matrix,img_h,img_w,vgg_layer,shrink_rate,filters,otherinfo_dim=None):
    """
	mv，即multi-view，这里指对不同通道的activation进行attention。
	这里可以自定义取出VGG19的哪一层，
	实验发现，取比较浅的层，效果依然很好，而训练速度可以大大提高
	"""
    # 标题部分：
    title_input = Input((maxlen,))
    title_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix])(title_input)
    x = LSTM(64,return_sequences=True)(title_emb)
    title_vec = LSTM(32)(x)
    
    # 图像部分：
    img_base_model = VGG19(weights='imagenet',include_top=False,input_shape=(img_h,img_w,3))
    for layer in img_base_model.layers:
        layer.trainable = False
    
    '''
    将VGG的中间某层的各通道提出来，而不是拿最后的FC层出来的向量做attention操作
    '''
    
    ffs = img_base_model.get_layer(vgg_layer).output
    flat_ffs = Reshape(target_shape=[filters,img_h*img_w//shrink_rate])(ffs)  # [None, 512, img_h*img_w//256]??
    # 这里可以使用keras的TimeDistributed层，把512当做是不同的时间维度来处理，6的一批
    img_features = TimeDistributed(Dense(32,activation='relu'))(flat_ffs) #[None, 512, 32]
    

    repeated_title_vec = RepeatVector(filters)(title_vec)
    #repeated_title_vec = Reshape(target_shape=[16,512])(repeated_title_vec)
    mv_cat = Concatenate()([img_features,repeated_title_vec]) #[None, 512, 64]
    att_base = TimeDistributed(Dense(32,activation='relu'))(mv_cat)
    att_base = TimeDistributed(Dense(1,activation='sigmoid'))(att_base)
    att_scores = Lambda(lambda b:K.softmax(b,axis=1),name='att_scores')(att_base)
    # keras的Multiply层只能接受相同shape的Tensor进行相乘，好像tf就不收这个限制
    att_mv_cat = Lambda(lambda pair:tf.multiply(pair[0],pair[1]),name='att_mv_cat')([att_scores,mv_cat])#[None, 512, 64]
    att_mv_vec = Lambda(lambda s:K.sum(s,axis=1),name='att_mv_vec')(att_mv_cat) # [None,64]
    
    inputs = [img_base_model.input,title_input]
    if otherinfo_dim:
        other_input = Input((otherinfo_dim,))
        dense_input = Dense(64,activation='relu')(other_input)
        att_mv_vec = Concatenate()([att_mv_vec,dense_input])
        inputs = [img_base_model.input,title_input,other_input]

    fusion = Dense(32,activation='relu')(att_mv_vec)
    prediction = Dense(1,activation='sigmoid')(fusion)
    
    model = Model(inputs=inputs,outputs=[prediction])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


def simple_cat_m(vocab_size,maxlen,wvdim,embedding_matrix,img_h,img_w,vgg_layer,shrink_rate,filters,otherinfo_dim=None,selected_features=None):
    """
	mv，即multi-view，这里指对不同通道的activation进行attention。
	这里可以自定义取出VGG19的哪一层，
	实验发现，取比较浅的层，效果依然很好，而训练速度可以大大提高
	"""
    my_dim = 1
    # 标题部分：
    title_input = Input((maxlen,))
    title_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix])(title_input)
    x = LSTM(64,return_sequences=True)(title_emb)
    title_vec = LSTM(my_dim)(x)
    
    # 图像部分：
    img_base_model = VGG19(weights='imagenet',include_top=False,input_shape=(img_h,img_w,3))
    for layer in img_base_model.layers:
        layer.trainable = False
    
    '''
    将VGG的中间某层的各通道提出来，而不是拿最后的FC层出来的向量做attention操作
    '''
    
    mid_output = img_base_model.get_layer(vgg_layer).output
    x = GlobalMaxPooling2D()(mid_output) 
    x = Dense(32,activation='relu')(x)
    img_vec = Dense(4,activation='tanh')(x)
    
#    cat_vec = Concatenate()([img_vec,title_vec])
#    
#    inputs = [img_base_model.input,title_input]

    # 此时，为了查看每一种特征的作用，我们用列表的方式输入
    other_inputs = []
    other_vecs = []
    for dim in otherinfo_dim:
        other_input = Input((dim,))
        other_inputs.append(other_input)
        other_vec = Dense(my_dim,activation='sigmoid')(other_input)
        other_vecs.append(other_vec)
        
    
    cat_vec = Concatenate()([img_vec,title_vec]+other_vecs)
    inputs = [img_base_model.input,title_input]+other_inputs

#    fusion = Dense(32,activation='relu')(cat_vec)
    prediction = Dense(1,activation='sigmoid',name='lr')(cat_vec)
    
    model = Model(inputs=inputs,outputs=[prediction])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model











