# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 18:38:47 2019

@author: HP
"""


from selenium import webdriver
import requests
import re
import pandas as pd
import time
import os
import random

# 【考虑增加ip池，或者添加header信息】
def get_image_from_DB(isbn_list):
    failures = []
    browser = webdriver.Chrome(r'D:\downloads\chromedriver_win32\chromedriver.exe')
    for i,isbn in enumerate(isbn_list):
        if '%s.jpg'%isbn in os.listdir('covers'):
            print("Book %s already exists!"%isbn)
            continue
        url = 'https://search.douban.com/book/subject_search?search_text=%s'%isbn
        time.sleep(round(random.uniform(0.5,3),2))
        browser.get(url)
        html = browser.page_source
        try:
            cover_src = re.findall(r'<img src="(.*?)".*class="cover">',html)[0]
            if 'book-default-lpic' in cover_src:
                print("*** No cover for book %s ..."%isbn)
                failures.append(isbn)
            else:
                print("Success for book %s! Image url:"%isbn,cover_src)
                response = requests.get(cover_src)
                with open("covers/%s.jpg"%isbn,"wb") as f :
                    f.write(response.content)
        except Exception as e:
            print('***ERROR for book %s:\n'%isbn,e)
            failures.append(isbn)
    print('------------Num of failures:',len(failures))
    return failures
            

# 豆瓣无法爬取的，就通过京东爬取：【考虑之后换成当当网，京东老是冷不丁给我推荐一些奇奇怪怪的东西】
def get_image_from_JD(isbn_list):
    failures = []
    browser = webdriver.Chrome(r'D:\downloads\chromedriver_win32\chromedriver.exe')
    for isbn in isbn_list:
        if '%s.jpg'%isbn in os.listdir('covers'):
            print("Book %s already exists!"%isbn)
            continue
        url = 'https://search.jd.com/Search?keyword=%s'%isbn
        browser.get(url)
        ps = browser.page_source
        if "抱歉" in ps:
            print("Unable to find this book in JD!")
            failures.append(isbn)
            continue
        try:
            item_tag = browser.find_element_by_css_selector("div.p-img>a")
            item_url = item_tag.get_attribute('href')
            browser.get(item_url)
            image_tag = browser.find_element_by_css_selector("#spec-n1 > img")
            image_url = image_tag.get_attribute('src')
            response = requests.get(image_url)
            with open("covers/%s.jpg"%isbn,"wb") as f :
                f.write(response.content)
            print("Success for book %s! Image url:"%isbn,image_url)
        except Exception as e:
            print('***ERROR for book %s:\n'%isbn,e)
            failures.append(isbn)
    print('------------Num of failures:',len(failures))
    return failures

def get_image_from_DD(isbn_list):
    failures = []
    browser = webdriver.Chrome(r'D:\downloads\chromedriver_win32\chromedriver.exe')
    for isbn in isbn_list:
        if '%s.jpg'%isbn in os.listdir('covers'):
            print("Book %s already exists!"%isbn)
            continue
        url = 'http://search.dangdang.com/?key=%s'%isbn
        browser.get(url)
        ps = browser.page_source
        if "抱歉" in ps:
            print("Unable to find this book in DD!")
            failures.append(isbn)
            continue
        try:
            item_tag = browser.find_element_by_css_selector("ul.bigimg > li.line1 > a") 
            item_url = item_tag.get_attribute('href')
            browser.get(item_url)
            image_tag = browser.find_element_by_id("largePic")
            image_url = image_tag.get_attribute('src')
            response = requests.get(image_url)
            with open("covers/%s.jpg"%isbn,"wb") as f :
                f.write(response.content)
            print("Success for book %s! Image url:"%isbn,image_url)
        except Exception as e:
            print('***ERROR for book %s:\n'%isbn,e)
            failures.append(isbn)
    print('------------Num of failures:',len(failures))
    return failures

books = os.listdir('books/外国books')
isbn_list = []
for book in books:
    book_detail = pd.read_excel(r'D:\PythonOK\图书封面影响\books\外国books\%s'%book,header=1)
    isbn_list += list(book_detail.ISBN)
#index = isbn_list.index(9787544272216)
#DB_failures = get_image_from_DB(isbn_list[435:])
DD_failures = get_image_from_DD(isbn_list)
JD_failures = get_image_from_JD(DD_failures)
DB_failures = get_image_from_DB(JD_failures)
                                                 


    








