
# coding: utf-8

import csv
import pickle
from collections import Counter
import pandas as pd
import math
import numpy as np
import functools
from stanfordcorenlp import StanfordCoreNLP
import operator
import re
from os import walk
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import warnings
warnings.filterwarnings("ignore")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from datetime import datetime, timedelta
import random
import calendar
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from tqdm import tqdm

class Analyze():
    def __init__(self):
        self.product = None
        self.product_comment = None
        self.dictionary_specific = None
        self.start_time = None
        self.end_time = None
        self.dataset = []
        self.col = ['Name', 'Brand', 'Price', 'Title', 'Score', 'Time', 'Text', 'Product']
        self.category = ['size', 'performance', 'appearance', 'quality', 'functionality', 'price', 'feeling', 'service', 'target']
        self.sid = SentimentIntensityAnalyzer()
        self.wordnet_lemmatizer = WordNetLemmatizer()
        with open('./dataset/general_dictionary.pkl','rb') as f:
            self.dictionary_general = pickle.load(f)
            
    # load product
    def load_product(self, product):
        self.product = product
        with open('{}/dictionary.pkl'.format(product), 'rb') as f:
            self.dictionary_specific = pickle.load(f)
            
        wb = load_workbook('{}/{}.xlsx'.format(product, product))
        sheet = wb.active
        count = 0
        for row in sheet.rows:
            if count == 0:
                count += 1
                continue
            d = {}
            name = 0
            for cell in row:
                d[self.col[name]] = cell.value
                name += 1
            self.dataset.append(d)
        self.dataset = [x for x in self.dataset if x['Text'] != '']
        self._process_datatime(self.dataset)

    # list all product name
    def list_product(self):
        assert len(self.product) > 0 
        product_name = list(set([x['Name'] for x in self.dataset]))
        for i in product_name:
            print(i)
    
    # list all brand of the product
    def list_brand(self):
        assert len(self.product) > 0 
        brand_list = list(set([x['Brand'] for x in self.dataset if len(x['Text'])!=0]))
        for i in brand_list:
            print(i)
    
    # list all feature which can be analyzed
    def list_feature(self):
        print(self.category)
    
    # change the format of time in reviews
    def _process_datatime(self, data):
        # 將無日期的隨機給日期，並轉換成datetime
        # 2016起
        unique = list(set([x['Time'] for x in data])-set(['The manufacturer commented on the review below'])-set(['']))
        for i in data:
            if i['Time'] == 'The manufacturer commented on the review below' or i['Time'] == '':
                i['Time'] = random.choice(unique)
            i['Time'] = datetime.strptime(i['Time'], '%B %d, %Y')

        data = [x for x in data if x['Time']>=datetime(2016, 1, 1)]
    
    # specify start time & end time
    def set_time_interval(self, start, end):
        assert len(start) == 3
        assert len(end) == 3
        self.start_time = datetime(*start)
        self.end_time = datetime(*end)
        
    # exrtact keyword from specific product name (flag=0) or specific brand (flag=1)        
    def keyword(self, feature, A, flag=0, time_interval=False):
        dataset = self.dataset
        if feature not in self.dictionary_specific.keys():
            k = self.dictionary_general[feature]
        elif feature not in self.dictionary_general.keys():
            k = self.dictionary_specific[feature]
        else:
            k = self.dictionary_general[feature] + self.dictionary_specific[feature]
            
        # Add time interval
        tmp = []
        if time_interval == True:
            assert self.start_time is not None
            assert self.end_time is not None
            tmp = [x for x in dataset if x['Time'] <= self.end_time]
            tmp = [x for x in tmp if x['Time'] >= self.start_time] 
        else:
            tmp = dataset
        
        if flag == 0:
            sentences = [x['Text'].lower() for x in tmp if x['Name'] == A]
        else:
            sentences = [x['Text'].lower() for x in tmp if x['Brand'] == A]# and len(x['Text']) > 0]
        
        try:
            assert len(sentences) > 0
            compare = pd.DataFrame({'term':list(set(k))})  
            tf_counter, df_counter = counter(sentences, self.wordnet_lemmatizer)
            compare['tf'] = compare.apply(lambda x:tf_counter[x['term']], axis = 1)
            compare['sent'] = 0
            compare['not'] = 0
            rank = compare.sort_values(by = 'tf', ascending = False)
            rank = sentiment(rank[rank['tf']>0], sentences, self.sid, self.wordnet_lemmatizer)

            return rank
        except:
            if flag == 0:
                print(A + ' does not exist in product list.')
            else:
                print(A + ' does not exist in brand list.')

    # compare between brands
    def two_brand_compare(self, brand_A, brand_B):
        A = self._time_score(brand_A)
        B = self._time_score(brand_B)
        
        plt.figure()
        plt.xticks(range(7), ['2016/06', '2016/12', '2017/06', '2017/12',
                             '2018/06', '2018/12', '2019/06'])
        plt.plot(A, '-o', label=brand_A)
        plt.plot(B, '-o', label=brand_B)
        plt.legend(loc='best')

    def _time_score(self, target_brand):
        data = [x for x in self.dataset if x['Brand']==target_brand]
        data = [x for x in data if x['Time']>=datetime(2016, 1, 1)]
        year = 2016
        month = 6
        all_score = []
        before = datetime(2015, 12, 31)
        while True:
            day = calendar.monthrange(year, month)
            date = datetime(year, month, day[1])
            tmp = [x for x in data if x['Time'] <= date]
            tmp = [x['Text'] for x in tmp if x['Time'] > before]
            before = date
            if tmp != []:
                all_score.append(sentiment_score(tmp, self.sid))
            else:
                all_score.append(0)

            if month == 12:
                year += 1
                month = 6
            else:
                month = 12
            if year == 2019 and month == 12:
                break

        return all_score
    
    def brand_overview(self):
        brand = pd.DataFrame({'Name':[x['Name'] for x in self.dataset],'Brand':[x['Brand'] for x in self.dataset], 'Product':self.product})
        brand = brand.drop_duplicates()
        brand1 = brand[['Brand', 'Product']]
        brand1 = brand1.drop_duplicates()
        brand1 = brand1[brand1['Brand'] != '-']
        
        result = pd.DataFrame(columns = self.category)
        for i in tqdm(list(brand1[brand1['Product']==self.product]['Brand'])):
            tmp = {}
            for j in self.category:
                rank = self.keyword(j, i, 1)
                rank = rank[rank['tf']>0]
                if len(rank) == 0:
                    tmp[j] = '-'
                tmp[j] = [tuple(x) for x in rank.values]
            a = pd.DataFrame({'size':[tmp['size']], 'performance':[tmp['performance']], 'appearance':[tmp['appearance']], 
                              'quality':[tmp['quality']], 'functionality':[tmp['functionality']], 'price':[tmp['price']], 
                              'feeling':[tmp['feeling']], 'service':[tmp['service']], 'target':[tmp['target']]})
            a = a.rename(index={0: i})
            result = pd.concat([result, a])
        return result
    
    def all_brand_compare(self, flag='sentiment', category=None):
        try:
            assert flag == 'sentiment' or flag == 'volume'
            if category == None:
                category = self.category
            score = []
            x_label = list(set([x['Brand'] for x in self.dataset]))
            for i in tqdm(x_label):
                tmp = []
                data = [x['Text'].lower() for x in self.dataset if x['Brand'] == i]
                for j in category:
                    tmp.append(self._all_brand(data, j, flag))
                score.append(tmp)  
            
            x_label = [x.split(' ')[0].lower() for x in x_label]
            plt.figure(figsize=(len(x_label),len(x_label)/2))
#             plt.xticks(range(len(x_label)), x_label)
            for i in range(len(category)):
                plt.plot(x_label, [x[i] for x in score], '-o', label=category[i])
            plt.legend()
            plt.show()
        except:
            print('Flag must be sentiment or volume.')

            
    def _all_brand(self, data, feature, flag):
        if feature not in self.dictionary_specific.keys():
            k = self.dictionary_general[feature]
        elif feature not in self.dictionary_general.keys():
            k = self.dictionary_specific[feature]
        else:
            k = self.dictionary_general[feature] + self.dictionary_specific[feature]
        
        if flag == 'sentiment':
            key_sentence = []
            for i in data:
                sentence = review_to_sentences(i, self.wordnet_lemmatizer)
                key_sentence += [x for x in sentence if any(y in x for y in k)]

            return sentiment_score(key_sentence, self.sid)
        
        elif flag == 'volume':
            count = 0
            for i in data:
                sentence = review_to_sentences(i, self.wordnet_lemmatizer)
                count += len([x for x in sentence if any(y in x for y in k)])
            return count
        
               
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None
    
def counter(sentences, wordnet_lemmatizer):
    nlp = StanfordCoreNLP("./stanford-corenlp-full-2018-10-05")
    tf_counter = Counter()
    df_list = []
    for i in sentences:
        pos = nlp.pos_tag(i)
        token = []
        for j in pos:
            wordnet_pos = get_wordnet_pos(j[1]) or wn.NOUN
            token.append(wordnet_lemmatizer.lemmatize(j[0].lower(),pos=wordnet_pos))

        tf_counter.update([x for x in token])
        df_list += list(set(token)) 

    df_counter = Counter()
    df_counter.update([x for x in df_list])
    nlp.close()
    return tf_counter, df_counter

def review_to_sentences(review, wordnet_lemmatizer):
#     review = review.replace(',','.')
    review = review.replace('.','. ')
    raw_sentences = sent_tokenize(review)
    return raw_sentences

def sentiment(x, dataset, sid, wordnet_lemmatizer):
    for i in range(len(x)):
        keyword = ' '+x.iloc[i]['term']+' '
        review = [k for k in dataset if keyword in k]
        key_sentence = []
        for j in review:
            sentence = review_to_sentences(j, wordnet_lemmatizer)
            key_sentence += [k for k in sentence if keyword in k]
        count = 0
        sent = 0
        for j in key_sentence:
            ss = sid.polarity_scores(j)
            sent += ss['compound']
            j = j.lower()
            if 'n’t' in j or 'not' in j or 'no' in j:
                count += 1
        if len(key_sentence) != 0:
            sent = sent/len(key_sentence)
        else:
            sent = 0
        x['sent'].iloc[i] = sent
        x['not'].iloc[i] = count
    return x

def sentiment_score(x, sid):
    score = 0
    for i in x:
        ss = sid.polarity_scores(i)
        score += ss['compound']
    if len(x) != 0:
        score = score/len(x)
    else:
        score = 0
    return score