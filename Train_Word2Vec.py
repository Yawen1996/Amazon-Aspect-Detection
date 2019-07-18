
# coding: utf-8

import pickle
from stanfordcorenlp import StanfordCoreNLP
import re
from gensim.models import word2vec
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import csv
from openpyxl import load_workbook
from nltk.tokenize import sent_tokenize


class Train_Word2vec():
    
    def __init__(self):
        self.col = ['Name', 'Brand', 'Price', 'Title', 'Score', 'Time', 'Text']
        self.stops = set(stopwords.words("english"))
        self.load()
        
    def load(self):
        self.nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
        with open('dataset/sentences.pkl', 'rb') as f:
            self.sentences = pickle.load(f)
        self.sentences = list(set(self.sentences))
        self.sentences = self.sentences[:100]
        self.sentence_list = []
        for i in self.sentences:
            self.sentence_list.append(self.review_to_wordlist(i[0].lower()))
        self.nlp.close()
    
    def review_to_wordlist(self, review):
        # Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review)
        # Convert words to lower case and split them
        words = self.nlp.word_tokenize(review_text) 
#         # Optionally remove stop words (false by default)
#         if remove_stopwords:
#             words = [w for w in words if not w in self.stops]
        return words
    
    def add_data(self, input_path):
        self.nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
        dict_list = []
        if '.csv' in input_path:
            with open(input_path, 'r', encoding='utf8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = {i:row[i] for i in col}
                    dict_list.append(d)
        elif '.xlsx' in input_path:
            wb = load_workbook(input_path)
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
                dict_list.append(d)
                
        dict_list = [x for x in dict_list if x['Text'] != '' and x['Text'] != None]        
        sentences = []
        for i in range(len(dict_list)):
            for j in self.review_to_sentences(dict_list[i]['Text']):
                sentences.append(j)
        sentences = [x for x in sentences if len(x) >= 5]
        for i in sentences:
            self.sentence_list.append(self.review_to_wordlist(i.lower()))
        self.nlp.close()
        
    def review_to_sentences(self, review):
    #     review = review.replace(',','.')
        review = review.replace('.','. ')
        raw_sentences = sent_tokenize(review)
        return raw_sentences
    
    def Train(self, num_features, min_word_count, context):         
        num_workers = 4       # Number of threads to run in parallel                                                                             
        downsampling = 1e-3   # Downsample setting for frequent words
        iteration = 3
        print("Training model...")
        self.model = word2vec.Word2Vec(self.sentence_list, workers=num_workers, size=num_features, min_count = min_word_count,                                                      window = context, sample = downsampling, iter = iteration)
        print("Finsih!")
        self.model.init_sims(replace=True)
    
    def Save(self, output_path):
        self.model.save(output_path)
        

