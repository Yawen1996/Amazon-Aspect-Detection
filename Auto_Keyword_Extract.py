
# coding: utf-8


from gensim.models import Word2Vec
import pickle
import operator
import functools
from os import walk


class Auto_Keyword_Extract():
    
    def __init__(self):
        self.model = Word2Vec.load("dataset/word2vec_model")
        self.category = ['size', 'performance', 'appearance', 'quality', 'functionality', 'price', 'feeling', 'service', 'target', 'food']
        with open('dataset/general_dictionary.pkl','rb') as f:
            self.dictionary_general = pickle.load(f)
        self.dictionary_specific = self.seed()
    
    def seed(self):
        mypath = 'dataset/Specific'
        tmp = []
        for root, dirs, files in walk(mypath):
            for i in files:
                with open('dataset/Specific/%s' % i,'rb') as f:
                    tmp.append(pickle.load(f))

        dictionary_specific = {}
        for i in self.category:
            a = [x[i] for x in tmp if i in x.keys()]
            if a != []:
                dictionary_specific[i] = list(set([item for sublist in a for item in sublist]))
        return dictionary_specific
    
    def specific_term(self, threshold):
        result = []
        for i in self.valid_IAC:   
            if i in self.model.wv.vocab:
                number = self.Similarity_count(i, 0.6)
                max_number = max(number.items(), key=operator.itemgetter(1))[0]
                if max(number.items(), key=operator.itemgetter(1))[1] > threshold:
                    print(i + ': ' + max_number)
                    result.append((i, max_number))
        return result
    
    def Similarity_count(self, term, threshold):
        number = {}
        for i in self.category:
            tmp = []
            if i in self.dictionary_specific.keys():
                tmp += self.dictionary_specific[i]
            if i in self.dictionary_general.keys():
                tmp += self.dictionary_general[i]

            tmp = list(set(tmp))
            count = 0
            for j in tmp:
                if j in self.model.wv.vocab:
                    if self.model.similarity(term, j) >= threshold:
                        count += 1
            number[i] = count
        return number
    
    def dic(self, result):
        dictionary = {}
        for i in self.category:
            tmp = [x[0] for x in result if x[1] == i]
            if tmp != []:
                dictionary[i] = list(set(tmp))
        return dictionary
    
    def Valid_IAC(self, IAC):
        self.valid_IAC = []
        for i in IAC:
            if i in self.model.wv.vocab:
                self.valid_IAC.append(i)
              
    def AutoLabeling(self, input_path):
        with open(input_path, 'rb') as f:
            IAC = pickle.load(f)
        
        self.Valid_IAC(IAC)
            
        threshold = 4
        result = self.specific_term(threshold)
        self.dictionary = self.dic(result)
        self.dictionary.pop('food', None)
        
    def Save(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.dictionary,f)
        
        

