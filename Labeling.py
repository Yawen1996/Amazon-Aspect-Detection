
# coding: utf-8


from gensim.models import word2vec
from gensim.models import Word2Vec
from scipy.cluster import hierarchy
from scipy.spatial import distance
import pickle
from collections import Counter
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import numpy as np
from os import walk
import operator


class Labeling():
    
    def __init__(self):
        self.model = Word2Vec.load("dataset/word2vec_model")
        self.category = ['size', 'performance', 'appearance', 'quality', 'functionality', 'price', 'feeling', 'service', 'food']
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
    
    def CreatingMatrix(self, input_path):
        with open(input_path, 'rb') as f:
            IAC_list = pickle.load(f)
        self.valid_IAC = []
        for i in IAC_list:
            if i in self.model.wv.vocab:
                self.valid_IAC.append(i)
    
        self.dissimilarity_matrix = []
        for i in tqdm(self.valid_IAC):
            row = []
            for j in self.valid_IAC:
                s = self.model.wv.similarity(i, j)
                row.append(1.0-s)
            self.dissimilarity_matrix.append(row)

        self.dissimilarity_matrix = np.array(self.dissimilarity_matrix)
        for i in range(len(self.dissimilarity_matrix)):
            self.dissimilarity_matrix[i][i] = 0.0

    def Clustering(self):
        dissimilarity = distance.squareform(self.dissimilarity_matrix)
        threshold = 0.4
        linkage = hierarchy.linkage(dissimilarity, method="ward")
        clusters = hierarchy.fcluster(linkage, threshold, criterion="distance")

        cluster_counter = Counter()
        cluster_counter.update([x for x in list(clusters)])
        self.all_cluster = []
        for key, value in cluster_counter.items():
            if cluster_counter[key] > 2:
                index = [i for i, e in enumerate(clusters) if e == key]
                tmp = []
                for i in index:
                    tmp.append(self.valid_IAC[i])
                self.all_cluster.append(tmp)
    
    def Labeling(self):
        self.Clustering()
        label = []
        category = dict(enumerate(self.category))
        category[9] = 'discard'
        for i in self.all_cluster:
            print('類別：0:size/1:performance/2:appearance/3:quality/4:functionality/5:price/6:feeling/7:service/8:target/9:discard')
            num = input('請輸入%s的類別:'%i)
            if num == '':
                num = input('請輸入%s的類別:'%i)
            num_list = num.split()
            convert = []
            for j in num_list:
                label.append((i,category[int(j)]))
        self.BuildDictionary(label)
        self.Auto_Labeling(4)
            
    def Auto_Labeling(self, threshold):
        label_term = [item for sublist in self.dictionary.values() for item in sublist]
        cluster_term = [item for sublist in self.all_cluster for item in sublist]
        no_label = list(set([x for x in self.valid_IAC if x not in label_term and x not in cluster_term]))
        result = []
        for i in no_label:
            if i in self.model.wv.vocab:
                number = self.Similarity_count(i, 0.6)
                max_number = max(number.items(), key=operator.itemgetter(1))[0]
                if max(number.items(), key=operator.itemgetter(1))[1] > threshold:
                    print(i + ': ' + max_number)
                    result.append((i, max_number))
                    
        for i in result:
            if i[1] in self.dictionary.keys():
                tmp = self.dictionary[i[1]]
            else:
                tmp = []
            tmp.append(i[0])
            self.dictionary[i[1]] = tmp
        self.dictionary.pop('food', None)
    
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
    
    def BuildDictionary(self, label):
        self.dictionary = {}
        for i in self.category:
            tmp = [x[0] for x in label if x[1] == i]
            tmp = [item for sublist in tmp for item in sublist]
            self.dictionary[i] = list(set(tmp))
#             synonyms = [] 
#             tmp = [x[0] for x in label if x[1] == i]
#             tmp = functools.reduce(lambda x,y :x+y ,tmp)
#             for j in tmp:
#                 synonyms.append(j)
#                 if len(wn.synsets(j)) >= 1:
#                     for l in wn.synsets(j)[0].lemmas():
#                         synonyms.append(l.name()) 
#                         if l.antonyms(): 
#                             synonyms.append(l.antonyms()[0].name()) 
#             synonyms = list(set(synonyms))
#             self.dictionary[i] = synonyms

    def Add(self, term, feature):
        if feature in self.dictionary.keys():
            tmp = self.dictionary[feature]
        else:
            tmp = []
        for i in term:
            if i not in tmp:
                tmp.append(i)
        self.dictionary[feature] = tmp
    
    def Save(self, output_path):
        with open(output_path,'wb') as f:
            pickle.dump(self.dictionary,f)

