
# coding: utf-8

from stanfordcorenlp import StanfordCoreNLP
from senticnet.senticnet import SenticNet
import csv
from openpyxl import load_workbook
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pickle


class Get_IAC():
    
    def __init__(self):
        self.col = ['Name', 'Brand', 'Price', 'Title', 'Score', 'Time', 'Text']
        self.sn = SenticNet('en') 
        self.wordnet_lemmatizer = WordNetLemmatizer()
    
    def review_to_sentences(self, review):
    #     review = review.replace(',','.')
        review = review.replace('.','. ')
        raw_sentences = sent_tokenize(review)
        return raw_sentences
        
    def InputData(self, input_path):
        self.dict_list = []
        if '.csv' in input_path:
            with open(input_path, 'r', encoding='utf8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = {i:row[i] for i in col}
                    self.dict_list.append(d)
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
                self.dict_list.append(d)
                
        self.dict_list = [x for x in self.dict_list if x['Text'] != '' and x['Text'] != None]        
        self.sentences = []
        for i in range(len(self.dict_list)):
            for j in self.review_to_sentences(self.dict_list[i]['Text']):
                self.sentences.append(j)
        self.sentences = [x for x in self.sentences if len(x) >= 5]        
          
    def GetIAC(self):
        self.nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
        self.IAC = []
        for i in tqdm(self.sentences):
            dependency = self.nlp.dependency_parse(i)
            token = self.nlp.word_tokenize(i)
            if [x for x in dependency if 'compound' in x] != []:
                for j in [x for x in dependency if 'compound' in x]:
                    token[j[2]-1] = token[j[2]-1] + '-' + token[j[1]-1]
                    token[j[1]-1] = ''
                i = ' '.join(token)

            parse = self.nlp.parse(i)
            dependency = self.nlp.dependency_parse(i)
            pos = self.nlp.pos_tag(i)
            token = []
            for j in pos:
                wordnet_pos = self.get_wordnet_pos(j[1])
                token.append(self.wordnet_lemmatizer.lemmatize(j[0].lower(),pos=wordnet_pos))

            # subject noun relation
            if [x for x in dependency if 'nsubj' in x] != []:
                for j in self.Subject_Noun_Rule(parse, dependency, token, pos):
                    self.IAC.append(j)
            else: # Non subject noun relation
                for j in self.Non_Subject_Noun_Rule(parse, dependency, token, pos):
                    self.IAC.append(j)
        self.nlp.close()
        self.IAC = list(set(self.IAC))

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN
        
    # Additional Rule: 對等連接詞
    def Conj(self, index, dependency, token):
        IAC = []
        index = list(set(index))
        if [x for x in dependency if 'conj' in x] != []:
            conj = [x for x in dependency if 'conj' in x]
            for j in conj:
                if j[1] in index or j[2] in index:
                    if j[1] not in index:
                        IAC.append(token[j[1]-1])
                        index.append(j[1])
                    if j[2] not in index:
                        IAC.append(token[j[2]-1])
                        index.append(j[2])
        return IAC
        
    def Subject_Noun_Rule(self, parse, dependency, token, pos):
        be = ['is','was','am','are','were']
        adv_mod = [x for x in dependency if 'advmod' in x]
        adj_mod = [x for x in dependency if 'amod' in x]
        active_token = token[[x for x in dependency if 'nsubj' in x][0][2]-1] # 主詞
        
        result = []
        index = []
        if adv_mod != [] or adj_mod != []:
            A, B = self.Rule1(adv_mod, adj_mod, active_token, token)
            result += A
            index += B

        #  does not have auxiliary verb   
        if any(k in token for k in be) == False and [x for x in pos if 'MD' in x] == []:
            A, B = self.Rule2(token, pos, dependency, active_token, adv_mod, adj_mod)
            result += A
            index += B

            if [x for x in dependency if 'dobj' in x] != []:
                A, B = self.Rule3(dependency, token, pos)
                result += A
                index += B

            if [x for x in dependency if 'xcomp' in x] != []:
                A, B = self.Rule4(dependency, token, pos)
                result += A
                index += B  

        if [x for x in dependency if 'cop' in x] != []:
            A, B = self.Rule5(dependency, pos, active_token, token)
            result += A
            index += B

        result += self.Conj(index, dependency, token)
        return list(set(result))
    
    # 3.3.3 Rule 1 
    def Rule1(self, adv_mod, adj_mod, active_token, token):
        IAC = []
        index = []
        if adv_mod != []:
            for j in adv_mod:
                try:
                    concept = self.sn.concept(token[j[2]-1])
                    IAC.append(token[j[2]-1])
                    index.append(j[2])
                except:
                    a = 0
    #                 print(token[j[2]-1] + ' Not in SenticNet')
        if adj_mod != []:
            for j in adj_mod:
                try:
                    concept = self.sn.concept(token[j[2]-1])
                    IAC.append(token[j[2]-1])
                    index.append(j[2])
                except:
                    a = 0
    #                 print(token[j[2]-1] + ' Not in SenticNet')
        return IAC, index

    # 3.3.3 Rule 2-1
    def Rule2(self, token, pos, dependency, active_token, adv_mod, adj_mod):
        IAC = []
        index = []
        advcl = [x for x in dependency if 'advcl' in x] # adverbial clause modifier
        if advcl != []:
            for j in advcl:
                IAC.append(token[j[1]-1])
                index.append(j[1])
                IAC.append(active_token)
                index.append([x for x in dependency if 'nsubj' in x][0][2])

        if adv_mod != []:
            for j in adv_mod:
                IAC.append(token[j[1]-1])
                index.append(j[1])
                IAC.append(active_token)
                index.append([x for x in dependency if 'nsubj' in x][0][2])

        if adj_mod != []:
            for j in adj_mod:
                IAC.append(token[j[1]-1])
                index.append(j[1])
                IAC.append(active_token)
                index.append([x for x in dependency if 'nsubj' in x][0][2])

        return IAC, index

    # 3.3.3 Rule 2-2 & 2-3
    def Rule3(self, dependency, token, pos):
        IAC = []
        index = []
        dobj = [x for x in dependency if 'dobj' in x] #  open clausal complement
        for j in dobj:
            if pos[j[2]-1][1] == 'NN':
                try:
                    # Rule 2-3
                    concept = self.sn.concept(token[j[2]-1]) 
                    IAC.append(token[j[2]-1])
                    index.append(j[2])
                    conj = []
                    conj.append(j[2])
                    if [x for x in dependency if 'conj' in x and j[2] in x] != []:
                        for i in [x for x in dependency if 'conj' in x and j[2] in x]:
                            conj.append(i[1])
                            conj.append(i[2])
                    conj = list(set(conj))
                    for i in conj:
                        t1 = i
                        connect = [x for x in dependency if t1 in x]
                        for k in connect:
                            if k[1] != t1:
                                if pos[k[1]-1][1] == 'NN':
                                    IAC.append(token[k[1]-1])
                                    index.append(k[1])
                            if k[2] != t1:
                                if pos[k[2]-1][1] == 'NN':
                                    IAC.append(token[k[2]-1])
                                    index.append(k[2])
                except:
                    # Rule 2-2
                    IAC.append(token[j[2]-1])
                    index.append(j[2])
    #                 print(token[j[2]-1] + ' Not in SenticNet')
        return IAC, index

    # 3.3.3 Rule 2-4
    def Rule4(self, dependency, token, pos):
        IAC = []
        index = []
        xcomp = [x for x in dependency if 'xcomp' in x] #  open clausal complement
        for j in xcomp:
            try:
                concept = self.sn.concept(token[j[1]-1] + '-' + token[j[2]-1]) 
                IAC.append(token[j[1]-1] + '-' + token[j[2]-1])
            except:
                a = 0
    #             print(token[j[1]-1] + '-' + token[j[2]-1] + ' Not in SenticNet')
            t1 = j[2]
            connect = [x for x in dependency if t1 in x]
            for k in connect:
                if pos[k[2]-1][1] == 'NN':
                    IAC.append(token[k[2]-1])
                    index.append(k[2])
        return IAC, index

    # 3.3.3 Rule 3 & 3.3.3 Rule 4 & 3.3.3 Rule 5
    def Rule5(self, dependency, pos, active_token, token):
        IAC = []
        index = []
        cop = [x for x in dependency if 'cop' in x] # copula
        # Rule 4
        if pos[[x for x in dependency if 'nsubj' in x][0][2]-1][1] == 'NN':
            IAC.append(active_token)
            index.append([x for x in dependency if 'nsubj' in x][0][2])

        # Rule 3 & Rule 5
        for j in cop:
            # Rule 3 
            conj = []
    #         if token[j[1]-1] in all_term:
            IAC.append(token[j[1]-1])
            index.append(j[1])
            conj.append(j[1])
            if [x for x in dependency if 'conj' in x and j[1] in x] != []:
                for i in [x for x in dependency if 'conj' in x and j[1] in x]:
                    conj.append(i[1])
                    conj.append(i[2])

            # Rule 5
            conj = list(set(conj))
            for i in conj:
                t1 = i
                connect = [x for x in dependency if t1 in x]
                for k in connect:
                    if k[1] != t1:
                        if pos[k[1]-1][1] == 'VB' or pos[k[1]-1][1] == 'VV':
                            IAC.append(token[k[1]-1])
                            index.append(k[1])
                            if token[t1-1] not in IAC:
                                IAC.append(token[t1-1])
                                index.append(t1)
                    if k[2] != t1:
                        if pos[k[2]-1][1] == 'VB' or pos[k[2]-1][1] == 'VV':
                            IAC.append(token[k[2]-1])
                            index.append(k[2])
                            if token[t1-1] not in IAC:
                                IAC.append(token[t1-1])
                                index.append(t1)
        return IAC, index
    
    def Non_Subject_Noun_Rule(self, parse, dependency, token, pos):
        result = []
        index = []
        if [x for x in dependency if 'xcomp' in x] != []:
            A, B = self.Rule6(dependency, token)
            result += A
            index += B

        if [x for x in dependency if 'case' in x] != []:
            A, B = self.Rule7(dependency, pos, token)
            result += A
            index += B

        if [x for x in dependency if 'dobj' in x] != []:
            A, B = self.Rule8(dependency, token)
            result += A
            index += B

        result += self.Conj(index, dependency, token)
        return list(set(result))

    # 3.3.4 Rule 1
    def Rule6(self, dependency, token):
        IAC = []
        index = []
        xcomp = [x for x in dependency if 'xcomp' in x] #  open clausal complement
        for j in xcomp:
    #         if token[j[1]-1] in all_term:
            IAC.append(token[j[1]-1])
            index.append(j[1])
        return IAC, index

    # 3.3.4 Rule 2
    def Rule7(self, dependency, pos, token):
        IAC = []
        index = []
        case = [x for x in dependency if 'case' in x] #  a prepositional relation
        for j in case:
            if pos[j[1]-1][1] == 'NN':
                connect = [x for x in dependency if j[1] in x and 'mod' in x[0]]
                for i in connect:
                    IAC.append(token[i[1]-1])
                    IAC.append(token[i[2]-1])
                    index.append(i[1])
                    index.append(i[2])
        return list(set(IAC)), list(set(index))

    # 3.3.4 Rule 3
    def Rule8(self, dependency, token):
        IAC = []
        index = []
        dobj = [x for x in dependency if 'dobj' in x] #  a direct object relation
        for j in dobj:
            IAC.append(token[j[2]-1])
            index.append(j[2])
        return IAC, index
    
    def Save(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.IAC, f)


