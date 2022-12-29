import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

class NaiveBayes:
    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('spanish'))
        self.TR_prob = {} #Probabilidad de ser tweet de turismo
        self.NTR_prob = {}
        self.P_TR = 0 #Probabilidad de que sea un tweet positivo
        self.P_NTR = 0 #Probabilidad de que sea un tweet negativo

    def train(self, train_source):
        df = pd.read_csv(train_source)
        targets = df['TARGET']
        tweets = df['TWEET']
        words_set = set()

        #Creación del conjunto de palabras y cuenta de las repeticiones
        # de cada palabra ya sea positiva o negativa del Turismo
        TR = {}
        NTR = {}
        TR_ones = 0
        NTR_ones = 0
        for tweet, target in zip(tweets, targets):
            words = tweet.split(' ')
            for word in words:
                #print(word)
                words_set.add(word)
                if word in self.stop_words:
                    continue

                if not word in TR:
                    TR[word] = 1
                    NTR[word] = 1

                if target:
                    TR_ones += 1
                    TR[word] += 1

                else:
                    NTR_ones +=1
                    NTR[word] += 1

        # Calculo de probabilidad de cada palabra
        self.TR_prob = {}
        self.NTR_prob = {}

        for word in words_set:
            self.TR_prob[word] = TR[word]/TR_ones
            self.NTR_prob[word] = NTR[word]/NTR_ones

        #Cálculo de la probabilidad de que sea un tweet positivo o negativo
        self.P_TR = sum(np.array(targets.tolist())) / len(targets.tolist())
        self.P_NTR = (len(targets.tolist()) - sum(np.array(targets.tolist()))) / len(targets.tolist())

        acc = 0
        for tweet, target in zip(tweets, targets):
            y = self.classify(tweet)
            if y == target:
                acc += 1

        acc = acc/len(targets)
        return acc
        # print('accuracy del entrenamiento:',acc)

    def classify(self, tweet):
        words = tweet.split(' ')
        NTR_max = self.P_NTR
        TR_max = self.P_TR

        for word in words:
            if word in self.stop_words:
                continue

            if word in self.NTR_prob:
                NTR_max *= self.NTR_prob[word]
                TR_max *= self.TR_prob[word]

        if TR_max > NTR_max:
            return 1
        else:
            return 0
