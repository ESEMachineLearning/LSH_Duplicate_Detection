#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:04:40 2022

@author: annemiekevisser
"""

import json
import math

import numpy as np
import pandas as pd
import itertools
import collections
import random
import re
import string
from collections import Counter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from random import shuffle
from collections import Counter
from itertools import combinations
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
import warnings

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import time
import json
import sys
from itertools import chain, combinations
from difflib import SequenceMatcher
import operator
from sklearn.cluster import AgglomerativeClustering
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

with open('/Users/annemiekevisser/Documents/Master BAQM/Computer Science/TVs-all-merged.json', 'r') as file:
    data_tv = json.load(file)
file.close()

# Data cleaning
# normalize hertz
def normalizeHertz(word):
    for i in ['Hertz', 'hertz', 'Hz', 'HZ', ' hz', '-hz', 'hz', '/hz']:
        data = word.replace(i, "hz")
        return data

# normalize inch
def normalizeInch(word):
    for i in ['"', 'Inch', '-inch', ' inch', 'inch', 'inches']:
        data = word.replace(i, "inch")
        return data

# replace upper-case characters by lower-case characters
def lowerCaseCharacters(word):
    data = word.lower()
    return data

def removeinterpunction(word):
    data = word.replace('/', "")
    return data

def removeinterpunction2(word):
    data = word.replace('-', "")
    return data

def removeinterpunction3(word):
    data = word.replace('#', "")
    return data

def removeinterpunction4(word):
    data = word.replace('&', "")
    return data

def removeinterpunction5(word):
    data = word.replace(';', "")
    return data

def removeWords(word, delete):
    strlist = word.split()
    data = list(set(strlist) - set(delete))
    return (data)

def cleanstring(string):
    result1 = lowerCaseCharacters(string)
    result2 = normalizeHertz(result1)
    result3 = normalizeInch(result2)
    result4 = removeinterpunction(result3)
    result5 = removeinterpunction2(result4)
    result6 = removeinterpunction3(result5)
    result7 = removeinterpunction4(result6)
    result8 = removeinterpunction5(result7)
    todel = ['newegg', 'neweggcom', 'bestbuycom', "thenerds.net", "nerds", 'buy', 'best', 'bestbuy', 'amazon', 'com',
              'amazoncom',
              'tv',
              'the', 'newegg.com', 'amazon.com', 'bestbuy.com']
    result9 = removeWords(result8, todel)
    return result9

#clean the data and extract all the model words.
df = pd.DataFrame()
keys = []
model_id = []
shop = []
titles = []
featuresMap = []

for key in data_tv.keys():
    for i in range(len(data_tv[key])):
        keys.append(key)
        titles.append(cleanstring(data_tv[key][i]['title']))
        model_id.append(data_tv[key][i]['modelID'])
        shop.append(data_tv[key][i]['shop'])
        listfeatures = []
        values = [cleanstring(item)[0] for item in data_tv[key][i]['featuresMap'].values()]
        for value in values:
            listfeatures.append(value)
        featuresMap.append(listfeatures)

# take out correct characters from titles


MWtitles = []
for i in range(len(titles)):
    MW = re.findall('((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9]*)', str(titles[i]))
    MWtitles.append(MW)

MWpairs = []
MWpairs2 = []
for i in range(len(featuresMap)):
    # results = re.findall("(ˆ\d+(\.\d+)?[a-zA-Z]+$|ˆ\d+(\.\d+)?$)", str(featuresMap[i]))
    MW2 = re.findall("((?:[a-zA-Z]+[\x21-\x7E]+[0-9]|[0-9]+[\x21-\x7E]+[a-zA-Z])[a-zA-Z0-9]*)", str(featuresMap[i]))
    MWpairs2.append(MW2)

for i in range(len(MWpairs2)):
    result = re.findall(r'\d+', str(MWpairs2[i]))
    MWpairs.append(result)

titleplusfeatures = []
for i in range(len(titles)):
    titleplusfeatures.append(MWtitles[i] + MWpairs[i])

allmodelwords2 = []
allmodelwords3 = []
allmodelwords = []
for i in range(len(titleplusfeatures)):
    # results1 = re.findall(" (\d+\.?\d+) "  , str(combinedtitlefeature[i]))
    results = (titleplusfeatures[i])
    allmodelwords2.append(results)
    # results2 = re.findall('[a-zA-Z\s]', str(titlewithmodelwords[i]))
    # titlewithmodelwords.remove(results2)
# ("(\d+\.\d+) +$| r'\d+'$"
for i in range(len(allmodelwords2)):
    result = list(dict.fromkeys(allmodelwords2[i]))
    allmodelwords.append(result)





# find brands and add to model words
brands = []
allbrands = []
for key in data_tv.keys():
    for i in range(len(data_tv[key])):
        if ('Brand' in data_tv[key][i]['featuresMap'].keys()) == True:
            brands.append(data_tv[key][i]['featuresMap']['Brand'])
        else:
            brands.append('a')

for i in range(len(brands)):
    allbrands.append(lowerCaseCharacters(brands[i]))

# remove duplicates
brandsclean = [*set(allbrands)]
brandsclean.remove('a')

brandfromtitle = {}
for i in range(len(titles)):
    brandfromtitle[i] = 0
    tit = titles[i]
    for brand in brandsclean:
        if brand in tit:
            brandfromtitle[i] = brand
            break

titlebrandslist = list(brandfromtitle.values())
combi_brands = []

for i in range(len(titlebrandslist)):
    if titlebrandslist[i] != 0:
        combi_brands.append(titlebrandslist[i])
    else:
        combi_brands.append(brands[i])
combi_brands = [i if i != 'a' else 0 for i in combi_brands]

# add brands to model words
#for i in range(len(allmodelwords)):
   # allmodelwords[i].append(combi_brands[i])

title = allmodelwords

# create Dataframa
df['keys'] = model_id
df['shop'] = shop
df['title'] = title #this is now title and features combinened, so all the model words per product.
df['featuresMap'] = featuresMap




# signature matrix
def createSignatureMatrix(binary_matrix, n_minhashes):
    n_minhashes = round((len(final_word_list)) / 2)
    N, M = binary_matrix.shape
    binary_matrix[binary_matrix == 0] =  math.inf
    sig_mat = np.zeros((n_minhashes, N), dtype=int)
    hashObj = list(range(0, M))
    for n in range(n_minhashes):
        shuffle(hashObj)
        sig_mat[n] = np.transpose(np.nanargmin(binary_matrix * hashObj, axis=1))
    return sig_mat

# LSH
def lsh(sig_mat, b, r):
    n_hash, n_products = sig_mat.shape  # number of hashkeys

    bands = np.array_split(sig_mat, b, axis=0)
    potentialPairs = set()
    for band in bands:
        hashDict = {}
        for item, column in enumerate(band.transpose()):
            hash_col = column.tobytes()
            # print(hash_col)
            if hash_col in hashDict:
                # print(hashDict[hash_col])
                hashDict[hash_col] = np.append(hashDict[hash_col], item)
            else:
                hashDict[hash_col] = np.array([item])
        for potential_pair in hashDict.values():
            if len(potential_pair) > 1:
                for i, item1 in enumerate(potential_pair):
                    for j in range(i + 1, len(potential_pair)):
                        if (potential_pair[i] < potential_pair[j]):
                            potentialPairs.add(tuple(sorted((potential_pair[i], potential_pair[j]))))
    print("potential pairs LSH found =",  len(potentialPairs))


    return potentialPairs

def hoi(potentials,data_cleaned):

    delete_key = []
    for pair in potentials:
        i = pair[0]
        j = pair[1]
        if (data_cleaned['shop'][i] == data_cleaned['shop'][j]) or (combi_brands[i] != combi_brands[j]):
            delete_key.append(pair)

    print("potential pairs LSH removed due to same shop or different brand =", len(delete_key))
    return delete_key





def shingles(text, length):
    return [text[i:i + length] for i in range(len(text) - length + 1)]


def jaccardDistance(shWord1, shWord2):
    return len(shWord1.intersection(shWord2)) / (len(shWord1.union(shWord2)) + 0.001)


def createDissimilarityMatrix(candidate_pairs, MWstring, sig_mat, length):
    nItems = len(MWstring)
    result1 = np.full((nItems, nItems), 1000, dtype=float)

    for pair in candidate_pairs:
        item1 = pair[0]
        item2 = pair[1]

        result1[item1, item2] = 1 - jaccardDistance(set(shingles(MWstring[item1], length)),
                                                    set(shingles(MWstring[item2], length)))

        # if brand not the same set simmilarity to 1000
        if (combi_brands[item1] != combi_brands[item2]):
            result1[item1, item2] = 1000
        # if shop the same set similarity to 1000
        if shop[item1] == shop[item2]:
            result1[item1, item2] = 1000


        result1[item2, item1] = result1[item1, item2]

    return result1

def similarityMatrix(candidate_pairs, MWstring, threshold, length):
    nItems = len(MWstring)
    result1 = np.full((nItems, nItems), 1000, dtype=float)

    pairsmeetingthreshold = []
    for pair in candidate_pairs:
        item1 = pair[0]
        item2 = pair[1]

        result1[item1, item2] = 1 - jaccardDistance(set(shingles(MWstring[item1], length)),
                                                    set(shingles(MWstring[item2], length)))

        # if brand not the same set simmilarity to 1000
        if (combi_brands[item1] != combi_brands[item2]):
            result1[item1, item2] = 1000
        # if shop the same set similarity to 1000
        if shop[item1] == shop[item2]:
            result1[item1, item2] = 1000


        result1[item2, item1] = result1[item1, item2]
        for item1, item2 in result1[item1,item2]:
            if result1[item1, item2] >= threshold:
                pairsmeetingthreshold.append(result1[item1, item2])

    return result1




def clusterMethod(dis_mat, t):
    linkage = AgglomerativeClustering(n_clusters=None, affinity="precomputed",
                                      linkage='average', distance_threshold=t)
    clusters = linkage.fit_predict(dis_mat)
    dictCluster = {}
    for index, clusternr in enumerate(clusters):
        if clusternr in dictCluster:
            dictCluster[clusternr] = np.append(dictCluster[clusternr], index)
        else:
            dictCluster[clusternr] = np.array([index])
    candidate_pairs = set()
    for potential_pair in dictCluster.values():
        if len(potential_pair) > 1:
            for i, item1 in enumerate(potential_pair):
                for j in range(i + 1, len(potential_pair)):
                    if (potential_pair[i] < potential_pair[j]):
                        candidate_pairs.add((potential_pair[i], potential_pair[j]))
                    else:
                        candidate_pairs.add((potential_pair[j], potential_pair[i]))
    return candidate_pairs



def truedup(data_cleaned):
    trueDup = []
    #numberDuplicates = 0
    for i in list(data_cleaned.index):
        for j in list(data_cleaned.index):
            if i >= j:
                continue
            else:
                if data_cleaned['keys'][i] == data_cleaned['keys'][j]:
                    pairs = tuple([i, j])
                    trueDup.append(pairs)
                    #numberDuplicates += 1
                    #print('numberDuplicates =', numberDuplicates)

    return trueDup
#trueDup = truedup(df)
# true positives



def evaluateResults(candidatepairs, candidate_pairsCluster, data_cleaned, t):

    #lsh
    tpLSH = 0
    for pair in candidatepairs:
        if data_cleaned['keys'][pair[0]] == data_cleaned['keys'][pair[1]]:
            tpLSH = tpLSH + 1
    print("true positives found LSH =  ", tpLSH)
    #cluster
    tpCluster = 0
    for pair in candidate_pairsCluster:
        if data_cleaned['keys'][pair[0]] == data_cleaned['keys'][pair[1]]:
            tpCluster = tpCluster + 1

    # for pair in candidate_pairsLSH:
    #   if keys[pair[0]] == keys[pair[1]]:
    #      tpLSH = tpLSH + 1

    numberDuplicates = len(truedup(data_cleaned))
    numberCandidatesLSH = len(candidatepairs) + 0.00001

    tpandfn = len(candidate_pairsCluster) + 0.000001
    PC = tpLSH / numberDuplicates

    PQ = tpLSH / numberCandidatesLSH
    F1StarLSH = (2 * (PQ * PC)) / (PQ + PC + 0.000001)

    precision = tpCluster / numberDuplicates
    recall = tpCluster / tpandfn
    F1 = 2 * (precision * recall) / (precision + recall + 0.000001)
    N = len((data_cleaned))
    totalComparisons = (N * (N - 1)) / 2
    fraction = min(len(candidatepairs) / totalComparisons, 1)
    print("total comparisons = ", totalComparisons)
    print("fraction =  ", len(candidatepairs) / totalComparisons )
    print("PC = ", PC)
    print("PQ = ", PQ)
    print("F1 = ", F1)
    print("F1starLSH = ", F1StarLSH)
    print("recall = ", recall)
    print("precision = ", precision)
    return [PC, PQ, F1StarLSH, precision, recall, F1, fraction, t]









#create dataframe for results

diffRowsresults = pd.DataFrame(columns=['PC', 'PQ', 'F1*', 'Precision', 'Recall', 'F1', 'fraction', 't'])

PC = []
PQ = []
F1_Star_LSH = []
Precision = []
Recall = []
F1 = []
Fraction = []
t = []

boot_it = 5
print('Now starting ' + str(boot_it) + ' bootstrap iterations')
for i in range(boot_it):
    print("|------------------------------------------ Bootstrap " + str(
        i + 1) + " ------------------------------------------|")
    indices_keep = []
    for j in range(len(df)):
        rand = random.randint(1, len(df))
        if rand not in indices_keep:
            indices_keep.append(rand)
    indices_delete = []
    for k in range(len(df)):
        if k not in indices_keep:
            indices_delete.append(k)
    data_cleaned = df.drop(indices_delete)
    data_cleaned = data_cleaned.reset_index(drop=True)
    print("Bootstrap with " + str(len(data_cleaned)) + " randomly selected products, equal to " + str(
        round((len(data_cleaned) / len(df)) * 100)) + '% of original Dataset')



    max_threshold = 1000
    thresholdmet = []
    numberMW = pd.Series(Counter([y for x in data_cleaned['title'] for y in x]))
    countwordsdf = numberMW.to_frame()
    countwordsdf.columns = ["frequency"]

    for indices in countwordsdf.index:
        if countwordsdf["frequency"][indices] == 1 or countwordsdf["frequency"][indices] > max_threshold:
            thresholdmet.append(indices)

    # %% Removing words that meet the threshold conditions
    MW_cleanlist = pd.Series.tolist(data_cleaned['title'])
    delete = thresholdmet

    for i in range(len(MW_cleanlist)):
        for word in MW_cleanlist[i]:
            if word in delete:
                MW_cleanlist[i].remove(word)

    data_cleaned['title'] = MW_cleanlist

    wordlist = []

    for i in range(len(MW_cleanlist)):
        for word in MW_cleanlist[i]:
            wordlist.append(word)

    # word list without duplicates
    final_word_list = []
    for i in wordlist:
        if i not in final_word_list:
            final_word_list.append(i)

    MWstring2 = []
    MWstring = []
    for i in range(len(MW_cleanlist)):
        makeString = ' '.join(map(str, MW_cleanlist[i]))
        MWstring2.append(makeString)

    for j in range(len(MWstring2)):
        makeString2 = MWstring2[j].replace(" ", "")
        MWstring.append(makeString2)

    print("length final word list =  ", len(final_word_list))
    print("true duplicates present =  ", len(truedup(data_cleaned)))
    rows1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for r in rows1:
        print("r =  ", r)
        #print(r)
        n_minhashes = round((len(final_word_list)) / 2)
        n = n_minhashes
        b = n / r
        b = int(b)
        print("b =  ", b)
        #print(b)
        threshold = (1 / b) ** (1 / r)
        print("treshold =  ", threshold)

        length = 3

        N = len(final_word_list)
        M = len(MW_cleanlist)
        binary_matrix = np.zeros((N, M))

        for i in range(M):
            binary_vector = np.zeros(len(final_word_list))
            for word in MW_cleanlist[i]:
                if word in final_word_list:
                    index = final_word_list.index(word)
                    binary_vector[index] = 1
                else:
                    binary_vector[index] = 0
            binary_matrix[:, i] = binary_vector

        binary_matrix = binary_matrix.transpose()
        # signature matrix
        sig_mat = createSignatureMatrix(binary_matrix, n_minhashes)
        candidate_pairsLSH = lsh(sig_mat, b, r)

        print("candidate_pairsLSH  =", len(candidate_pairsLSH))

        dis_mat = createDissimilarityMatrix(candidate_pairsLSH, MWstring, sig_mat, length)
        candidate_pairsCluster = clusterMethod(dis_mat, threshold)
        AAA = evaluateResults(candidate_pairsLSH, candidate_pairsCluster, data_cleaned, threshold)

        PC.append(AAA[0])
        PQ.append(AAA[1])

        F1_Star_LSH.append(AAA[2])
        Precision.append(AAA[3])
        Recall.append(AAA[4])
        F1.append(AAA[5])
        Fraction.append(AAA[6])
        t.append(AAA[7])
        print('  ')



diffRowsresults['PC'] = PC
diffRowsresults['PQ'] = PQ
diffRowsresults['F1*'] = F1_Star_LSH
diffRowsresults['Precision'] = Precision
diffRowsresults['Recall'] = Recall
diffRowsresults['F1'] = F1
# fraction = [len(listcandidatepairs) / totalpossiblepairs for totalpossiblepairs in listcandidatepairs]
diffRowsresults['fraction'] = Fraction
diffRowsresults['treshold'] = t


#plt.plot(diffRowsresults['fraction'], diffRowsresults['F1'])
#plt.ylabel("F1-measure")
#plt.xlabel("Fraction of comparisons")
#plt.show()