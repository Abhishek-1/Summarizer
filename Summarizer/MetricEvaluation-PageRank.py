from nltk.stem import PorterStemmer
import nltk.classify.util
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from collections import OrderedDict
import collections
import math
import heapq
#import bs4 as bs  
#import urllib.request
from nltk import ngrams
import os
import glob
from pathlib import Path
import re
from sklearn.externals import joblib


wordGraph = joblib.load('WordGraph.pkl')

inpIDFDict = joblib.load('IDFVal.pkl')

"""
sentenceList = []
sentenceListEn = []
wordListDoc = []
wordListVocab = collections.defaultdict(list)
vocabidf = {}
sentence_score = {}
summary = ""
summaryNew = []
articlegram = []
summarygram = []
"""
rogue = []
senListComplete = []
sumListComplete = []

path = "Data\\metric\\News Articles\\business\\*"
summarypath = "Data\\metric\\Summaries\\business\\*"
files = glob.glob(path)
summaryfiles = glob.glob(summarypath)

"""
Allowed POS Tagging
"""
allowedPOS=['J','V','R','N']


"""
Reading StopWords 
"""

with open('stopwords.txt', 'r') as stopFile:
    stops = stopFile.read().split('\n')
        
ps = PorterStemmer()

"""
    Function for Punctuation Removal
"""
def punctuation_remove(word):
    punctuations = '''!()-[]{};:'"\,<>./?@#+=$%^&*_~'''
    # remove punctuation from the string
    no_punct = ""
    for char in word:
        if char not in punctuations and not char.isdigit():
            no_punct = no_punct + char
    return no_punct

"""
    Porter Stemmer
"""
def word_stemming(word):
     return ps.stem(word)
 
"""
    Preprocessing omn Input Sentence 
"""

def sentenceCreation(inputSentenceList, outSentenceList):
    
    for inptemp in inputSentenceList:
        inp = []
        inp = inptemp.split()
        itemListNew = []
        item = []
        itemList = ''
        """POS Tagging"""
        posTagged= nltk.pos_tag(inp)
        for word in posTagged:
            if word[1][0] in allowedPOS:
                item.append(word[0].lower())
        for word in item:
            ## Call to punctuation removal ##
            word = punctuation_remove(word)
            wordNew  = word_stemming(word.lower())
            if wordNew != '' and wordNew not in stops:
                itemListNew.append(wordNew)
        itemList = ' '.join(itemListNew)
        outSentenceList.append(itemList)
 
"""
Tokenizing Input Sentence List and pre processing 

"""
def tokenizeInput(SentenceList):
    iVal = 0
    for sentence in SentenceList:
        wordList = []
        itemList = sentence.split()
        posTagged= nltk.pos_tag(itemList)
        for word in posTagged:
            if word[1][0] in allowedPOS:
                word = punctuation_remove(word[0])
                wordNew  = word_stemming(word.lower())
                if wordNew not in stops and  wordNew != '':
                    wordList.append(wordNew)
        sentenceDict[iVal] = wordList
        iVal += 1
    return sentenceDict
        



"""
Instantiate Score Graph
"""     
def createScoreGraph(inputSentDict):
    for key, val in inputSentDict.items():
        scoreNode = {}
        getList = inputSentDict[key]
        n = len(getList)
        for word in getList:
            if word not in scoreNode.keys():
                scoreNode[word] = (1/n)
        scoreGraph[key] = scoreNode
    return scoreGraph
        
                
"""
Calculate PageRank
""" 

def calculatePageRank(inputSentDict, inputGraph):
    for key, val in inputSentDict.items():
        getList = inputSentDict[key]
        n = len(getList)
        for word in getList:
            wij = 0
            wjk = 0
            nodesum = 0
            if word in  wordGraph.keys():
                listNeighbours = wordGraph[word]
                for item in listNeighbours:
                    wjk = 1
                    if item in wordGraph.keys():
                        nbrsNeighbours = wordGraph[item]
                        for subsequentitems in nbrsNeighbours:
                            wjk += nbrsNeighbours.count(subsequentitems)                        
                    wij = listNeighbours.count(item)                      
                    nodesum += ((wij*inputGraph[key][word])/wjk)         
            inputGraph[key][word] = ((0.85)*(nodesum)) + ((0.15)*(1/n))
    return inputGraph
        
                      

"""
Showing Summary
"""
def rankSentences(inputSentDict, inputGraph):
    for key, val in inputSentDict.items():
        pageRank[key] = 0
        getList = inputSentDict[key]
        for word in getList:
            if word in inputGraph[key].keys():
                pageRank[key] += inputGraph[key][word]
    return pageRank
    
"""
Convert dictionary to List
"""
def dicttoList(inpDict):
    outList = []
    for key,val in inpDict.items():
        outList.append(inpDict[key])
    return outList
        
    

def findmetrics(sentence_score, inpSummary, k):
   
    sumSize = len(inpSummary)
    summary_sentences = heapq.nlargest(sumSize, sentence_score, key=sentence_score.get)
    for i in summary_sentences:
        if len(sentenceListEn[i].split()) > 2:
            kgram = ngrams(sentenceListEn[i].split(), k)
            #print(len(kgram))
            for grams in kgram:
                articlegram.append(grams)
    for item in inpSummary:
        if len(item.split()) > 2: 
            sgram = ngrams(item.split(), k)
            for agrams in sgram:
               summarygram.append(agrams)
        
def calculateRogue(articlegram, summarygram):
    count = 0
    for item in articlegram:
        if item in summarygram:
            count += 1
    qval = len(summarygram)
    rogue.append(count/qval)
    



k = 3
m = 0
for paths in files:
    sentenceList = []
    sentenceListOrig = []
    summaryListOrig = []
    summaryList = []
    summaryListEn = []
    sentenceList = []
    sentenceListEn = []
    wordListDoc = []
    wordListVocab = collections.defaultdict(list)
    vocabidf = {}
    sentence_score = {}
    summary = ""
    summaryNew = []
    articlegram = []
    summarygram = []
    sentenceDict = {}
    tfidfDict = {}
    outSentenceDict = {}
    outSummaryDict = {}
    outTfIdf = {}
    sentenceDict = {}
    scoreGraph = {}
    pageRank = {}
    
    
    filetoOpen = Path(paths)
    summaryfile = Path(summaryfiles[m])
    
    with open(filetoOpen,encoding="utf8") as file:
        FIleInp = file.read()
        FIleInp = FIleInp.strip()
        sentenceListOrig = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', FIleInp)
        
        
    i = 0
    
    for item in sentenceListOrig:
        if item.strip() != '':
            sentenceList.append(item)
        
    with open(summaryfile,encoding="utf8") as file:
        FIleSummaryInp = file.read()
        FIleSummaryInp = FIleSummaryInp.strip()
        summaryListOrig = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', FIleSummaryInp)
    
    for item in summaryListOrig:
        if item.strip() != '':
            summaryList.append(item)
        
    
    sentenceCreation(sentenceList, sentenceListEn)
    sentenceCreation(summaryList, summaryListEn)    
    outSentenceDict = tokenizeInput(sentenceListOrig)
    outGraph = createScoreGraph(outSentenceDict)
    resGraph = calculatePageRank(outSentenceDict, outGraph)
    outPageRank = rankSentences(outSentenceDict,resGraph)
    #outSummaryList = dicttoList(outSummaryDict)
    #outSentenceList = dicttoList(outSentenceDict)
    #n = len(wordListDoc)
    findmetrics(outPageRank, summaryListEn, k)
    calculateRogue(articlegram, summarygram)
    m = m + 1

stat = {}
avgVal = 0
#Overall = {}
stat["0-25"] = 0
stat["25-50"] = 0
stat["50-75"] = 0
stat["75+"] = 0  
stat["100"] = 0
for item in rogue:
    if item < 0.25:
        stat["0-25"] += 1
    elif item >= 0.25 and item < 0.50:
        stat["25-50"] += 1
    elif item >= 0.50 and item < 0.75:
        stat["50-75"] += 1
    elif item >= 0.75 and item < 0.99:
        stat["75+"] += 1
    elif item == 1.00:
        stat["100"] += 1
    avgVal += item

x = avgVal/len(rogue)
print("Rogue value for k = "+ str(k) +" vale = "+ str(x))

        
    
#Overall[k] = stat
    
        

