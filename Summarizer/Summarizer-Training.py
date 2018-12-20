import os
import glob
import re
import math
import nltk.classify.util
from nltk.stem import PorterStemmer
from sklearn.externals import joblib

path = 'Data\\dataset\\News Articles\\business\\*'
files = glob.glob(path)
corpus = {}
wordList = []
VocabDict = {}
idfDict = {}
wordNode = {}
ScoreGraph = {}

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
 


for paths in files:
    with open(paths,encoding="utf8") as file:
        fileName = paths.split(os.sep)
        FIleInp = file.read()
        sentenceListOrig = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', FIleInp)
    corpus[fileName[-1]] = sentenceListOrig
    


"""
Tokenizing Corpus and finding total Vocabulary 

"""
def findVocab(totalCorpus):
    for key, val in totalCorpus.items():        
        for sentence in val:
            itemList = sentence.split()
            posTagged= nltk.pos_tag(itemList)
            for word in posTagged:
                if word[1][0] in allowedPOS:
                    word = punctuation_remove(word[0])
                    wordNew  = word_stemming(word.lower())
                    if wordNew not in wordList and wordNew not in stops and  wordNew != '':
                        wordList.append(wordNew)
    return wordList
                        
"""
Creating Empty dictionary
"""
def createVocabDict(vocabList):
    for word in vocabList:
        newList = []
        VocabDict[word] = newList
    return VocabDict
                        
"""
Calculating idf for words
"""

def findidf(totalCorpus, inpVocab):
    for key, val in totalCorpus.items():
        for sentence in val:
            itemList = sentence.split()
            posTagged= nltk.pos_tag(itemList)
            for word in posTagged:
                if word[1][0] in allowedPOS:
                    word = punctuation_remove(word[0])
                    wordNew  = word_stemming(word.lower())
                    if wordNew not in stops and  wordNew != '':
                        if wordNew in inpVocab:
                            getlist = inpVocab[wordNew]
                            if key not in getlist:
                                getlist.append(key)
                                inpVocab[wordNew] = getlist
                        else:
                            getlist = inpVocab[wordNew]
                            getlist.append(key)
                            inpVocab[wordNew] = getlist
    return inpVocab


"""
Calculate IDF
"""
def calculateIDF(vocabulary, totalsize):
    for key, val in vocabulary.items():
        getlist = vocabulary[key]
        n = len(getlist)
        idfval = math.log(totalsize/n)
        idfDict[key] = idfval
    return idfDict
        
        


"""
Starting training for PageRank - Making wordGraph
"""
def createWordGraph(corpus):
    for key, val in corpus.items():
        getfile = corpus[key]        
        prev = ""
        current = ""
        for sentence in getfile:
            contin = 0
            itemList = sentence.split()
            posTagged= nltk.pos_tag(itemList)
            for word in posTagged:
                if word[1][0] in allowedPOS:
                    word = punctuation_remove(word[0])
                    wordNew  = word_stemming(word.lower())
                    if wordNew not in stops and  wordNew != '':
                        contin += 1
                        prev = current
                        current = wordNew
                        if contin >= 2:
                            
                            if current in wordNode.keys():
                                getNbrs = wordNode[current]
                                if prev not in getNbrs:
                                    getNbrs.append(prev)
                                    wordNode[current] = getNbrs
                            else:
                                newList = [prev]
                                wordNode[current] = newList
                                    
                                
                            if prev in wordNode.keys():
                                getNbrs = wordNode[prev]
                                if current not in getNbrs:
                                    getNbrs.append(current)
                                    wordNode[prev] = getNbrs
                            else:
                                newList = [prev]
                                wordNode[current] = newList
                        
                    else:
                        contin = 0
                else:
                    contin = 0
    return wordNode



                        
           
            

"""
Calling function to get the Vocabulary
"""
totalDoc = len(corpus)
outList = findVocab(corpus)
outDict = createVocabDict(outList)  
outvocabDoc = findidf(corpus, outDict)
outIDF = calculateIDF(outvocabDoc, totalDoc)

"""
Using pickle to save Idf Values
"""

joblib.dump(outIDF, 'IDFVal.pkl')

"""
Now PageRank Implementation
"""
outWordGraph = createWordGraph(corpus)

"""
Using pickle to save WordGraph Values
"""

joblib.dump(outWordGraph, 'WordGraph.pkl')





    
    
