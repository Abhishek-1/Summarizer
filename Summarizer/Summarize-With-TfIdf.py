from sklearn.externals import joblib
import re
import nltk.classify.util
from nltk.stem import PorterStemmer
import heapq

sentenceDict = {}
tfidfDict = {}


inpIDFDict = joblib.load('IDFVal.pkl')

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
Calculate Tf-Idf
"""
def calculateTfIdf(inputDict):
    for key, val in inputDict.items():
        getList = inputDict[key]
        n = len(getList)
        tfidfVal = 0
        for word in getList:
            if word in inpIDFDict.keys():
                tfidfVal += (1/n)*inpIDFDict[word]        
        tfidfDict[key] = tfidfVal
    return tfidfDict
                        

"""
Showing Summary
"""
def showSummary(inpSentenceScore, OriginalSentence):
    summaryNew = []
    sumSize = int(len(inpSentenceScore)* 0.15)
    sort = []
    if sumSize == 0:
        sumSize = 1
    
    summary_sentences = heapq.nlargest(sumSize, inpSentenceScore, key=inpSentenceScore.get)
    sort = sorted(summary_sentences, key=int)
    for i in sort:
        summaryNew.append(OriginalSentence[i])
    summary = '\n\n '.join(summaryNew)
    print(summary)
    file = open("summaryTfIdf.txt","w")
    file.write(summary)
    file.close()
        

with open('input-article.txt', 'r', encoding="utf8") as inpFile:
    FIleInp = inpFile.read()
    sentenceListOrig = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', FIleInp)
    
outSentenceDict = tokenizeInput(sentenceListOrig)
outTfIdf = calculateTfIdf(outSentenceDict)
showSummary(outTfIdf, sentenceListOrig)



    
    
    
    