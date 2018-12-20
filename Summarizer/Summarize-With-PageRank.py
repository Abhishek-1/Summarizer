from sklearn.externals import joblib
import re
import nltk.classify.util
from nltk.stem import PorterStemmer
import heapq

sentenceDict = {}
scoreGraph = {}
pageRank = {}


wordGraph = joblib.load('WordGraph.pkl')

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
        
    
def showSummary(inpPageRank, originalSentence):    
    summaryNew = []
    sumSize = int(len(inpPageRank)* 0.15)
    sort = []
    if sumSize == 0:
        sumSize = 1    
    summary_sentences = heapq.nlargest(sumSize, inpPageRank, key=inpPageRank.get)
    sort = sorted(summary_sentences, key=int)
    for i in sort:
        summaryNew.append(originalSentence[i])
    summary = '\n\n '.join(summaryNew)
    print(summary)
    file = open("SummaryPageRank.txt","w")
    file.write(summary)
    file.close()
        

with open('input-article.txt', 'r', encoding="utf8") as inpFile:
    FIleInp = inpFile.read()
    sentenceListOrig = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', FIleInp)
    
outSentenceDict = tokenizeInput(sentenceListOrig)
outGraph = createScoreGraph(outSentenceDict)
resGraph = calculatePageRank(outSentenceDict, outGraph)
outPageRank = rankSentences(outSentenceDict,resGraph)
showSummary(outPageRank, sentenceListOrig)




    
    
    
    