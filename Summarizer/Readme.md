﻿# Text Summarizer

Text Summarization is an important problem in our fast-growing information age. A user cannot manually look through documents and will require precise summaries of the original text. Text Summarization is the process of obtaining salient information from an authentic text document. The text can be sourced from sources like newspapers, books etc., and then summarized into documents, in order to create the summary that contains the key extracts and the overall meaning of the document. The process of text summarization has two approaches which are, extractive text summarization and abstractive text summarization. In Extractive summarization, the summary is generated by using most important sentences from the document exactly as they appear in it and then concatenating them in shorter form. The extractive summarization technique focuses on choosing how paragraphs, important sentences, phrases etc. produce the original documents in precise form. The impact of sentences is determined based on statistical and linguistic features. In abstractive text summarization the semantic of the original document is understood and then the summary is paraphrased without modifying the meaning of the original text. Our project paper presents some techniques used in extractive summarization method.





1) Summarizer-Training.py

			Program to train on a financial corpus and create IDF mapping and Word Vector mapping, which are stored in - "IDFVal.pkl" and "WordGraph.pkl"

2) Datasets
            Data is also provided in this project folder
			Datasets can be download from below path as well.
			https://github.com/Abhishek-1/SummarizerData/tree/master/SummarizerData   

3) IDFVal.pkl

			File containing IDF mapping for financial documents
			
4) WordGraph.pkl

			File containing WordGraph mapping for financial documents
			
5) Summarize-With-TfIdf.py

			Program to create Summary of given Document using TFIDF, it uses "IDFVal.pkl" created from training.
			Summary output is displayed and is stored in  "summaryTfIdf.txt"
			
6) Summarize-With-PageRank.py

			Program to create Summary of given Document using PageRank, it uses "WordGraph.pkl" created from training
			Summary output is displayed and is stored in  "SummaryPageRank.txt"
			
7) stopwords.txt

			File containing list of stopwords
			
8) MetricEvaluation-TfIdf.py

			Program to calculate Rouge-1, Rouge-2, Rouge-3 metric for TfIdf summarizer
			
9) MetricEvaluation-PageRank.py

			Program to calculate Rouge-1, Rouge-2, Rouge-3 metric for PageRank summarizer
			
10) input-article.txt

			Containing input article which needs to be summarized

11) Rouge-Metric.txt

			Contains rouge score for both summaries



/********************************************************************/
--------How to Run------



1> Place article to be summarized in "input-article.txt"

2> To create summary based on TfIdf, Run "Summarize-With-TfIdf.py"
   Summary output is stored in "summaryTfIdf.txt"

3> To create summary based on PageRank, Run "Summarize-With-PageRank.py"
   Summary output is stored in "SummaryPageRank.txt"
   
   
/********************************************************************/
--------Rouge Score------

Rouge metric - TfIdf

Rogue-1 value 0.5687637501169132
Rogue-2 value 0.4119476266861668
Rogue-3 value 0.4272123725329563



Rouge metric - PageRank

Rogue-1 value 0.700651203639658
Rogue-2 value 0.5452239485606658
Rogue-3 value 0.5218949110954344




