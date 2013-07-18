#!/usr/bin/python -tt
# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import numpy
import time
import os.path
import sys
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix

def createTermDocumentMatrix(filename):

  f = open(filename,'rU')
  
  wordList = []
  wordFreqDict = {}
  topicList = []
  
  docCount = 0
  count = 0
  while 1:
    line = f.readline()
    if not line: break
    topicFlag = True
    words = line.split()
    for word in words:
      if topicFlag:
        topicList.append(word)
        topicFlag = False
      else:
        if word not in wordFreqDict:
          wordFreqDict[word] = 1
          wordList.append(word)
          count = count + 1
        else:
          wordFreqDict[word] = wordFreqDict[word] + 1
    docCount = docCount + 1
              
  f.close()
  wordList = sorted(wordList)
  wordCount = len(wordList)
  wordIndDict = dict(zip(wordList,range(0,wordCount)))
  print docCount
  print wordCount
  
  tdMatrix = dok_matrix((wordCount,docCount),dtype=numpy.int16)
  docItr = 0
   
  f = open(filename,'rU')
  while 1:
    line = f.readline()
    
    if not line: break # EOF 
    
    words = line.split()
    for word in words:
      if word not in topicList:
        idx = wordIndDict[word]
        tdMatrix[idx,docItr] = tdMatrix[idx,docItr] + 1

    docItr = docItr + 1
  f.close()
  
  tdMatrix = tdMatrix.tocsr()
  
  return (tdMatrix,wordList,topicList)
 
def getTop(redtdMatrix=None, redWordList=None,
           topicList=None, maxFeatures=1e3):
  
  gatherTopWordsDict = {}
  docIndex = range(0,len(topicList))
  uniqueTopicList = list(set(topicList))
  docDict = {}
  for idx in docIndex:
    topic = topicList[idx]
    if topic not in docDict.keys():
      docDict[topic] = []
      docDict[topic].append(idx)
    else:
      docDict[topic].append(idx)
   
  sortedWordsDict = {}
  topWordsDict = {}
  
  for topic in uniqueTopicList:
    wordTupleList = []
    gatherTopWordsDict[topic] = []
    indices = docDict[topic]
    topicDataMatrix = redtdMatrix[:,indices]
    sumElements = topicDataMatrix.sum(axis=1)
    
    for idx in range(0,len(sumElements)):
      count = sumElements[idx,0]
      wordTupleList.append((redWordList[idx],count))
    
    sortedWordsDict[topic] = sorted(wordTupleList,key=lambda tupleValue:tupleValue[1],reverse=True)
    topWordsTupleList = sortedWordsDict[topic][:maxFeatures]
    
    for word,count in topWordsTupleList:
      gatherTopWordsDict[topic].append(word)
    
    topWordsDict[topic] = sortedWordsDict[topic][:maxFeatures]
  
  return gatherTopWordsDict

def remStopWords(tdMatrix=None,wordList=None,
                 topicList=None,filename=None):
  
  wordIndDict = dict(zip(wordList,range(0,len(wordList))))
  indices = []
  
  f = open(filename)
  stopWords = f.read()
  f.close()
  stopWords = stopWords.split('\n')
  stopWords = sorted(stopWords) # sorted list
  redWordList = list(set(wordList) - set(stopWords))
  redWordList = sorted(redWordList) # sorted list
  print '# of stopWords: ' + str(len(stopWords))
  print '# of reduced words: ' + str(len(redWordList))
  print '# of total words: ' + str(len(wordList))
  for word in redWordList:
    indices.append(wordIndDict[word])

  redtdMatrix = tdMatrix[indices,:]
  
  return (redtdMatrix,redWordList)

def loadData(dataCategory='train',all_categories=False,
             categories=None,path=None):

  if len(sys.argv) == 0:
    print "Atleast one argument needs to be passed."
    sys.exit(1)

  if path is None:
    print "Please specify the path to the data folder"
    sys.exit(1)
  
  if (len(categories) == 0):
    if all_categories:
      categories = ['all-categories']
    else:
      print "Please specify either --categories or --all_categories option."
      sys.exit(1)
  
  for category in categories:
    filename = category + '-' + dataCategory + '.txt'
    abspath = os.path.abspath(path) + '/' + filename
    print abspath
    pathToDataMatrix = abspath + '-tdDataMatrix.npy'
    pathToTopicList = abspath + '-topicList.pkl'
    pathToWordList = abspath + '-wordList.pkl'
    
    if not os.path.exists(abspath):
      print 'File does not exist in the specified path. Please check the existence of the file'
      sys.exit(1)   
  
    else: 
      if not (os.path.exists(pathToDataMatrix) | (os.path.exists(pathToTopicList)) | (os.path.exists(pathToWordList))):
        print 'Creating term document matrix, topic lists and word lists'
        start = time.clock()
        (tdMatrix,wordList,topicList) = createTermDocumentMatrix(abspath)
        print 'Time elapsed in data processing:' + str(time.clock() - start)
      
        print 'Saving the matrices in binary format'
        (wordCount,docCount) = tdMatrix.shape
        print wordCount
        print docCount
        data = tdMatrix.data
        indices = tdMatrix.indices
        indptr = tdMatrix.indptr
        numpy.save(pathToDataMatrix,(data,indices,indptr,wordCount,docCount))
        tlFile = open(pathToTopicList,'wb')
        wlFile = open(pathToWordList,'wb')
        pickle.dump(topicList,tlFile,-1)
        pickle.dump(wordList,wlFile,-1)
        tlFile.close()
        wlFile.close()
      
      else:
        print 'Formatted files already exist. Loading them...:'
        start = time.clock()
        (data,indices,indptr,wordCount,docCount) = numpy.load(pathToDataMatrix)
        tlFile = open(pathToTopicList,'rb')
        wlFile = open(pathToWordList,'rb')
        topicList = pickle.load(tlFile)
        wordList = pickle.load(wlFile)
        tlFile.close()
        wlFile.close()
        print ' Time to load files:' + str(time.clock() - start)
        tdMatrix = csr_matrix((data,indices,indptr),shape=(wordCount,docCount))
    
  return (tdMatrix,wordList,topicList)

# def main():
#   
#   global tdMatrix
#   global redtdMatrix
#   global topicList
#   global wordList
#   
#   # parse commandline arguments
#   op = OptionParser()
#   op.add_option("--dataCategory", type="string",
#                 action="store", dest="dataCategory", default="train",
#                 help="Data category options: 'train' or 'test' ")
#   op.add_option("--categories",
#               action="store", type="string", 
#               dest="categories", 
#               help="List of categories")
#   op.add_option("--path",
#               action="store", dest="path",
#               help="Path to folder containing datasets.")
#   op.add_option("--all_categories",
#               action="store_true", dest="all_categories",
#               help="Whether to use all categories or not.")
#   op.add_option("--maxFeatures",
#               action="store", type=int, default=2 ** 16, dest="maxFeatures")
# 
#   (opts, args) = op.parse_args()
#   if len(args) > 0:
#     op.error("this script takes no arguments.")
#     sys.exit(1)
#   
#   dataCategory = opts.dataCategory
#   
#   if opts.path is None:
#     print "Please specify the path to the data folder"
#     sys.exit(1)
#   
#   if opts.all_categories:
#     categories = ['all-categories']
#   elif len(opts.categories) > 0:
#     categories = opts.categories.split(',')
#     print categories
#   else:
#     print "Please specify either --categories or --all_categories option."
#     sys.exit(1)
#     
#   path = opts.path
#   
#   for category in categories:
#     filename = category + '-' + dataCategory + '.txt'
#     abspath = os.path.abspath(path) + '/' + filename
#     pathToDataMatrix = abspath + '-tdDataMatrix.npy'
#     pathToTopicList = abspath + '-topicList.pkl'
#     pathToWordList = abspath + '-wordList.pkl'
#     
#     if not os.path.exists(abspath):
#       print 'File does not exist in the specified path. Please check the existence of the file'
#       sys.exit(1)   
#   
#     else: 
#       if not (os.path.exists(pathToDataMatrix) | (os.path.exists(pathToTopicList)) | (os.path.exists(pathToWordList))):
#         print 'Creating term document matrix, topic lists and word lists'
#         start = time.clock()
#         createTermDocumentMatrix(filename)
#         print 'Time elapsed in data processing:' + str(time.clock() - start)
#       
#         print 'Saving the matrices in binary format'
#         (wordCount,docCount) = tdMatrix.shape
#         data = tdMatrix.data
#         indices = tdMatrix.indices
#         indptr = tdMatrix.indptr
#         numpy.save(pathToDataMatrix,(data,indices,indptr,wordCount,docCount))
#         tlFile = open(pathToTopicList,'wb')
#         wlFile = open(pathToWordList,'wb')
#         pickle.dump(topicList,tlFile,-1)
#         pickle.dump(wordList,wlFile,-1)
#         tlFile.close()
#         wlFile.close()
#       
#       else:
#         print 'Formatted files already exist. Loading them...:'
#         start = time.clock()
#         (data,indices,indptr,wordCount,docCount) = numpy.load(pathToDataMatrix)
#         tlFile = open(pathToTopicList,'rb')
#         wlFile = open(pathToWordList,'rb')
#         topicList = pickle.load(tlFile)
#         wordList = pickle.load(wlFile)
#         tlFile.close()
#         wlFile.close()
#         print ' Time to load files:' + str(time.clock() - start)
#         tdMatrix = csc_matrix((data,indices,indptr),shape=(wordCount,docCount))
#     
#     remStopWords(os.path.abspath(path) + '/StopWordsLong.txt')
#      
#     getTop(opts.maxFeatures)
#     
#     gatherTopWordsDict = gatherTopWords()
#     
#     uniqueTopicList = list(set(topicList))
#       
#     for i in range(0,len(uniqueTopicList)):
#       print uniqueTopicList[i], topWordsDict[uniqueTopicList[i]]
# 
# if __name__ == '__main__':
#   main()
