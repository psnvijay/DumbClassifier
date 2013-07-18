import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt

from optparse import OptionParser
from LSI import remStopWords
from LSI import getTop
from LSI import loadData
from sets import Set
from numpy import array
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import dok_matrix

# Parsing input arguments:
###########################################################
op = OptionParser()
op.add_option("--categories",
              action="store", type="string", 
              dest="categories", 
              help="List of categories")
op.add_option("--path",
              action="store", dest="path",
              help="Path to folder containing datasets.")
op.add_option("--all_categories",default=False,
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--maxFeatures",
              action="store", type=int, default=1e6, dest="maxFeatures")
op.add_option("--stopWordFile",action="store", dest="stopWordFile")


(opts, args) = op.parse_args()
if len(args) > 0:
  op.error("this script takes no arguments.")
  sys.exit(1)
  
# Gather default input arguments:
all_categories = opts.all_categories
maxFeatures = opts.maxFeatures

# Gather path to the datasets:
path = opts.path
abspath = os.path.abspath(path)
if path is None:
  print "Please specify the path to the data folder"
  sys.exit(1)

# Gather all_categories flag:
if all_categories:
  f = open(abspath + '/all-categories/all-categories-list.txt')
  categories = f.read().split('\n')
  print categories
  f.close()
elif len(opts.categories) > 0:
  categories = opts.categories.split(',')
  print 'Categories:' + str(categories)
else:
  print "Please specify either --categories or --all_categories option."
  sys.exit(1)

# Gather stopWordFile:
stopWordFile = opts.stopWordFile
if stopWordFile is None:
  stopWordFile = '/StopWordsLong.txt'

gatherTopWordsDict = {}
###########################################################


# Load and pre-processing training data:
###########################################################
print 80 * '*'
dataStatsDict = {}
for category in categories:
  print 80 * '*'
  print 'Category:' + category
  print 'Loading training data:'
  categoryPath = abspath + '/' + category
  (tdMatrixTrain,wordListTrain,topicListTrain) = loadData(dataCategory='train',all_categories=False,
                                                          categories=[category],path=categoryPath)
  print "Done loading training data."

# Remove stop words: 
  print 'Removing Stop Words:'
  (redtdMatrix,redWordList) = remStopWords(tdMatrix=tdMatrixTrain,wordList=wordListTrain,
                                           topicList=topicListTrain,
                                           filename=abspath + stopWordFile)
  
  redtdMatrix = redtdMatrix.tocsr()
  redWordIndDict = dict(zip(redWordList,range(0,len(redWordList))))
  (redWordCount,docCount) = redtdMatrix.shape
  
  
  print 'Gathering the top frequency words:'
  topWordsDict = getTop(redtdMatrix=redtdMatrix, redWordList=redWordList,
                              topicList=topicListTrain, maxFeatures=maxFeatures)
  print 'Top Frequency words:' + str(topWordsDict[category])
  topWordList = topWordsDict[category]
  gatherTopWordsDict.update(topWordsDict)
  
  topWordCount = len(topWordList)
  topWordsDocMatrix = dok_matrix((topWordCount,docCount),dtype=np.int16)
  for count,word in enumerate(topWordList):
    idx = redWordIndDict[word]
    getAllDocValues = redtdMatrix[idx,:]
    for doc in range(0,docCount):
      topWordsDocMatrix[count,doc] = getAllDocValues[0,doc]
  
  topWordsDocMatrix = topWordsDocMatrix.astype(float)
  avgTopWordsDocMatrix = topWordsDocMatrix.mean(1)
  dataStatsDict[category] = dict(zip(topWordList,avgTopWordsDocMatrix))
     
#   redtdMatrix = redtdMatrix.astype(float)
#   avgRedtdMatrix = redtdMatrix.mean(1)
#   dataStatsDict[category] = dict(zip(redWordList,avgRedtdMatrix))
  
  print 80 * '*'
###########################################################

globalWordSet = Set()
catCount = len(categories)
for category in categories:
  createTempDict = dataStatsDict[category]
  redWordList = createTempDict.keys()
  if globalWordSet is None:
    globalWordSet = Set(redWordList)
  else:
    globalWordSet = Set(globalWordSet) | Set(redWordList)

globalWordSet = sorted(list(globalWordSet))
wordCount = len(globalWordSet)
print 'Global dictionary count:' + str(wordCount)
wordIndDict = dict(zip(globalWordSet,range(0,wordCount)))
catIndexDict = dict(zip(sorted(categories),range(0,catCount)))
avgFreqDataMatrix = dok_matrix((catCount,wordCount),dtype=np.float)
for word in globalWordSet:
  wordIndex = wordIndDict[word]
  for category in categories:
    catIndex = catIndexDict[category]
    createTempDict = dataStatsDict[category]
    if word in createTempDict:
      avgValue = float(createTempDict[word])
    else:
      avgValue = 0.0
    avgFreqDataMatrix[catIndex,wordIndex] = avgValue
  
avgFreqDataMatrix = avgFreqDataMatrix.todense()
lnkgMatrix = linkage(avgFreqDataMatrix,'single')
print lnkgMatrix

plt.figure(101)
plt.title("ascending")
dendrogram(lnkgMatrix,
           color_threshold=1,
           truncate_mode='lastp',
           labels=array(categories),
           distance_sort='ascending',
           orientation='right')

plt.show()