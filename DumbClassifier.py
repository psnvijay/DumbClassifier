import numpy
import os.path
import sys
import pickle

from optparse import OptionParser
from LSI import remStopWords
from LSI import getTop
from LSI import loadData

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
  categories = opts.categories. split(',')
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
print 'In training phase:'
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

  print 'Gathering the top frequency words:'
  topWordsDict = getTop(redtdMatrix=redtdMatrix, redWordList=redWordList,
                              topicList=topicListTrain, maxFeatures=maxFeatures)
  print 'Top Frequency words:' + str(topWordsDict[category])
  gatherTopWordsDict.update(topWordsDict)
  print 80 * '*'
###########################################################


# Loading and pre-processing testing data:
###########################################################

print 80 * '*'
print 'In testing phase:'
correctLabelsDict = {}
accuracy = {}
for category in categories:
  correctLabels = []
  print 80 * '*'
  print 'Category:' + category
  categoryPath = abspath + '/' + category
  print 'Loading testing data:'
  (tdMatrixTest,wordListTest,topicListTest) = loadData(dataCategory='test',all_categories=False,
                                                       categories=[category],path=categoryPath)
  print "Done loading testing data."
  
  # Remove stop words from testing data
  print 'Removing Stop Words:'
  (redtdMatrixTest,redWordListTest) = remStopWords(tdMatrix=tdMatrixTest,wordList=wordListTest,
                                                   topicList=topicListTest,
                                                   filename=abspath + stopWordFile)
  
  (wordCount,docCount) = redtdMatrixTest.shape
  print wordCount,docCount
  wordIndex = numpy.array(range(0,wordCount))

  for doc in range(0,docCount):
    maxValue = 0
    modelLabel = categories
    testSample = redtdMatrixTest[:,doc]
    testSample = testSample.toarray()
    logicalWordVector = numpy.zeros(shape=(wordCount,), dtype=numpy.int)
    
    for idx,itr in enumerate(testSample):
      if (itr>0):
        logicalWordVector[idx] = 1
    
    activeWordsInd = wordIndex[logicalWordVector==1]
    activeWords = []
    for val in activeWordsInd:
      activeWords.append(redWordListTest[val])
  
    for count,topic in enumerate(categories):
      commonWords = set(activeWords) & set(gatherTopWordsDict[topic])
      probScore = len(commonWords)/float(maxFeatures)
      if (probScore > maxValue):
        maxValue = probScore
        modelLabel = topic

    correctLabels.append(modelLabel == topicListTest[doc])
  
  acc = sum(correctLabels)/float(len(correctLabels))
  accuracy[category] = acc 
  correctLabelsDict[category] = correctLabels
  print 'Accuracies:' + str(accuracy.items())
  print 80 * '*'

curFile = os.path.basename(__file__)
resultsPath = abspath + '/Results/' + curFile + '-' + str(maxFeatures) 
resFile = open(resultsPath,'wb')
pickle.dump(accuracy,resFile,-1)
resFile.close()
