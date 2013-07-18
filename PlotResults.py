import numpy as np
import os.path
import matplotlib.pyplot as plt
import re
import pickle
from pylab import *

from optparse import OptionParser
from numpy import array

# Parsing input arguments:
###########################################################
op = OptionParser()
# op.add_option("--categories",
#               action="store", type="string", 
#               dest="categories", 
#               help="List of categories")
op.add_option("--path",
              action="store", dest="path",
              help="Path to folder containing datasets.")
op.add_option("--nCategories",
              action="store", dest="nCategories",
              help="Number of categories.")
# op.add_option("--all_categories",default=False,
#               action="store_true", dest="all_categories",
#               help="Whether to use all categories or not.")
# op.add_option("--maxFeatures",
#               action="store", type=int, default=1e6, dest="maxFeatures")
# op.add_option("--stopWordFile",action="store", dest="stopWordFile")


(opts,args) = op.parse_args()
# if len(args) > 0:
#   op.error("this script takes no arguments.")
#   sys.exit(1)

# Gather the no of categories:
nCategories = int(opts.nCategories)

# Gather path to the results:
path = opts.path
abspath = os.path.abspath(path)
if path is None:
  print "Please specify the path to the data folder"
  sys.exit(1)

resultsDir = path + '/Results/'

###########################################################


numFeatureArray = []
for count,filename in enumerate(os.listdir(resultsDir)):
  match = re.match(r"(\S+)-(\d+)",filename)
  numFeatures = int(match.group(2))
  numFeatureArray.append(numFeatures)

numFeatureArray = sorted(array(numFeatureArray))

fileCount = len(os.listdir(resultsDir))
accMatrix = np.ndarray(shape=(nCategories,fileCount),dtype=float)
for count,filename in enumerate(os.listdir(resultsDir)): 
  filePath = resultsDir + filename
  accFile = open(filePath,'rb')
  accuracy = pickle.load(accFile)
  accFile.close()
  categories = accuracy.keys()
  categories = sorted(categories)
  for idx,category in enumerate(categories):
    accMatrix[idx,count] = accuracy[category]

print accMatrix.shape
print numFeatureArray

colormap = plt.cm.Paired
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0,0.9,nCategories)])
print colormap

# fig = plt.figure()
# axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)

for idx in range(0,nCategories):
    plt.plot(np.log(numFeatureArray),accMatrix[idx,:])

plt.legend(categories, ncol=4, loc='upper center', 
           bbox_to_anchor=[0.5, 1.1], 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)

plt.xlabel('# of features')
plt.ylabel('Accuracies')
plt.title('DumbClassifier Results');
plt.show()