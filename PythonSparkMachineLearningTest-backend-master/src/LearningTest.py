from __future__ import print_function

from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

import os
# 更新为您实际的 Python 解释器路径
os.environ['PYSPARK_PYTHON'] = 'C:\\Users\\apr2333\\AppData\\Local\\Programs\\Python\\Python39\\python.exe'

# 确保这些路径是正确的
TEST_DATA_PATH = "../dataset/useful_dataset/test"
TEST_MODEL_PATH = "../model"

NUM_OF_CLASSES = 14
NUM_OF_TREES = 11 #
MAXDEPTH = 20 # 10-100
MAXBINS = 32 ##
def train():
	data = MLUtils.loadLibSVMFile(sc,TEST_DATA_PATH)
	print("[INFO] load complete.")
	# 划分训练集
	data = data.randomSplit([0.2,0.8])[0]
	(trainingData, testData) = data.randomSplit([0.7, 0.3])

	# Train a RandomForest model.
	#  Empty categoricalFeaturesInfo indicates all features are continuous.
	#  Note: Use larger numTrees in practice.
	#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
	model = RandomForest.trainClassifier(trainingData, numClasses= NUM_OF_CLASSES, categoricalFeaturesInfo={},
										 numTrees=NUM_OF_TREES, featureSubsetStrategy="auto",
										 impurity='gini', maxDepth=MAXDEPTH, maxBins = MAXBINS)

	# Evaluate model on test instances and compute test error
	predictions = model.predict(testData.map(lambda x: x.features))
	labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
	testErr = labelsAndPredictions.filter(
		lambda lp: lp[0] != lp[1]).count() / float(testData.count())
	print('[INFO] Test Error = ' + str(testErr))
	print('[INFO] Learned classification forest model:')
	print(model.toDebugString())

	# Save and load model
	model.save(sc,TEST_MODEL_PATH)
	sameModel = RandomForestModel.load(sc,TEST_MODEL_PATH)

if __name__ == "__main__":
	sc = SparkContext(appName="PyRandomForestLearningTest")
	train()