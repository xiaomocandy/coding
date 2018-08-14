from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkContext
sc = SparkContext.getOrCreate()

data = [
    LabeledPoint(0.0, [0.0]),
    LabeledPoint(1.0, [1.0]),
    LabeledPoint(1.0, [2.0]),
    LabeledPoint(1.0, [3.0])
]
model = DecisionTree.trainClassifier(sc.parallelize(data), 2, {})
print(model)