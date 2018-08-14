from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
import numpy as np
from pyspark.mllib.classification import LogisticRegressionWithSGD
from time import time

data_file = './mt/kddcup.data_10_percent.gz'
raw_data = sc.textFile(data_file)
# print "Train data size is {}".format(raw_data.count())

test_data_file = "./mt/corrected.gz"
test_raw_data = sc.textFile(test_data_file)
# print "Test data size is {}".format(test_raw_data.count())
# print raw_data.take(100)

# ------------- LR with SGD ------------------
# def parse_interaction(line):
#     line_split = line.split(",")
#     # leave_out = [1,2,3,41]
#     clean_line_split = line_split[0:1]+line_split[4:41]
#     attack = 1.0
#     if line_split[41]=='normal.':
#         attack = 0.0
#     return LabeledPoint(attack, np.array([float(x) for x in clean_line_split]))
#
# training_data = raw_data.map(parse_interaction)
# test_data = test_raw_data.map(parse_interaction)
#
# t0 = time()
# logit_model = LogisticRegressionWithSGD.train(training_data)
# tt = time() - t0
#
# print "------------------ Classifier trained in {} seconds".format(round(tt,3))
#
# labels_and_preds = test_data.map(lambda p: (p.label, logit_model.predict(p.features)))
#
# t0 = time()
# test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data.count())
# tt = time() - t0
#
# print "------------------ Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))


# ------------- Model Selection by hypothesis testing ------------------
feature_names = ["land","wrong_fragment",
             "urgent","hot","num_failed_logins","logged_in","num_compromised",
             "root_shell","su_attempted","num_root","num_file_creations",
             "num_shells","num_access_files","num_outbound_cmds",
             "is_hot_login","is_guest_login","count","srv_count","serror_rate",
             "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
             "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
             "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
             "dst_host_rerror_rate","dst_host_srv_rerror_rate"]


def parse_interaction_categorical(line):
    line_split = line.split(",")
    clean_line_split = line_split[6:41]
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
    return LabeledPoint(attack, np.array([float(x) for x in clean_line_split]))

training_data_categorical = raw_data.map(parse_interaction_categorical)


from pyspark.mllib.stat import Statistics
chi = Statistics.chiSqTest(training_data_categorical)

import pandas as pd
pd.set_option('display.max_colwidth', 30)
records = [(result.statistic, result.pValue) for result in chi]
chi_df = pd.DataFrame(data=records, index=feature_names, columns=["Statistic","p-value"])
print(chi_df)

def parse_interaction_chi(line):
    line_split = line.split(",")
    # leave_out = [1,2,3,19,20.41]
    clean_line_split = line_split[0:1] + line_split[4:19] + line_split[21:41]
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
    return LabeledPoint(attack, np.array([float(x) for x in clean_line_split]))


training_data_chi = raw_data.map(parse_interaction_chi)
test_data_chi = test_raw_data.map(parse_interaction_chi)

t0 = time()
logit_model_chi = LogisticRegressionWithSGD.train(training_data_chi)
tt = time() - t0

print "Classifier trained in {} seconds".format(round(tt,3))


labels_and_preds = test_data_chi.map(lambda p: (p.label, logit_model_chi.predict(p.features)))
t0 = time()
test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data_chi.count())
tt = time() - t0

print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))

# ------------- RDD basics ------------------

# normal_raw_data = raw_data.filter(lambda x: 'normal.' in x)
# t0 = time()
# normal_count = normal_raw_data.count()
# tt = time()-t0
# print"There are {} 'normal' interactions".format(normal_count)
# print"---------------Count completed in {} seconds".format(round(tt,3))


# ------------- SVM with SGD ------------------
# from pyspark.mllib.classification import SVMWithSGD
# def parse_interaction(line):
#     line_split = line.split(",")
#     # leave_out = [1,2,3,41]
#     clean_line_split = line_split[0:1]+line_split[4:41]
#     attack = 1.0
#     if line_split[41]=='normal.':
#         attack = 0.0
#     return LabeledPoint(attack, np.array([float(x) for x in clean_line_split]))

# training_data = raw_data.map(parse_interaction)
# test_data = test_raw_data.map(parse_interaction)

# t0 = time()
# logit_model = SVMWithSGD.train(training_data)
# tt = time() - t0

# print "--------------------- Classifier trained in {} seconds".format(round(tt,3))

# labels_and_preds = test_data.map(lambda p: (p.label, logit_model.predict(p.features)))

# t0 = time()
# test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data.count())
# tt = time() - t0

# print "-------------------- Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))

