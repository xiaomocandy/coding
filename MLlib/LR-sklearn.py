import numpy as np
import gzip
import pandas as pd

def parse_interaction(line):
    line_split = line.split(",")
    # leave_out = [1,2,3,41]
    clean_line_split = line_split[0:1] + line_split[4:41]
    attack = 1.0
    if line_split[41] == 'normal.':
        attack = 0.0
    x = np.array([float(x) for x in clean_line_split])
    x = np.append(x, attack)
    return x


train_data = np.empty(39)

with gzip.open('/root/mt/kddcup.data.gz', 'rb') as f:
    training_data = map(parse_interaction,f.readlines())
train_data = pd.DataFrame(train_data)
train_data.to_csv('/root/mt/train_data.csv')

# test_data_file = "./mt/corrected.gz"
# test_raw_data = sc.textFile(test_data_file)
# print "Test data size is {}".format(test_raw_data.count())
# print raw_data.take(100)

#
# training_data = file_content.map(parse_interaction)
# test_data = test_raw_data.map(parse_interaction)

# t0 = time()
# logit_model = LogisticRegressionWithSGD.train(training_data)
# tt = time() - t0

# print "Classifier trained in {} seconds".format(round(tt,3))

# labels_and_preds = test_data.map(lambda p: (p.label, logit_model.predict(p.features)))
#
# t0 = time()
# test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data.count())
# tt = time() - t0
#
# print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))