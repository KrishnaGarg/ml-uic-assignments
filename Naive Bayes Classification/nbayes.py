import matplotlib.pyplot as plt
import re
import random
import math
import numpy as np

random.seed(10)
"""
Read text data from file and pre-process text by doing the following
1. convert to lowercase
2. convert tabs to spaces
3. remove "non-word" characters
Store resulting "words" into an array
"""
FILENAME='SMSSpamCollection'
all_data = open(FILENAME).readlines()

# split into train and test
num_samples = len(all_data)
all_idx = list(range(num_samples))
random.shuffle(all_idx)
idx_limit = int(0.8*num_samples)
train_idx = all_idx[:idx_limit]
test_idx = all_idx[idx_limit:]
train_examples = [all_data[ii] for ii in train_idx]
test_examples = [all_data[ii] for ii in test_idx]

# Preprocess train and test examples
train_words = []
train_labels = []
test_words = []
test_labels = []
train_hams = 0
train_spams = 1

# train examples
for line in train_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige returne
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    if label == 'spam':
        label = 1
        train_spams+=1
    else:
        label = 0
        train_hams+=1

    line_words = line_words[1:]

    train_words.append(line_words)
    train_labels.append(label)

# train-set
spam_words = []
ham_words = []
alpha = 0.1
for ii in range(len(train_words)):  # we pass through words in each (train) SMS
    words = train_words[ii]
    label = train_labels[ii]
    if label == 1:
        spam_words += words
    else:
        ham_words += words
input_words = spam_words + ham_words  # all words in the input vocabulary

num_spam = len(spam_words)
num_ham = len(ham_words)
prob_class_spam = train_spams/ (train_hams + train_spams)
prob_class_ham = train_hams/ (train_hams + train_spams)

def calculate_ham_spam_counts(alpha):
    # Count spam and ham occurances for each word
    spam_counts = {}; ham_counts = {}
    # Spamcounts
    for word in spam_words:
        try:
            word_spam_count = spam_counts.get(word)
            spam_counts[word] = word_spam_count + 1
        except:
            spam_counts[word] = 1 + alpha  # smoothening

    for word in ham_words:
        try:
            word_ham_count = ham_counts.get(word)
            ham_counts[word] = word_ham_count + 1
        except:
            ham_counts[word] = 1 + alpha  # smoothening

    return spam_counts, ham_counts

def train(alpha):
    spam_counts, ham_counts = calculate_ham_spam_counts(alpha)

    tp = tn = fp = fn = 0
    # test examples
    for line in train_examples:
        line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige return
        line = line.lower()  # lowercase
        line = line.replace("\t", ' ')  # convert tabs to spae
        line_words = re.findall(r'\w+', line)
        line_words = [xx for xx in line_words if xx != '']  # remove empty words

        label = line_words[0]
        label = 1 if label == 'spam' else 0

        line_words = line_words[1:]

        prob_xi_given_spam = 1
        prob_xi_given_ham = 1

        for word in line_words:
            if spam_counts.get(word) != None:
                temp = (spam_counts[word])/ (num_spam + alpha * 20000)
            else:
                temp = alpha / (num_spam + alpha * 20000)
        
            if ham_counts.get(word) != None:
                temp2 = ham_counts[word]/ (num_ham + alpha * 20000)
            else:
                temp2 = alpha / (num_ham + alpha * 20000)
            
            prob_xi_given_spam *= temp
            prob_xi_given_ham *= temp2

        # normalize
        prob_spam_given_linewords = (prob_xi_given_spam*prob_class_spam) 
        prob_ham_given_linewords = (prob_xi_given_ham*prob_class_ham) 

        # label is 'ground truth' & 'prob_spam...' is 'predicted label'
        if prob_spam_given_linewords > prob_ham_given_linewords:
            if label == 1: 
                tp+=1 
            else:
                fp+=1
        else:
            if label == 1:
                fn+=1
            else:
                tn+=1

    accuracy = (tp + tn) / (tp + tn + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2*precision*recall / (precision + recall)

    # print("testing accuracy", str(accuracy))
    # print("precision", str(precision))
    # print("recall", str(recall))
    # print("fscore", str(fscore))
    return accuracy

def test(alpha):
    spam_counts, ham_counts = calculate_ham_spam_counts(alpha)

    tp = tn = fp = fn = 0
    # test examples
    for line in test_examples:
        line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige return
        line = line.lower()  # lowercase
        line = line.replace("\t", ' ')  # convert tabs to spae
        line_words = re.findall(r'\w+', line)
        line_words = [xx for xx in line_words if xx != '']  # remove empty words

        label = line_words[0]
        label = 1 if label == 'spam' else 0

        line_words = line_words[1:]

        prob_xi_given_spam = 1
        prob_xi_given_ham = 1

        for word in line_words:
            if spam_counts.get(word) != None:
                temp = (spam_counts[word])/ (num_spam + alpha * 20000)
            else:
                temp = alpha / (num_spam + alpha * 20000)
        
            if ham_counts.get(word) != None:
                temp2 = ham_counts[word]/ (num_ham + alpha * 20000)
            else:
                temp2 = alpha / (num_ham + alpha * 20000)
            
            prob_xi_given_spam *= temp
            prob_xi_given_ham *= temp2

        # normalize
        prob_spam_given_linewords = (prob_xi_given_spam*prob_class_spam) 
        prob_ham_given_linewords = (prob_xi_given_ham*prob_class_ham) 

        # label is 'ground truth' & 'prob_spam...' is 'predicted label'
        if prob_spam_given_linewords > prob_ham_given_linewords:
            if label == 1: 
                tp+=1 
            else:
                fp+=1
        else:
            if label == 1:
                fn+=1
            else:
                tn+=1

    accuracy = (tp + tn) / (tp + tn + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2*precision*recall / (precision + recall)

    if alpha == 0.1:
        print("tp", tp)
        print("fp", fp)
        print("fn", fn)
        print("tn", tn)
        print("testing accuracy", str(accuracy))
        print("precision", str(precision))
        print("recall", str(recall))
        print("fscore", str(fscore))
    return accuracy, precision, recall, fscore

print("********************** Results for part a [alpha = 0.1] **********************")
accuracy, precision, recall, fscore = test(0.1)

print("********************** Plotting graphs for part b **********************")
accuracyM = [0] * 6 ; precisionM = [0] * 6; recallM = [0] * 6; fscoreM = [0] * 6
train_accuracyM = [0] * 6

for i in range (-5, 1):
    accuracyM[i+5], precisionM[i+5], recallM[i+5], fscoreM[i+5] = test(math.pow(2, i))

for i in range (-5, 1):
    train_accuracyM[i+5] = train(math.pow(2, i))

# plot accuracy vs alpha
x = [-5, -4, -3, -2, -1, 0]
plt.plot(x, accuracyM, 'r', label='test-accuracy')
plt.plot(x, train_accuracyM, 'g', label='train-accuracy')
plt.xlabel('alpha') 
plt.ylabel('accuracy')
plt.title('Accuracy vs alpha')
plt.gca().legend(('test-accuracy', 'train-accuracy'))
plt.show()

# plot fscore vs alpha
plt.plot(x, fscoreM)
plt.xlabel('alpha') 
plt.ylabel('fscore') 
plt.title('Fscore vs alpha')
plt.show()