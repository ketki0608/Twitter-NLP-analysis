'''
  This program shell reads tweet data for the twitter sentiment classification problem.
  The input to the program is the path to the Semeval directory "corpus" and a limit number.
  The program reads the first limit number of tweets
  It creates a "tweetdocs" variable with a list of tweets consisting of a pair
    with the list of tokenized words from the tweet and the label pos, neg or neu.

  Usage:  python classifySemevalTweets.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
# while the semeval tweet task b data has tags for "positive", "negative", 
#  "objective", "neutral", "objective-OR-neutral", we will combine the last 3 into "neutral"
import os
import sys
import nltk
import re
import random
import subjectivity
from nltk.tokenize import TweetTokenizer
import opinion
import sklearn
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# function to read tweet training file, train and test a classifier 
def processtweets(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  # initialize NLTK built-in tweet tokenizer
  twtokenizer = TweetTokenizer()
  
  os.chdir(dirPath)
  
  f = open('./downloaded-tweeti-b-dist.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  #    assuming that the tweets are sufficiently randomized
  tweetdata = []
  for line in f:
    if (len(tweetdata) < limit):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the tweet and user ids, and keep the sentiment and tweet text
      tweetdata.append(line.split('\t')[2:4])
  
  # create list of tweet documents as (list of words, label)
  # where the labels are condensed to just 3:  'pos', 'neg', 'neu'
  tweetdocs = []
  # add all the tweets except the ones whose text is Not Available
  for tweet in tweetdata:
    if (tweet[1] != 'Not Available'):
      processedtweet = _processtweets(tweet[1])
      # run the tweet tokenizer on the text string - returns unicode tokens, so convert to utf8
      tokens = twtokenizer.tokenize(processedtweet)

      if tweet[0] == '"positive"':
        label = 'pos'
      else:
        if tweet[0] == '"negative"':
          label = 'neg'
        else:
          if (tweet[0] == '"neutral"') or (tweet[0] == '"objective"') or (tweet[0] == '"objective-OR-neutral"'):
            label = 'neu'
          else:
            label = ''
      tweetdocs.append((tokens, label))

  ## shuffle dataset and generate feature set with most common 2000 words
  random.shuffle(tweetdocs)
  all_words_list = [word for (sent,cat) in tweetdocs for word in sent]
  all_words = nltk.FreqDist(all_words_list)
  word_items = all_words.most_common(2000)
  word_features = [word for (word,count) in word_items]
  
  ## Base Accuracy with all the words
  featuresets = getfeatureset(tweetdocs, word_features)
  setsize = int(limit/5)
  train_set, test_set = featuresets[setsize:], featuresets[:setsize]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print("Base accuracy with all the words: " + str(nltk.classify.accuracy(classifier, test_set)))
  print("****** Most informative 20 features *********")
  classifier.show_most_informative_features(20)
  print("**********************************************")

  ## Cross validation with k fold
  num_folds = 5
  print("Cross validation with 5 folds: ")
  cross_validation_accuracy(num_folds, featuresets)
  cross_validation_metrics(num_folds, featuresets)
  
  ## Stop words removal
  dirname = os.path.realpath('..')
  stopwordspath = dirname + '/stopwords_twitter.txt'
  stopwords = [line.strip() for line in open(stopwordspath)]
  negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
  negationwords.extend(['ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])
  newstopwords = [word for word in stopwords if word not in negationwords]
  new_all_words_list = [word for (sent,cat) in tweetdocs for word in sent if word not in newstopwords]
  new_all_words = nltk.FreqDist(new_all_words_list)
  new_word_items = new_all_words.most_common(2000)
  new_word_features = [word for (word,count) in new_word_items]

  ## Accuracy with stop word removal
  new_featuresets = getfeatureset(tweetdocs, new_word_features)
  setsize = int(limit/5)
  train_set, test_set = new_featuresets[setsize:], new_featuresets[:setsize]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print("Accuracy with stop words removal: " + str(nltk.classify.accuracy(classifier, test_set)))

  ## Subjectivity lexicon
  SLfeaturesets = getSLfeatureset(tweetdocs, word_features)
  setsize = int(limit/5)
  train_set, test_set = SLfeaturesets[setsize:], SLfeaturesets[:setsize]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print("Accuracy with Subjectivity lexicon: " + str(nltk.classify.accuracy(classifier, test_set)))

  ## Negation words
  notfeaturesets = getNOTfeatureset(tweetdocs, word_features, negationwords)
  setsize = int(limit/5)
  train_set, test_set = notfeaturesets[setsize:], notfeaturesets[:setsize]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print("Accuracy with negation words features: " + str(nltk.classify.accuracy(classifier, test_set)))

  ## Negation words and stop words removal
  newnotfeaturesets = getNOTfeatureset(tweetdocs, new_word_features, negationwords)
  setsize = int(limit/5)
  train_set, test_set = newnotfeaturesets[setsize:], newnotfeaturesets[:setsize]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print("Accuracy with negation words features and stop words removal: " + str(nltk.classify.accuracy(classifier, test_set)))

  ## Opinion lexicon
  opinionlexfeaturesets = getopinionfeatureset(tweetdocs, word_features)
  train_set, test_set = opinionlexfeaturesets[1000:], opinionlexfeaturesets[:1000]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print("Accuracy with Opinion lexicon: " + str(nltk.classify.accuracy(classifier, test_set)))

  ## Advanced experiments
  print("*************** Advanced experiments **************")

  ## sklearn SGDClassifier
  skfetaureset = [tweet_features(d, word_features) for (d, c) in tweetdocs]
  X = pd.DataFrame.from_dict(skfetaureset)
  y = [c for (d, c) in tweetdocs]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  clf = SGDClassifier()
  clf.fit(X_train, y_train)
  predicted = clf.predict(X_test)
  print("Accuracy with SGDClassifier - default parameters: " + str(np.mean(predicted == y_test)))

  text_clf_svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=6, random_state=42)
  text_clf_svm.fit(X_train, y_train)
  predicted_svm = text_clf_svm.predict(X_test)
  print("Accuracy with SGDClassifier: " + str(np.mean(predicted_svm == y_test)))


  ## Different classifiers from sklearn - ref https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
  skfetaureset = [tweet_features(d, word_features) for (d, c) in tweetdocs]
  X = pd.DataFrame.from_dict(skfetaureset)
  y = [c for (d, c) in tweetdocs]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  h = .02  # step size in the mesh
  names = ["Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest", "Naive Bayes"]
  classifiers = [
               KNeighborsClassifier(3),
               SVC(kernel="linear", C=0.025),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               GaussianNB()]
  for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    predicted = clf.predict(X_test)
    print(name+" : "+ str(score))

  

  ## Different classifiers from sklearn with stop words removal - ref https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
  sknewfetaureset = [tweet_features(d, new_word_features) for (d, c) in tweetdocs]
  print("Classifiers with stop word removal")
  X = pd.DataFrame.from_dict(sknewfetaureset)
  y = [c for (d, c) in tweetdocs]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  h = .02  # step size in the mesh
  names = ["Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest", "Naive Bayes"]
  classifiers = [
               KNeighborsClassifier(3),
               SVC(kernel="linear", C=0.025),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               GaussianNB()]
  for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    predicted = clf.predict(X_test)
    print(name+" : "+ str(score))


#####################################################################

def _processtweets(tweet):
  tweet = tweet.lower()
  tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet) # remove URLs
  tweet = re.sub('@[^\s]+', '', tweet) # remove usernames
  tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  #remove # in hashtags
  return tweet

def tweet_features(tweet, word_features):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in tweet_words)
    return features

## No stop words removal
def getfeatureset(tweetdocs, word_features):
  return [(tweet_features(d, word_features), c) for (d, c) in tweetdocs]


def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

def cross_validation_metrics(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    recall_list = []
    precision_list = []
    F1_list = []
    labels = []
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        gold = []
        predicted = []
        for (features, label) in test_this_round:
            gold.append(label)
            predicted.append(classifier.classify(features))
            labels = list(set(gold))
            # these lists have values for each label
        recall_l = []
        precision_l = []
        F1_l = []
        for lab in labels:
            # for each label, compare gold and predicted lists and compute values
            TP = FP = FN = TN = 0
            for i, val in enumerate(gold):
                if val == lab and predicted[i] == lab:  TP += 1
                if val == lab and predicted[i] != lab:  FN += 1
                if val != lab and predicted[i] == lab:  FP += 1
                if val != lab and predicted[i] != lab:  TN += 1
            # use these to compute recall, precision, F1
            recall = TP / (TP + FP)
            precision = TP / (TP + FN)
            recall_l.append(recall)
            precision_l.append(precision)
            F1_l.append( 2 * (recall * precision) / (recall + precision))
        recall_list.append(recall_l)
        precision_list.append(precision_l)
        F1_list.append(F1_l)
    recall_flist = list(map(mean, zip(*recall_list)))
    precision_flist = list(map(mean, zip(*precision_list)))
    F1_flist = list(map(mean, zip(*F1_list)))
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_flist[i]), \
              "{:10.3f}".format(recall_flist[i]), "{:10.3f}".format(F1_flist[i]))

def mean(a):
    return sum(a) / len(a)


def SL_features(tweet, word_features, SL):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in tweet_words)
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in tweet_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)
    return features

def Opinion_features(tweet, word_features, opinionlexicon):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in tweet_words)
    pos = 0
    neg = 0
    for word in tweet_words:
        if word in opinionlexicon:
            polarity = opinionlexicon[word]
            if polarity == 'positive':
                pos += 1
            if polarity == 'negative':
                neg += 1
            features['positivecount'] = pos
            features['negativecount'] = neg
    return features



def getopinionfeatureset(tweetdocs, word_features):
  dirname = os.path.realpath('..')
  path = dirname + '/opinion-lexicon-English'
  #path = "/Users/ketki.potdar/Syracuse/IST664/FinalProjectData/SemEval2014TweetData/opinion-lexicon-English"
  opinionlexicon = opinion.readOpinion(path)
  return [(Opinion_features(d, word_features, opinionlexicon), c) for (d, c) in tweetdocs]


def getSLfeatureset(tweetdocs, word_features):
  dirname = os.path.realpath('..')
  SLpath = dirname + '/subjclueslen1-HLTEMNLP05.tff'
  ##SLpath = "/Users/ketki.potdar/Syracuse/IST664/FinalProjectData/SemEval2014TweetData/subjclueslen1-HLTEMNLP05.tff"
  SL = subjectivity.readSubjectivity(SLpath)
  return [(SL_features(d, word_features, SL), c) for (d, c) in tweetdocs]


def NOT_features(tweet, word_features, negationwords):
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(tweet)):
        word = tweet[i]
        if ((i + 1) < len(tweet)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(tweet[i])] = (tweet[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features


def getNOTfeatureset(tweetdocs, word_features, negationwords):
  return [(NOT_features(d, word_features, negationwords), c) for (d, c) in tweetdocs]

"""
commandline interface takes a directory name with semeval task b training subdirectory 
       for downloaded-tweeti-b-dist.tsv
   and a limit to the number of tweets to use
It then processes the files and trains a tweet sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifytweets.py <corpus-dir> <limit>')
        sys.exit(0)
    processtweets(sys.argv[1], sys.argv[2])
