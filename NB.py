import codecs
from random import shuffle
import nltk
from pandas import DataFrame
import os
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

txtpath = '/home/sunito/Desktop/spamdata/TEXT/'
respath = '/home/sunito/Desktop/out/'

def getfilenames(dir):
    return [name for name in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, name))]

def readLines(file):
    with codecs.open(file, encoding="latin-1") as readtxt:
        return readtxt.read().splitlines()

def createframes(path, files, label):
    rows = []
    index = []
    #for file in files
    for file in files:
        content = "\n".join(readLines(path + file))
        rows.append({'content': content, 'class': label})
	if file in index:
            print('found duplicate: ' + file)
        else:
            index.append(file)
    data_frame = DataFrame(rows, index=index)
    return data_frame


print('load ham and spam text files')
hamtxtfiles = getfilenames(txtpath + 'ham')
spamtxtfiles = getfilenames(txtpath + 'spam')
print('ham count: ' + str(len(hamtxtfiles)))
print('spam count: ' + str(len(spamtxtfiles)) + '\n')
print('creating dataframes...')
data = DataFrame({'content': [], 'class': []})
data = data.append(createframes(txtpath + 'ham/', hamtxtfiles, 'ham'))
data = data.append(createframes(txtpath + 'spam/', spamtxtfiles, 'spam'))

print('shuffle the data...')
#data = data.reindex(numpy.random.permutation(data.index))
#data.to_csv(respath + 'dataframes')

print('\n')
print('creating pipeline (bag of words) -> (MultinomialNB)')
pipeline_bag_mnb = Pipeline([('vectorizer', CountVectorizer()), ('classifier', MultinomialNB())])

print('creating pipeline (BIGRAMS) -> (MultinomialNB)')
pipeline_bag_bg_mnb = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))), ('classifier', MultinomialNB())])

print('creating pipeline (TF-IDF) -> (MultinomialNB)')
pipeline_tfidf_mnb = Pipeline([('vectorizer', TfidfVectorizer()), ('classifier', MultinomialNB())])

print('creating pipeline (bag of words) -> (BernoulliNB)')
pipeline_bag_bnb = Pipeline([('vectorizer', CountVectorizer()), ('classifier', BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True))])

print('creating pipeline (BIGRAMS) -> (BernoulliNB)')
pipeline_bag_bg_bnb = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))), ('classifier', BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True))])

print('creating pipeline (TF-IDF) -> (BernoulliNB)')
pipeline_tfidf_bnb = Pipeline([('vectorizer', TfidfVectorizer()), ('classifier', BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True))])

print('creating pipeline (BI-GRAMS) -> (TF-IDF) -> (MultinomialNB)')
pipeline_bg_tfidf_mnb = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))), ('tf-idf_transformer', TfidfTransformer()), ('classifier', MultinomialNB())])

print('creating pipeline (BI-GRAMS) -> (TF-IDF) -> (BernoulliNB)')
pipeline_bg_tfidf_bnb = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))), ('tf-idf_transformer', TfidfTransformer()), ('classifier', BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True))])

#pipeline.fit(data['content'].values, data['class'].values)

def crossvalidate(pipeline, method, classifier, folds):
    print('Using ' + method + ' as features and ' + classifier)
    print('Running cross validation with ' + str(folds) + ' folds ....')
    k_fold= KFold(n=len(data), n_folds=folds)
    scores = []
    confusion = numpy.array([[0,0], [0,0]])

    for train_indices, test_indices in k_fold:
        train_text = data.iloc[train_indices]['content'].values
        train_y = data.iloc[train_indices]['class'].values
        
        test_text = data.iloc[test_indices]['content'].values
        test_y = data.iloc[test_indices]['class'].values
	print('train set: ' + str(len(train_y)))
	print('test  set:  ' + str(len(test_y)))
        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        #score = f1_score(test_y, predictions, pos_label='spam')
        score = accuracy_score(test_y, predictions)
        scores.append(score)

    #print('\n')
    #print('Total emails classified: ' + str(len(data)))
    print('Accuracy: '+ str(sum(scores)/len(scores)))
    print('Confusion matrix:')
    print(confusion)
    print('\n')

print('\n')
folds = 6
crossvalidate(pipeline_bag_mnb, 'BAG OF WORDS', 'MultinomialNaiveBayes Classifier', 6)
crossvalidate(pipeline_bag_bg_mnb, 'BI-GRAMS', 'MultinomialNaiveBayes Classifier', 6)
crossvalidate(pipeline_tfidf_mnb, 'TF-IDF', 'MultinomialNaiveBayes Classifier', 6)
crossvalidate(pipeline_bag_mnb, 'BAG OF WORDS', 'BernoulliNaiveBayes Classifier', 6)
crossvalidate(pipeline_bag_bg_mnb, 'BI-GRAMS', 'BernoulliNaiveBayes Classifier', 6)
crossvalidate(pipeline_tfidf_mnb, 'TF-IDF', 'BernoulliNaiveBayes Classifier', 6)
crossvalidate(pipeline_bg_tfidf_mnb, 'BIGRAMS and TF-IDF', 'BernoulliNaiveBayes Classifier', 6)
crossvalidate(pipeline_bg_tfidf_bnb, 'BIGRAMS and TF-IDF', 'BernoulliNaiveBayes Classifier', 6)
