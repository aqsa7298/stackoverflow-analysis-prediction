from inspect import TPFLAGS_IS_ABSTRACT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MaxAbsScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize
from nltk.corpus import stopwords
from analysis import read_df

# preprocess data of dataframe
def preprocess_data(df):
    
    #combine and drop unnecessary columns 
    df['Text'] = df['Title'] + ' ' + df['Body']
    df.drop(['Title', 'Body', 'Tags', 'CreationDate'], axis = 1, inplace = True)
    df['Text'] = df['Text'].str.lower()
    
    # clean text by removig punctuations and stopwords
    stop = set(stopwords.words('english'))
    stop.add('p')
    df['Text'] = df['Text'].apply(lambda row: remove_punctuation_stopwords(row, stop))

    # replace te labels with numbers as these numbers will be fed into all models
    mapping = {'HQ':0 , 'LQ_CLOSE':1, 'LQ_EDIT':2}
    df['Y'].replace(mapping, inplace = True)

    # shuffle the DataFrame rows
    df = df.sample(frac = 1)
    # print updated informarion about dataframe 
    df.info()
    
    return df

# apply machine learning models after tfidf
def preprocess_tfidf_models(csv_file_list):

    df = read_df(csv_file_list)
    data = preprocess_data(df)

    #split data into train and test
    train, test = train_test_split(data, test_size=0.25, random_state=0)
    x_train = train['Text']
    y_train = train['Y']
    x_test = test['Text']
    y_test = test['Y']

    # TFIDF 
    #idf = inverse document frequency (how important word is) made it true so it takes importance of word as well along with it frequency
    tfidf_vect = TfidfVectorizer(use_idf=True, max_features=20000)
    x_train_tfidf = tfidf_vect.fit_transform(x_train)
    x_test_tfidf = tfidf_vect.transform(x_test)
    transformer = MaxAbsScaler()
    x_train_maxabs = transformer.fit_transform(x_train_tfidf)
    x_test_maxabs = transformer.transform(x_test_tfidf)

    print('Running Logistic Regression model. Please wait as it will take some time')

    #logistic regression model 
    log_reg = LogisticRegression(random_state=0, max_iter=200)      
    log_reg.fit(x_train_maxabs, y_train)
    y_pred = log_reg.predict(x_test_maxabs) 

    lg_matrix = metrics.confusion_matrix(y_test, y_pred, labels = [0,1,2])
    accuracy_log = metrics.accuracy_score(y_test, y_pred)
    precision_log = metrics.precision_score(y_test, y_pred, average='weighted')
    recall_log = metrics.recall_score(y_test, y_pred, average='weighted')
    title_log = 'Logistic Regression Confusion Matrix using TFIDF'
    fig_name_log = 'log_reg_tfidf.png'
    print('Logistic Regression Results using TFIDF')
    confusion_matrix_heatmap(lg_matrix, accuracy_log, precision_log, recall_log, title_log, fig_name_log)

    print('Running Multinomoial Naive Bayes model. Please wait as it will take some time')

    # multinomial naive bayes classifier
    multinomial_nb = MultinomialNB()
    multinomial_nb.fit(x_train_maxabs, y_train)    
    y_pred_nb = multinomial_nb.predict(x_test_maxabs)

    nb_matrix = metrics.confusion_matrix(y_test, y_pred_nb, labels = [0,1,2])
    accuracy_nb = metrics.accuracy_score(y_test, y_pred_nb)
    precision_nb = metrics.precision_score(y_test, y_pred_nb, average='weighted')
    recall_nb = metrics.recall_score(y_test, y_pred_nb, average='weighted')
    title_nb = 'Naive Bayes Confusion Matrix using TFIDF'
    fig_name_nb = 'naive_bayes_tfidf.png'
    print('Naive Bayes Results using TFIDF')
    confusion_matrix_heatmap(nb_matrix, accuracy_nb, precision_nb, recall_nb, title_nb, fig_name_nb)

    print('Running XGBoost model. Please wait as it will take some time')
    
    # XGBoost classifier
    xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_classifier.fit(x_train_maxabs, y_train)
    y_pred_xgb = xgb_classifier.predict(x_test_maxabs)

    xgb_matrix = metrics.confusion_matrix(y_test, y_pred_xgb, labels = [0,1,2])
    accuracy_xgb = metrics.accuracy_score(y_test, y_pred_xgb)
    precision_xgb = metrics.precision_score(y_test, y_pred_xgb, average='weighted')
    recall_xgb = metrics.recall_score(y_test, y_pred_xgb, average='weighted')
    title_xgb = 'XGBoost Confusion Matrix using TFIDF'
    fig_name_xgb = 'xgboost_tfidf.png'
    print('XGBoost Results using TFIDF')
    confusion_matrix_heatmap(xgb_matrix, accuracy_xgb, precision_xgb, recall_xgb, title_xgb, fig_name_xgb)

    print('Running SVM model. Please wait as It will take some time')

    # Support Vector Machines SVM
    svm_classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_classifier.fit(x_train_maxabs, y_train)
    y_pred_svm = svm_classifier.predict(x_test_maxabs)
    
    svm_matrix = metrics.confusion_matrix(y_test, y_pred_svm, labels = [0,1,2])
    accuracy_svm = metrics.accuracy_score(y_test, y_pred_svm)
    precision_svm = metrics.precision_score(y_test, y_pred_svm, average='weighted')
    recall_svm = metrics.recall_score(y_test, y_pred_svm, average='weighted')
    title_svm = 'Support Vector Machine Confusion Matrix using TFIDF'
    fig_name_svm = 'svm_tfidf.png'
    print('Support Vector Machine Results using TFIDF')
    confusion_matrix_heatmap(svm_matrix, accuracy_svm, precision_svm, recall_svm, title_svm, fig_name_svm)

    print('Running Random Forest. Please wait as it will take some time')

    # Random Forest 
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(x_train_maxabs, y_train)
    y_pred_rf = rf_classifier.predict(x_test_maxabs)

    rf_matrix = metrics.confusion_matrix(y_test, y_pred_rf, labels = [0,1,2])
    accuracy_rf = metrics.accuracy_score(y_test, y_pred_rf)
    precision_rf = metrics.precision_score(y_test, y_pred_rf, average='weighted')
    recall_rf = metrics.recall_score(y_test, y_pred_rf, average='weighted')
    title_rf = 'Random Forest Confusion Matrix using TFIDF'
    fig_name_rf = 'random_forest_tfidf.png'
    print('Random Forest Results using TFIDF')
    confusion_matrix_heatmap(rf_matrix, accuracy_rf, precision_rf, recall_rf, title_rf, fig_name_rf)

    print('Running MLP (Multilayer Perceptron) classifier. Please wait as it will take some time')

    # MLP (Multilayer Perceptron) classifier
    mlp_classifier = MLPClassifier(solver='adam', alpha=0.01, random_state=0, hidden_layer_sizes=(10,))
    mlp_classifier.fit(x_train_maxabs, y_train)
    y_pred_mlp = mlp_classifier.predict(x_test_maxabs)
    
    mlp_matrix = metrics.confusion_matrix(y_test, y_pred_mlp, labels = [0,1,2])
    accuracy_mlp = metrics.accuracy_score(y_test, y_pred_mlp)
    precision_mlp = metrics.precision_score(y_test, y_pred_mlp, average='weighted')
    recall_mlp = metrics.recall_score(y_test, y_pred_mlp, average='weighted')
    title_mlp = 'MLP Confusion Matrix using TFIDF'
    fig_name_mlp = 'MLP_tfidf.png'
    print('MLP Results using TFIDF')
    confusion_matrix_heatmap(mlp_matrix, accuracy_mlp, precision_mlp, recall_mlp, title_mlp, fig_name_mlp)

    
# apply models after converting to doc2vec
def preprocess_doc2vec_models(csv_file_list):

    df = read_df(csv_file_list)
    data = preprocess_data(df)

    #split data into train and test
    train, test = train_test_split(data, test_size=0.25, random_state=0)

    #Create Tagged document for each row of dataframe
    train_tagged = train.apply(
    lambda r: TaggedDocument(words=word_tokenize(r['Text']), tags=[r.Y]), axis=1)
    test_tagged = test.apply(
    lambda r: TaggedDocument(words=word_tokenize(r['Text']), tags=[r.Y]), axis=1)

    print('Tagged Document created. You can view one by typing "print(train_tagged.values[0])"')
    
    # distributed bag of word model (DBOW)
    model = Doc2Vec(vector_size=100, min_count=2, epochs=30)

    #build vocabulary
    model.build_vocab(train_tagged)
    print('Training complete Vocabulary::::::   ')
    print(model.wv.key_to_index['variables'])

    #training doc2vec model
    model.train(train_tagged, total_examples = len(train_tagged), epochs = model.epochs)

    print('Doc2Vec Model Trained')
    
    # generate vector features of both training and testing data
    x_train, y_train = vec_for_learning(model, train_tagged)
    x_test, y_test = vec_for_learning(model, test_tagged)

    print ('Vectors generated for data. You can view one by typing "print(x_train[1])"')
    
    #logistic regression model with default parameters
    log_reg = LogisticRegression(random_state=0, max_iter=200)   
    log_reg.fit(x_train, y_train)
    y_pred = log_reg.predict(x_test) 

    lg_matrix = metrics.confusion_matrix(y_test, y_pred, labels = [0,1,2])
    accuracy_log = metrics.accuracy_score(y_test, y_pred)
    precision_log = metrics.precision_score(y_test, y_pred, average='weighted')
    recall_log = metrics.recall_score(y_test, y_pred, average='weighted')
    title_log = 'Logistic Regression Confusion Matrix using Doc2Vec'
    fig_name_log = 'log_reg_doc2vec.png'
    print('Logistic Regression Results using Doc2Vec')
    confusion_matrix_heatmap(lg_matrix, accuracy_log, precision_log, recall_log, title_log, fig_name_log)

    print('Running SVM model. Please wait as it will take some time')

    # Support Vector Machines SVM
    svm_classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_classifier.fit(x_train, y_train)
    y_pred_svm = svm_classifier.predict(x_test)
    
    svm_matrix = metrics.confusion_matrix(y_test, y_pred_svm, labels = [0,1,2])
    accuracy_svm = metrics.accuracy_score(y_test, y_pred_svm)
    precision_svm = metrics.precision_score(y_test, y_pred_svm, average='weighted')
    recall_svm = metrics.recall_score(y_test, y_pred_svm, average='weighted')
    title_svm = 'Support Vector Machine Confusion Matrix using Doc2Vec'
    fig_name_svm = 'svm_doc2vec.png'
    print('Support Vector Machine Results using Doc2Vec')
    confusion_matrix_heatmap(svm_matrix, accuracy_svm, precision_svm, recall_svm, title_svm, fig_name_svm)

    print('Running Random Forest model. Please wait as it will take some time')

    # Random Forest 
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(x_train, y_train)
    y_pred_rf = rf_classifier.predict(x_test)
    
    rf_matrix = metrics.confusion_matrix(y_test, y_pred_rf, labels = [0,1,2])
    accuracy_rf = metrics.accuracy_score(y_test, y_pred_rf)
    precision_rf = metrics.precision_score(y_test, y_pred_rf, average='weighted')
    recall_rf = metrics.recall_score(y_test, y_pred_rf, average='weighted')
    title_rf = 'Random Forest Confusion Matrix using Doc2Vec'
    fig_name_rf = 'random_forest_doc2vec.png'
    print('Random Forest Results using Doc2Vec')
    confusion_matrix_heatmap(rf_matrix, accuracy_rf, precision_rf, recall_rf, title_rf, fig_name_rf)
    
def vec_for_learning(model, tagged_doc):
    
    tagged = tagged_doc.values
    targets, regressors= zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in tagged])
    return regressors, targets

def remove_punctuation_stopwords(row, st):
    
    row = re.sub(r'[^a-zA-Z]',' ', row)
    row = row.split()
    row = ' '.join([x for x in row if x not in st])
    return row

def confusion_matrix_heatmap(mat, accuracy, precision, recall, title, fig_name):

    print("Accuracy: ",accuracy)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print('Confusion Matrix: \n',mat)

    sns.heatmap(mat/np.sum(mat), annot=True, fmt='.2%', cmap='Blues', xticklabels = ['HQ', 'LQ_CLOSE', 'LQ_EDIT'], yticklabels = ['HQ', 'LQ_CLOSE', 'LQ_EDIT'])
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label \nAccuracy={:0.4f}, Precison={:0.4f}, Recall={:0.4f}'.format(accuracy, precision, recall))
    plt.title(title)
    plt.tight_layout()    
    plt.savefig("./images/"+fig_name)
    plt.show()
    
if __name__ == "__main__": 

    csv_file_list = ['./stackoverflow/train.csv', './stackoverflow/valid.csv']
    preprocess_tfidf_models(csv_file_list)
    preprocess_doc2vec_models(csv_file_list)
    