import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
from sklearn.model_selection import train_test_split
#from keras.models import Sequential
#from keras.layers import Dense
#from sklearn.model_selection import cross_val_score
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import  mutual_info_regression



def load_data():
    # Read in retinaopatyh dataset
#    df = pd.read_csv('/Users/majed/Desktop/Retinopathy/dataset.csv', header=0,
#                     names=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14',
#                            'f15', 'f16', 'f17', 'f18', 'f19', 'class'])
    
    # this dataset after doing the features reduction procedure
    df = pd.read_csv('/Users/majed/Desktop/Retinopathy/dataset3.csv', header=0,
                     names=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'class'])
    
    # Specify the data
    X = df.ix[:, 0:14]
    print(X.shape)

    # Specify the target labels
    y = df.ix[:, 14]

    normal = df.loc[df['class'] == 0]
    abnormal = df.loc[df['class'] == 1]

    #Standardization/Normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)


    # Split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return normal, abnormal, df, X, y, X_train, X_test, y_train, y_test;

def exploreData():

    # Now it's the time to know more about the data.
    print(df.describe())
    print(df.head(10))

    #Here we need to know the number of rows labeled as (1=positive="abnormal") and (0=negative="normal")
    df['class'].value_counts()
    counts = df['class'].value_counts()
    print(counts)

    # The next graph presents the number of samples that labeled either positive or negative
    labels = 'normal', 'abnormal'
    sizes = [counts[0], counts[1]]
    colors = ['yellowgreen', 'lightcoral']
    explode = (0, 0.05)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.show()

def kde_algo():

    # kernel density estimation (KDE)
    # It's good to see the univariate distribution of the data by applying KDS function for each feature in order to
    # figure out the distribution.

    # ploting all features
    f, ax = plt.subplots(figsize=(60, 10))

    #ax.set_aspect("equal")
    sns.set(style="darkgrid")

    col = df.columns[:19]
    for x in range(19):
         ax = plt.subplot(19, 3, x + 1)
         ax.set_title(col[x])
         sns.kdeplot(normal[col[x]], label='normal', ax=ax)
         sns.kdeplot(abnormal[col[x]], label='abnormal', ax=ax)
    plt.show()


def create_model():
    # create your model using this function
    # Multilayer Perceptron (MLP) - Binary Classification
    model = Sequential()
    model.add(Dense(12, input_dim=14, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model;


def train_and_evaluate__model(model, X_train, y_train, X_test, y_test):

    # fit and evaluate here.
    model.fit(X_train , y_train, epochs=150, verbose=1, batch_size=10)
    # evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    

def predit(model, X_test):

    # Prediction
    model.predict(X_test)
    predictions = model.predict_classes(X_test)
    rounded = [round(x[0]) for x in predictions]
    print(rounded)
    print("Score:", model.scores(X_test, y_test))
    

def keras_CV():
    
    # cross vaidation with Keras
    seed = 7
    numpy.random.seed(seed)
    model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=4, verbose=1)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(model, X, y, cv=kfold)
    print(results)
    print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std() * 2)) 

    
    
    
def svm_model():
        # SVM
    clf = svm.SVC(kernel='linear', C=3)
    scores = cross_val_score(clf, X, y, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 
    
def ensumble_model():
    
    # ensumble RandomForestClassifier

    seed = 7
    numpy.random.seed(seed)
    X, y = make_classification(n_samples=1151, n_features=14, n_informative=2, n_redundant=6,random_state=0,
                               shuffle=False)
    clf = RandomForestClassifier(max_depth=12, random_state=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=kfold)
    print(scores)
    print(scores.mean())
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 

    
def neural_model():
    
    #Nueral Network
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-8,hidden_layer_sizes=(100, 20), random_state=1)
    scores = cross_val_score(clf, X, y, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 
    

def svc_model():
    
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, X, y, cv=10)                           
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 
    
def features_ranking():
    
    mi = mutual_info_regression(X, y)
    mi /= np.max(mi)
    print(mi)
    
    
if __name__ == "__main__":

    #loading the data
    normal, abnormal, df, X, y, X_train, X_test, y_train, y_test = load_data()

    #exploreData()

    #kde_algo()
    
    #svc_model()
    
    ensumble_model()

    # keras_CV()
    
    #svm_model()
    
    #neural_model()

    #features_ranking()

    #predit(model, X_test)
    


