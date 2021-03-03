import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
from sklearn.model_selection import train_test_split
import matplotlib

from keras.models import Sequential
from keras.layers import Dense
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import train_test_split


def load_data():
    # Read in retinaopatyh dataset
    df = pd.read_csv('file:///C:/Users/Dr. M Alauthman/Dropbox/my_research/2020/majed_research/Retinopathy/update_7_2018/messidor1.csv', header=0,
                     names=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14',
                            'f15', 'f16', 'f17', 'f18', 'f19', 'class'])

    # Specify the data
    X = df.ix[:, 0:19]

    # Specify the target labels
    y = df.ix[:, 19]

    normal = df.loc[df['class'] == 0]
    abnormal = df.loc[df['class'] == 1]

    #Standardization/Normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    # Split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

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
    model.add(Dense(12, input_dim=19, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model;


def train_and_evaluate__model(model, X_train, y_train, X_test, y_test):

    # fit and evaluate here.
    model.fit(X_train , y_train, epochs=50, verbose=2, batch_size=32)
    # evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def predit(model, X_test):

    # Prediction
    model.predict(X_test)
    predictions = model.predict_classes(X_test)
    rounded = [round(x[0]) for x in predictions]
    print(rounded)


if __name__ == "__main__":

    #loading the data
    normal, abnormal, df, X, y, X_train, X_test, y_train, y_test = load_data()

    #exolring tge dataset
    #exploreData()

    # kernel density estimation (KDE)
    #kde_algo()

    # cross vaidation
    
    model = None # Clearing the NN.
    model = create_model()
    scores = cross_val_score(model, X, y, cv=10)
    
