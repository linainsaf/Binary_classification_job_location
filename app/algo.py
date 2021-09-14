# import libraries
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import joblib

# import Data
df = pd.read_csv('../Data/data_cleaned.csv')

# split my data into train and test data
X_train, X_test, y_train, y_test = train_test_split(df['input_data'],
                                                    df['labels'],
                                                    train_size=0.75)

# pipeline
model = make_pipeline(CountVectorizer(stop_words='english',strip_accents='unicode'), TfidfTransformer(),
                      svm.SVC(kernel='linear',C=1, probability=True))

# fit the model
model = model.fit(X_train, y_train)

# save weights
model_filename = "../Models/model.pkl"
joblib.dump((model), model_filename)
