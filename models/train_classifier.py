import sys
import os
from sqlalchemy import create_engine
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import multiprocessing
import pickle


def load_data(database_filepath):
    '''Uses sqlalchemy engine to load the data from sqlite, splits into X and
    Y, drops the child_alone column (due to lack of sample data), then returns
    X, Y, and the column names.'''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('messages_table', engine)
    X = df.iloc[:, 0:3]
    Y = df.iloc[:, 4:-1]
    Y.drop(columns=['child_alone'], inplace=True)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text, vocab_size):
    '''Loads the Keras tokenizer trained in train_classifier.py, tokenizes the
    text, and then returns the tokenized text.'''
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(text)
    encoded_docs = tokenizer.texts_to_matrix(text, mode='count')
    return tokenizer, encoded_docs


def build_model():
    '''Gets the number of available CPU's, threads the classifier accordingly
    builds the classifier pipeline, returns the pipeline'''
    num_cpus = multiprocessing.cpu_count()
    pipeline = Pipeline([('classifier', MultiOutputClassifier(
        RandomForestClassifier(n_estimators=40, n_jobs=num_cpus - 1,
        verbose=1)))])
    return pipeline


def evaluate_model(model, X_test, Y_test):
    '''Predicts the outcomes of Y_test and evaluates against X_test. Prints
    the outputs of the sklearn classification report.'''
    Y_pred = model.predict(X_test)
    for i in range(0, len(Y_test.iloc[0])):
        print(i)
        truth = Y_test.iloc[:,i:i+1]
        prediction = Y_pred[:,i:i+1]
        print(Y_test.columns[i], "\n",
            classification_report(truth, prediction))


def save_model(tok, model, model_filepath):
    '''Saves models to disk'''
    
    tokenizer_filename = "tokenizer.pickle"
    classifier_filename = "classifier.pickle"

    tokenizer_file = open(model_filepath + tokenizer_filename, 'wb')
    pickle.dump(tok, tokenizer_file)

    classifier_file = open(model_filepath + classifier_filename, 'wb')
    pickle.dump(model, classifier_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        # The below can be uncommented for dynamic testing
        # database_filepath = "data\messages.db"
        # model_filepath = ''
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        print('Tokenizing messages...')
        tok, X = tokenize(X['message'], 5000)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('CPU Count {}'.format(multiprocessing.cpu_count()))
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(tok, model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()