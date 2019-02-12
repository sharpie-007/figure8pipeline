import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    '''tokenizes text from query using tokenizer built in 
    train_classifer.py'''
    clean_tokens = tok.texts_to_matrix(text, mode='count')
    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/messages.db')
df = pd.read_sql_table('messages_table', engine)
df.iloc[:,4:-1] = df.iloc[:,4:-1].astype(int)

# load model
model = joblib.load("../models/classifier.pickle")

# Load tokenizer
tok = joblib.load("../models/tokenizer.pickle")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_counts = df.iloc[:, 4:-1].sum()
    
    # category_counts.index.values.tolist()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
         {
            'data': [
                Bar(
                    x=category_counts.index.values.tolist(),
                    y=category_counts.values.tolist()
                )
            ],

            'layout': {
                'title': 'Breakdown by Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    print(query)
    text = tokenize([query])
    # use model to predict classification for query
    classification_labels = model.predict(text)
    print(classification_labels.shape)
    print(classification_labels[0])
    
    classification_results = dict(zip(df.columns[4:], classification_labels[0]))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()