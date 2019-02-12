# Disaster Response Pipeline Project

The purpose of this project is to provide classification support to disaster response workers in order to help them triage the large volumes of data coming in after a major event. The project takes data from 2 different extracts (disaster_categories.csv and disaster_messages.csv), merges them, cleans the resulting data and writes it out in sql format. An important notes is that I used Keras' tokenizer function for this project as it's incredibly quick, and handles most common text transformation problems natively (punctuation, casing, etc).
A Random Forest Classifier is then trained on the data, evaluated with classification_report, then improved with GridSearchCV. The resulting classifier is stored along with the Keras tokenizer for use in a demonstrator website.
Once the cleaning and training scripts have been run, a demo Flask app is available to enable a user to interact with the Classifier.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        This will output to data/messages.db
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/messages.db ./`
        this will output to models/classifier.pickle and models/tokenizer.pickle.

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
