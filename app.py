# from flask import Flask, render_template, request, url_for, Markup, jsonify
# import pickle
# import pandas as pd
# import numpy as np
# import pandas as pd
# import numpy as np
# import sys
# import os
# import glob
# import re
# import numpy as np
# import tensorflow as tf
# import tensorflow as tf
# import clean_data.preprocessing
# from eli5.lime import TextExplainer
# from eli5.lime.samplers import MaskingTextSampler
# from sklearn.feature_extraction.text import TfidfVectorizer

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# # Keras
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import pandas as pd
# import pickle
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# import keras.models
# from keras.models import model_from_json

# # Flask utils
# from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence
# from keras.models import Model, Input, Sequential, load_model
# import pickle
# import h5py

# # create Flask application
# app = Flask(__name__)

# # Used in pickle pipeline on TF-IDF
# def dummy(token):
#     return token

# # read object TfidfVectorizer and model from disk
# MODEL_PATH ='DL.h5'
# model = load_model(MODEL_PATH)
 
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# model2 = pickle.load(open('model.pkl', 'rb'))

# df = pd.read_csv("Tweets.csv")
# tweet_df = df[['text','airline_sentiment']]
# tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
# tweet_df["airline_sentiment"].value_counts()
# sentiment_label = tweet_df.airline_sentiment.factorize()



# reading = clean_data.preprocessing.preprocessing(convert_lower=True, use_spell_corrector=True, only_verbs_nouns=False)

# # clf: define ML classifier
# # vec: define vectorizer
# # n_samples: sets the number of random examples to generate from given instance of text (default value 5000)
# # use LIME method to train a white box classifier to make the same prediction as the black box one (pipeline)
# te = TextExplainer(vec=TfidfVectorizer(ngram_range=(1, 2), preprocessor=dummy, token_pattern='(?u)\\b\\w+\\b'),
#                    n_samples=5000, char_based=False, random_state=42)


# def one_word_get_prediction_class_name(prediction):
#     '''
#     Pipeline with XGBoost - translate the prediction class number into words
#     :param prediction: the predicted number/class
#     :return: the predicted class in natural language
#     '''
#     # The order of classes in predict_proba: ['hate speech', 'neither', 'offensive language']
#     if prediction == 0:
#         output = "hate speech"
#     elif prediction == 1:
#         output = "neither"
#     else:
#         output = "offensive language"

#     return output


# def predict_prob(text):
#     '''
#     MUST function that returns predicted probas of pipeline model, because text MUST BE TOKENIZED + CLEANED from empty strings
#     :param text: all 5000 random generated instances from the initial given text
#     :return: predicted probas for each data instance from pickled pipeline model
#     '''
#     text = [sentence.split() for sentence in text]  # TOKENIZE TEXT

#     prob = model.predict_proba(text)

#     return prob












# @app.route('/')
# @app.route('/first') 
# def first():
# 	return render_template('first.html')
# @app.route('/login') 
# def login():
# 	return render_template('login.html')    
    
 
 
# @app.route('/upload') 
# def upload():
# 	return render_template('upload.html') 
# @app.route('/preview',methods=["POST"])
# def preview():
#     if request.method == 'POST':
#         dataset = request.files['datasetfile']
#         df = pd.read_csv(dataset,encoding = 'unicode_escape')
#         return render_template("preview.html",df_view = df)    

 
# @app.route('/home')
# def home():
#     return render_template('index.html')


# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     error = None
# #     if request.method == 'POST':
# #         # message
# #         msg = request.form['message']
# #         msg = pd.DataFrame(index=[0], data=msg, columns=['data'])

# #         # transform data
# #         """new_text = sequence.pad_sequences((tokenizer.texts_to_sequences(msg['data'].astype('U'))), maxlen=547)
          
# #         # model
# #         result = model.predict(new_text,batch_size=1,verbose=2)"""
# #         tw = tokenizer.texts_to_sequences(msg['data'].astype('U'))
# #         tw = sequence.pad_sequences(tw,maxlen=547)
# #         result = int(model.predict(tw).round().item())
# #         print(result)
# #         results = sentiment_label[1][result]
# #         print("Predicted label: ", sentiment_label[1][result], "Speech")
# #         #print(result)


# #         return render_template('index.html', prediction_value=results)
# #     else:
# #         error = "Invalid message"
# #         return render_template('index.html', error=error)
# @app.route('/predict', methods=['POST'])
# def predict():
#     form_text = [request.form.get('hate_speech_text_field')]
#     # str(bytes_string, 'utf-8') # convert byte-string variable into a regular string

#     if len(form_text): # page not reloaded, form_text array not empty [fix the bug of page reloading -  it return no values from forms]
#         # preprocessing of the sentence about to predict
#         final_features = reading.clean_text(form_text[0])

#         print(final_features)

#         # ==============================================================================================================
#         # Explain prediction
#         # ==============================================================================================================

#         # The paths of produced wordcloud for each individual class of the dataset
#         wordcloud_descr = [('Hate Speech class', 'static/images/hate_speech.png'),
#                            ('Offensive Language class', 'static/images/offens_lang.png'),
#                            ('Neither class', 'static/images/neither.png')]

#         if len(final_features) >= 1:  # given sentence has 1 or more words after pre-processing
#             output = model.predict([final_features])[0]  # predict only one text
#             print(output)

#             # Set the sampling method of the Text Explainer (LIME algo)
#             sampler = MaskingTextSampler(
#                 # generate samples that contain all the original words of the given text
#                 min_replace=0,
#                 # replace no more than [number of words in final_features - 1] in order to never generate empty strings
#                 max_replace=len(final_features) - 1
#             )

#             te.set_params(sampler=sampler)  # set the sampler that creates the 5000 random text samples

#             # predict_proba: Black-box classification pipeline. predict_proba should be a function which takes a list of
#             #                strings (documents) and return a matrix of shape (n_samples, n_classes)
#             # LIME algorithm:
#             # generate distorted versions of the text,
#             # predict probabilities for these distorted texts using the black-box classifier,
#             # train another classifier which tries to predict output of a black-box classifier on these texts
#             # By default TextExplainer generates 5000 distorted texts
#             final_features = ' '.join(final_features)
#             print(final_features)
#             te.fit(final_features, predict_prob)  # form_text[0]

#             # top_targets: number of targets/classes to show
#             # targets=[output]: select targets/classes to show by name
#             # target_names: the order of the classes is the order produced by the classifier (XGBoost used in pipeline)
#             top_2_preds = te.show_prediction(top_targets=2, target_names=["hate speech", "neither", "offensive language"])
#             # print(top_2_preds.data)  # see the HTML code of the explanation

#             # show how close the results of the white box classifier are compared to the black box one (pipeline)
#             print(te.metrics_)  # mean_KL_divergence -> small (0%), score -> big (100%)

#         else:  # given sentence has 0 words after pre-processing
#             from flask import Markup
#             # Pass html code from FLASK to HTML template (needed for HTML file to recognise text as HTML)
#             explain_html = Markup("<p>Given sentence is categorized as 'neither' because it contains only stopwords, thus after pre-processing it results in an empty string.</p>")
#             return render_template('index.html', hate_speech_text="Hate Speech / Offensive Language Prediction:",
#                                    pre_predict_text="'" + form_text[0] + "'", predict_text="is",
#                                    prediction_text='neither',
#                                    expla_text='Explanation',
#                                    explain_top_2_preds=explain_html,
#                                    wordclouds=wordcloud_descr)

#         # ==============================================================================================================

#         return render_template('index.html', hate_speech_text="Hate Speech / Offensive Language Prediction:",
#                                pre_predict_text="'"+form_text[0]+"'", predict_text="is mostly", prediction_text=output,
#                                expla_text='Explanation', explain_top_2_preds=top_2_preds, wordclouds=wordcloud_descr)
#     else:  # if page is reloaded the form_text array will be empty
#         return render_template('index.html')


# if __name__ == "__main__":
#     app.run()
# from flask import Flask, render_template, request, Markup
# import pandas as pd
# import numpy as np
# import pickle
# import clean_data.preprocessing
# from eli5.lime import TextExplainer
# from eli5.lime.samplers import MaskingTextSampler
# from sklearn.feature_extraction.text import TfidfVectorizer
# from tensorflow.compat.v1 import ConfigProto, InteractiveSession
# from tensorflow.keras.models import load_model
# from keras.preprocessing import sequence

# # Flask utils
# from werkzeug.utils import secure_filename

# # Configure TensorFlow
# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# # Initialize Flask application
# app = Flask(__name__)

# app.debug = False
# app.secret_key = "your_key"

# # Define function to return token
# def dummy(token):
#     return token

# # Load models and tokenizer
# MODEL_PATH = 'DL.h5'
# model = load_model(MODEL_PATH)
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# model2 = pickle.load(open('model.pkl', 'rb'))

# # # Load and preprocess dataset
# # df = pd.read_csv("dataset/hate_tweets.csv")
# # tweet_df = df[['text', 'airline_sentiment']]
# # tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
# # sentiment_label = tweet_df.airline_sentiment.factorize()

# # Text preprocessing
# reading = clean_data.preprocessing.preprocessing(convert_lower=True, use_spell_corrector=True, only_verbs_nouns=False)

# # TextExplainer instance
# te = TextExplainer(vec=TfidfVectorizer(ngram_range=(1, 2), preprocessor=dummy, token_pattern='(?u)\\b\\w+\\b'),
#                    n_samples=5000, char_based=False, random_state=42)

# # Function to get prediction class name
# def one_word_get_prediction_class_name(prediction):
#     if prediction == 0:
#         return "hate speech"
#     elif prediction == 1:
#         return "neither"
#     else:
#         return "offensive language"

# # Function to predict probabilities
# def predict_prob(text):
#     text = [sentence.split() for sentence in text]
#     prob = model.predict_proba(text)
#     return prob

# @app.route('/')
# @app.route('/first')
# def first():
#     return render_template('first.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

# @app.route('/upload')
# def upload():
#     return render_template('upload.html')

# @app.route('/preview', methods=["POST"])
# def preview():
#     if request.method == 'POST':
#         dataset = request.files['datasetfile']
#         df = pd.read_csv(dataset, encoding='unicode_escape')
#         return render_template("preview.html", df_view=df)

# @app.route('/home')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     form_text = [request.form.get('hate_speech_text_field')]

#     if len(form_text):
#         final_features = reading.clean_text(form_text[0])
#         print(final_features)

#         wordcloud_descr = [('Hate Speech class', 'static/images/hate_speech.png'),
#                            ('Offensive Language class', 'static/images/offens_lang.png'),
#                            ('Neither class', 'static/images/neither.png')]

#         if len(final_features) >= 1:
#             output = model.predict([final_features])[0]
#             print(output)

#             sampler = MaskingTextSampler(min_replace=0, max_replace=len(final_features) - 1)
#             te.set_params(sampler=sampler)
#             final_features = ' '.join(final_features)
#             print(final_features)
#             te.fit(final_features, predict_prob)
#             top_2_preds = te.show_prediction(top_targets=2, target_names=["hate speech", "neither", "offensive language"])
#             print(te.metrics_)

#         else:
#             explain_html = Markup("<p>Given sentence is categorized as 'neither' because it contains only stopwords, thus after pre-processing it results in an empty string.</p>")
#             return render_template('index.html', hate_speech_text="Hate Speech / Offensive Language Prediction:",
#                                    pre_predict_text="'" + form_text[0] + "'", predict_text="is",
#                                    prediction_text='neither', expla_text='Explanation',
#                                    explain_top_2_preds=explain_html, wordclouds=wordcloud_descr)

#         return render_template('index.html', hate_speech_text="Hate Speech / Offensive Language Prediction:",
#                                pre_predict_text="'" + form_text[0] + "'", predict_text="is mostly",
#                                prediction_text=output, expla_text='Explanation',
#                                explain_top_2_preds=top_2_preds, wordclouds=wordcloud_descr)
#     else:
#         return render_template('index.html')

# if __name__ == "__main__":
#     app.run()
from flask import Flask, render_template, request, Markup
import pandas as pd
import numpy as np
import pickle
import clean_data.preprocessing
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.models import load_model
from keras.preprocessing import sequence

# Flask utils
from werkzeug.utils import secure_filename

# Configure TensorFlow
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Initialize Flask application
app = Flask(__name__)

# Define function to return token
def dummy(token):
    return token

# Load models and tokenizer
# MODEL_PATH = 'DL.h5'

model = pickle.load(open('model.pkl', 'rb'))
# model = load_model(MODEL_PATH)
# import pickle

# with open('model.pkl', 'rb') as handle:
#     tokenizer = pickle.load(handle)


# model2 = pickle.load(open('model.pkl', 'rb'))

# Load and preprocess dataset
df = pd.read_csv("dataset/hate_tweets.csv")
# tweet_df = df[['text', 'airline_sentiment']]
# tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
# sentiment_label = tweet_df.airline_sentiment.factorize()

# Text preprocessing
reading = clean_data.preprocessing.preprocessing(convert_lower=True, use_spell_corrector=True, only_verbs_nouns=False)

# TextExplainer instance
te = TextExplainer(vec=TfidfVectorizer(ngram_range=(1, 2), preprocessor=dummy, token_pattern='(?u)\\b\\w+\\b'),
                   n_samples=5000, char_based=False, random_state=42)

# Function to get prediction class name
def one_word_get_prediction_class_name(prediction):
    if prediction == 0:
        return "hate speech"
    elif prediction == 1:
        return "neither"
    else:
        return "offensive language"

# Function to predict probabilities
def predict_prob(text):
    '''
    MUST function that returns predicted probas of pipeline model, because text MUST BE TOKENIZED + CLEANED from empty strings
    :param text: all 5000 random generated instances from the initial given text
    :return: predicted probas for each data instance from pickled pipeline model
    '''
    text = [sentence.split() for sentence in text]  # TOKENIZE TEXT

    prob = model.predict_proba(text)

    return prob

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        return render_template("preview.html", df_view=df)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_text = [request.form.get('hate_speech_text_field')]
    # str(bytes_string, 'utf-8') # convert byte-string variable into a regular string

    if len(form_text): # page not reloaded, form_text array not empty [fix the bug of page reloading -  it return no values from forms]
        # preprocessing of the sentence about to predict
        final_features = reading.clean_text(form_text[0])

        print(final_features)

        # ==============================================================================================================
        # Explain prediction
        # ==============================================================================================================

        # The paths of produced wordcloud for each individual class of the dataset
        wordcloud_descr = [('Hate Speech class', 'static/images/hate_speech.png'),
                           ('Offensive Language class', 'static/images/offens_lang.png'),
                           ('Neither class', 'static/images/neither.png')]

        if len(final_features) >= 1:  # given sentence has 1 or more words after pre-processing
            output = model.predict([final_features])[0]  # predict only one text
            print(output)

            # Set the sampling method of the Text Explainer (LIME algo)
            sampler = MaskingTextSampler(
                # generate samples that contain all the original words of the given text
                min_replace=0,
                # replace no more than [number of words in final_features - 1] in order to never generate empty strings
                max_replace=len(final_features) - 1
            )

            te.set_params(sampler=sampler)  # set the sampler that creates the 5000 random text samples

            # predict_proba: Black-box classification pipeline. predict_proba should be a function which takes a list of
            #                strings (documents) and return a matrix of shape (n_samples, n_classes)
            # LIME algorithm:
            # generate distorted versions of the text,
            # predict probabilities for these distorted texts using the black-box classifier,
            # train another classifier which tries to predict output of a black-box classifier on these texts
            # By default TextExplainer generates 5000 distorted texts
            final_features = ' '.join(final_features)
            print(final_features)
            te.fit(final_features, predict_prob)  # form_text[0]

            # top_targets: number of targets/classes to show
            # targets=[output]: select targets/classes to show by name
            # target_names: the order of the classes is the order produced by the classifier (XGBoost used in pipeline)
            top_2_preds = te.show_prediction(top_targets=2, target_names=["hate speech", "neither", "offensive language"])
            # print(top_2_preds.data)  # see the HTML code of the explanation

            # show how close the results of the white box classifier are compared to the black box one (pipeline)
            print(te.metrics_)  # mean_KL_divergence -> small (0%), score -> big (100%)

        else:  # given sentence has 0 words after pre-processing
            from flask import Markup
            # Pass html code from FLASK to HTML template (needed for HTML file to recognise text as HTML)
            explain_html = Markup("<p>Given sentence is categorized as 'neither' because it contains only stopwords, thus after pre-processing it results in an empty string.</p>")
            return render_template('index.html', hate_speech_text="Hate Speech / Offensive Language Prediction:",
                                   pre_predict_text="'" + form_text[0] + "'", predict_text="is",
                                   prediction_text='neither',
                                   expla_text='Explanation',
                                   explain_top_2_preds=explain_html,
                                   wordclouds=wordcloud_descr)

        # ==============================================================================================================

        return render_template('index.html', hate_speech_text="Hate Speech / Offensive Language Prediction:",
                               pre_predict_text="'"+form_text[0]+"'", predict_text="is mostly", prediction_text=output,
                               expla_text='Explanation', explain_top_2_preds=top_2_preds, wordclouds=wordcloud_descr)
    else:  # if page is reloaded the form_text array will be empty
        return render_template('index.html')
if __name__ == "__main__":
    app.run(port=5002)
