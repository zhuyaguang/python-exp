import kfp
import kfp.dsl as dsl
import kfp.components as components
from kfp.components import func_to_container_op, InputPath, OutputPath
from typing import NamedTuple

def twitter_sample_dowload_and_preprocess(log_folder:str) -> NamedTuple('Outputs', [('logdir',str)]):
    import re
    import string
    import pandas as pd
    from random import shuffle
    import nltk
    import joblib
    from nltk.corpus import twitter_samples
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import TweetTokenizer
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    from nltk import data
    
    data.path.append(log_folder)
    nltk.download('twitter_samples', download_dir = log_folder)
    nltk.download('stopwords', download_dir = log_folder)
    
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    print(f"positive sentiment GOOD total samples {len(pos_tweets)}")
    print(f"negative sentiment  Bad total samples {len(neg_tweets)}")
    
    class Twitter_Preprocess():
    
        def __init__(self):
            self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                           reduce_len=True)
            self.stopwords_en = stopwords.words('english') 
            self.punctuation_en = string.punctuation
            self.stemmer = PorterStemmer() 

        def __remove_unwanted_characters__(self, tweet):
            tweet = re.sub(r'^RT[\s]+', '', tweet)
            tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
            tweet = re.sub(r'#', '', tweet)
            tweet = re.sub('\S+@\S+', '', tweet)
            tweet = re.sub(r'\d+', '', tweet)
            return tweet

        def __tokenize_tweet__(self, tweet):        
            return self.tokenizer.tokenize(tweet)

        def __remove_stopwords__(self, tweet_tokens):
            tweets_clean = []

            for word in tweet_tokens:
                if (word not in self.stopwords_en and 
                    word not in self.punctuation_en):
                    tweets_clean.append(word)
            return tweets_clean

        def __text_stemming__(self,tweet_tokens):
            tweets_stem = [] 

            for word in tweet_tokens:
                stem_word = self.stemmer.stem(word)  
                tweets_stem.append(stem_word)
            return tweets_stem

        def preprocess(self, tweets):
            tweets_processed = []
            for _, tweet in tqdm(enumerate(tweets)):        
                tweet = self.__remove_unwanted_characters__(tweet)            
                tweet_tokens = self.__tokenize_tweet__(tweet)            
                tweet_clean = self.__remove_stopwords__(tweet_tokens)
                tweet_stems = self.__text_stemming__(tweet_clean)
                tweets_processed.extend([tweet_stems])
            return tweets_processed
    
    twitter_text_processor = Twitter_Preprocess()
    processed_pos_tweets = twitter_text_processor.preprocess(pos_tweets)
    processed_neg_tweets = twitter_text_processor.preprocess(neg_tweets)
    
    def build_bow_dict(tweets, labels):
        freq = {}
        for tweet, label in list(zip(tweets, labels)):
            for word in tweet:
                freq[(word, label)] = freq.get((word, label), 0) + 1    
        return freq

    labels = [1 for i in range(len(processed_pos_tweets))]
    labels.extend([0 for i in range(len(processed_neg_tweets))])
    
    twitter_processed_corpus = processed_pos_tweets + processed_neg_tweets
    bow_word_frequency = build_bow_dict(twitter_processed_corpus, labels)
    
    shuffle(processed_pos_tweets)
    shuffle(processed_neg_tweets)

    positive_tweet_label = [1 for i in processed_pos_tweets]
    negative_tweet_label = [0 for i in processed_neg_tweets]

    tweet_df = pd.DataFrame(list(zip(twitter_processed_corpus,
                            positive_tweet_label+negative_tweet_label)),
                            columns=["processed_tweet", "label"])
    
    train_X_tweet, test_X_tweet, train_Y, test_Y = train_test_split(tweet_df["processed_tweet"],
                                                                    tweet_df["label"],
                                                                    test_size = 0.20,
                                                                    stratify=tweet_df["label"])
    
    print(f"train_X_tweet {train_X_tweet.shape}, test_X_tweet {test_X_tweet.shape}")
    print(f"train_Y {train_Y.shape}, test_Y {test_Y.shape}")
    
    joblib.dump(bow_word_frequency, log_folder + '/bow_word_frequency.pkl')
    joblib.dump(train_X_tweet, log_folder + '/train_X_tweet.pkl')
    joblib.dump(test_X_tweet, log_folder + '/test_X_tweet.pkl')
    joblib.dump(train_Y, log_folder + '/train_Y.pkl')
    joblib.dump(test_Y, log_folder + '/test_Y.pkl')
    
    return ([log_folder])

def numpy_process(log_folder:str) -> NamedTuple('Outputs', [('logdir',str), ('numpydir',str)]):
    
    import numpy as np
    import joblib
    import os
    
    bow_word_frequency = joblib.load(open(log_folder + '/bow_word_frequency.pkl','rb'))
    train_X_tweet = joblib.load(open(log_folder + '/train_X_tweet.pkl','rb'))
    test_X_tweet = joblib.load(open(log_folder + '/test_X_tweet.pkl','rb'))
    train_Y = joblib.load(open(log_folder + '/train_Y.pkl','rb'))
    test_Y = joblib.load(open(log_folder + '/test_Y.pkl','rb'))
    
    def extract_features(processed_tweet, bow_word_frequency):
        features = np.zeros((1,3))
        features[0,0] = 1

        for word in processed_tweet:
            features[0,1] = bow_word_frequency.get((word, 1), 0)+features[0,1]
            features[0,2] = bow_word_frequency.get((word, 0), 0)+features[0,2]
        return features
    
    train_X = np.zeros((len(train_X_tweet), 3))
    for index, tweet in enumerate(train_X_tweet):
        train_X[index, :] = extract_features(tweet, bow_word_frequency)

    test_X = np.zeros((len(test_X_tweet), 3))
    for index, tweet in enumerate(test_X_tweet):
        test_X[index, :] = extract_features(tweet, bow_word_frequency)

    print(f"train_X {train_X.shape}, test_X {test_X.shape}")
    
    if not os.path.isdir(log_folder + '/numpy'):
        os.makedirs(log_folder + '/numpy')
    
    numpy_folder = log_folder + '/numpy'
    
    joblib.dump(train_X, numpy_folder + '/train_X.pkl')
    joblib.dump(test_X, numpy_folder + '/test_X.pkl')
    
    return ([log_folder, numpy_folder])

def sklearn_logistic(log_folder:str, numpy_folder:str)->NamedTuple('Outputs',[('logdir',str), ('sklearndir',str), ('sklearnscore',float)]):
    
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    import joblib
    import os
    
    train_X = joblib.load(open(numpy_folder + '/train_X.pkl','rb'))
    test_X = joblib.load(open(numpy_folder + '/test_X.pkl','rb'))
    
    train_Y = joblib.load(open(log_folder + '/train_Y.pkl','rb'))
    test_Y = joblib.load(open(log_folder + '/test_Y.pkl','rb'))
    
    clf = SGDClassifier(loss='log')
    clf.fit(train_X, np.array(train_Y).reshape(-1,1))
    y_pred = clf.predict(test_X)
    y_pred_probs = clf.predict(test_X)
    
    print(f"Scikit learn logistic regression accuracy is {accuracy_score(test_Y , y_pred)*100:.2f}")
    
    if not os.path.isdir(numpy_folder + '/sklearn'):
        os.makedirs(numpy_folder + '/sklearn')
    sklearn_folder = numpy_folder + '/sklearn'
    joblib.dump(clf, sklearn_folder + '/sklearn.pkl')
    
    sklearn_score = accuracy_score(test_Y , y_pred)
    
    return ([log_folder, sklearn_folder, sklearn_score])

def logistic(log_folder:str, numpy_folder:str) -> NamedTuple('Outputs', [('logdir',str), ('logisticdir',str), ('logisticscore',float)]):
    
    import numpy as np
    import joblib
    import os
    
    train_X = joblib.load(open(numpy_folder + '/train_X.pkl','rb'))
    test_X = joblib.load(open(numpy_folder + '/test_X.pkl','rb'))
    train_Y = joblib.load(open(log_folder + '/train_Y.pkl','rb'))
    test_Y = joblib.load(open(log_folder + '/test_Y.pkl','rb'))
    
    def sigmoid(z): 
        h = 1 / (1+ np.exp(-z))
        return h
    
    def gradientDescent(x, y, theta, alpha, num_iters, c):
        m = x.shape[0]
        for i in range(0, num_iters):
            z = np.dot(x, theta)
            h = sigmoid(z)
            J = (-1/m) * ((np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1-h))) + (c * np.sum(theta)))
            theta = theta - (alpha / m) * np.dot((x.T), (h - y))
            J = float(J)
        return J, theta
    
    np.random.seed(1)
    J, theta = gradientDescent(train_X, np.array(train_Y).reshape(-1,1), np.zeros((3, 1)), 1e-7, 1000, 0.1)
    print(f"The cost after training is {J:.8f}.")
    print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")
    
    def predict_tweet(x, theta):
        y_pred = sigmoid(np.dot(x, theta))
        return y_pred
    
    predicted_probs = predict_tweet(test_X, theta)
    predicted_labels = np.where(predicted_probs > 0.5, 1, 0)
    print(f"Own implementation of logistic regression accuracy is {len(predicted_labels[predicted_labels == np.array(test_Y).reshape(-1,1)]) / len(test_Y)*100:.2f}")
    
    if not os.path.isdir(numpy_folder + '/logistic'):
        os.makedirs(numpy_folder + '/logistic')
    logistic_folder = numpy_folder + '/logistic'
    joblib.dump(theta, logistic_folder + '/logistic.pkl')
    
    logistic_score = len(predicted_labels[predicted_labels == np.array(test_Y).reshape(-1,1)]) / len(test_Y)
    
    return ([log_folder, logistic_folder, logistic_score])

def torch_process_logistic(log_folder:str) -> NamedTuple('Outputs', [('logdir',str), ('torchdir',str),  ('torchscore',float)]):
    
    import torch
    import joblib
    import os

    bow_word_frequency = joblib.load(open(log_folder + '/bow_word_frequency.pkl','rb'))
    train_X_tweet = joblib.load(open(log_folder + '/train_X_tweet.pkl','rb'))
    test_X_tweet = joblib.load(open(log_folder + '/test_X_tweet.pkl','rb'))
    train_Y = joblib.load(open(log_folder + '/train_Y.pkl','rb'))
    test_Y = joblib.load(open(log_folder + '/test_Y.pkl','rb'))
    
    def extract_features(processed_tweet, bow_word_frequency):
        features = torch.zeros((1,3))
        features[0,0] = 1

        for word in processed_tweet:
            features[0,1] = bow_word_frequency.get((word, 1), 0) + features[0,1]
            features[0,2] = bow_word_frequency.get((word, 0), 0) + features[0,2]
        return features
    
    train_X_Tensor = torch.zeros((len(train_X_tweet), 3))
    for index, tweet in enumerate(train_X_tweet):
        train_X_Tensor[index, :] = extract_features(tweet, bow_word_frequency)

    test_X_Tensor = torch.zeros((len(test_X_tweet), 3))
    for index, tweet in enumerate(test_X_tweet):
        test_X_Tensor[index, :] = extract_features(tweet, bow_word_frequency)

    print(f"train_X_Tensor {train_X_Tensor.shape}, test_X_Tensor {test_X_Tensor.shape}")
    type(train_X_Tensor)
    
    def sigmoid(z):
        h = 1 / (1+ torch.exp(-z))
        return h
    
    def gradientDescent(x, y, theta, alpha, num_iters, c):

        m = x.shape[0]

        for i in range(0, num_iters):
            z = torch.mm(x, theta)
            h = sigmoid(z)
            J = (-1/m) * ((torch.mm(y.T,torch.log(h)) + torch.mm((1 - y).T, torch.log(1-h))) 
                          + (c * torch.sum(theta)))
            theta = theta - (alpha / m) * torch.mm((x.T), (h - y))
            J = float(J)
        return J, theta

    torch.manual_seed(1)
    J, theta = gradientDescent(train_X_Tensor,
                               torch.reshape(torch.Tensor(train_Y.to_numpy()),(-1,1)),
                               torch.zeros((3,1)),1e-7,1000,0.1)
    print(f"The cost after training is {J:.8f}.")
    
    def predict_tweet(x,theta):
        y_pred = sigmoid(torch.mm(x,theta))
        return y_pred
    
    predicted_probs =predict_tweet(test_X_Tensor, theta)
    prediceted_probs=torch.tensor(predicted_probs)
    predicted_labels = torch.where(predicted_probs >0.5, torch.tensor(1), torch.tensor(0))
    print(f"Pytorch of logistic regression accuracy is {len(predicted_labels[predicted_labels == torch.reshape(torch.Tensor(test_Y.to_numpy()),(-1,1))]) / len(test_Y)*100:.2f}")
    
    if not os.path.isdir(log_folder + '/torch'):
        os.makedirs(log_folder + '/torch')
    torch_folder = log_folder + '/torch'
    joblib.dump(theta, torch_folder + '/torch.pkl')
    
    torch_score = len(predicted_labels[predicted_labels == torch.reshape(torch.Tensor(test_Y.to_numpy()),(-1,1))]) / len(test_Y)
    
    return ([log_folder, torch_folder, torch_score])


def svm_process(log_folder:str, numpy_folder:str) -> NamedTuple('Outputs', [('svmdir',str), ('svmscore',float)]):
    import joblib
    import os
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    train_X = joblib.load(open(numpy_folder + '/train_X.pkl','rb'))
    test_X = joblib.load(open(numpy_folder + '/test_X.pkl','rb'))
    train_Y = joblib.load(open(log_folder + '/train_Y.pkl','rb'))
    test_Y = joblib.load(open(log_folder + '/test_Y.pkl','rb'))
    
    scaler = StandardScaler()
    train_X_s = scaler.fit(train_X).transform(train_X)
    
    clf = SVC(kernel='linear')
    t = clf.fit(train_X_s, np.array(train_Y).reshape(-1,1))
    y_pred = clf.predict(test_X)
    svm_score = accuracy_score(test_Y , y_pred)
    
    if not os.path.isdir(numpy_folder + '/svm'):
        os.makedirs(numpy_folder + '/svm')
    svm_folder = numpy_folder + '/svm'
    joblib.dump(t, svm_folder + '/svm.pkl')

    return ([svm_folder, svm_score])

def accuracy(sklearn_score:float,logistic_score:float,torch_score:float,svm_score:float) -> NamedTuple('Outputs', [('mlpipeline_metrics', 'Metrics')]):
  import json

  metrics = {
    'metrics': [
        {
          'name': 'sklearn_score',
          'numberValue':  sklearn_score,
          'format': "PERCENTAGE",
        },
        {
          'name': 'logistic_score',
          'numberValue':  logistic_score,
          'format': "PERCENTAGE",
        },
        {
          'name': 'torch_score',
          'numberValue':  torch_score,
          'format': "PERCENTAGE",
        },
        {
          'name': 'svm_score',
          'numberValue':  svm_score,
          'format': "PERCENTAGE",
        },
    ]
  }
  return [json.dumps(metrics)]

def http_port(log_folder:str, sklearn_folder:str, logistic_folder:str, torch_folder:str, svm_folder:str):
    
    import re
    import string
    import pandas as pd
    from random import shuffle
    import torch
    import numpy as np
    import nltk
    import joblib
    from nltk.corpus import twitter_samples
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import TweetTokenizer
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    from nltk import data
    from flask import Flask,render_template,url_for,request
    
    data.path.append(log_folder)

    app = Flask(__name__,template_folder='/http-port/templates')

    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/predict', methods=['POST'])
    def predict():

        class Preprocess():   
            def __init__(self):
                self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
                self.stopwords_en = stopwords.words('english') 
                self.punctuation_en = string.punctuation
                self.stemmer = PorterStemmer()        
            def __remove_unwanted_characters__(self, tweet):
                tweet = re.sub(r'^RT[\s]+', '', tweet)
                tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
                tweet = re.sub(r'#', '', tweet)
                tweet = re.sub('\S+@\S+', '', tweet)
                tweet = re.sub(r'\d+', '', tweet)
                return tweet    
            def __tokenize_tweet__(self, tweet):        
                return self.tokenizer.tokenize(tweet)   
            def __remove_stopwords__(self, tweet_tokens):
                tweets_clean = []
                for word in tweet_tokens:
                    if (word not in self.stopwords_en and 
                        word not in self.punctuation_en):
                        tweets_clean.append(word)
                return tweets_clean   
            def __text_stemming__(self,tweet_tokens):
                tweets_stem = [] 
                for word in tweet_tokens:
                    stem_word = self.stemmer.stem(word)  
                    tweets_stem.append(stem_word)
                return tweets_stem
            def preprocess(self, tweets):
                tweets_processed = []
                for _, tweet in tqdm(enumerate(tweets)):        
                    tweet = self.__remove_unwanted_characters__(tweet)            
                    tweet_tokens = self.__tokenize_tweet__(tweet)            
                    tweet_clean = self.__remove_stopwords__(tweet_tokens)
                    tweet_stems = self.__text_stemming__(tweet_clean)
                    tweets_processed.extend([tweet_stems])
                return tweets_processed
            
        def extract_features(processed_tweet, bow_word_frequency):
            features = np.zeros((1,3))
            features[0,0] = 1
            for word in processed_tweet:
                features[0,1] = bow_word_frequency.get((word, 1), 0) + features[0,1]
                features[0,2] = bow_word_frequency.get((word, 0), 0) + features[0,2]
            return features

        def sigmoid(z): 
            h = 1 / (1+ np.exp(-z))
            return h

        def predict_tweet(x, theta_ns):
            y_pred = sigmoid(np.dot(x, theta_ns))   
            return y_pred

        def extract_features_torch(processed_tweet, bow_word_frequency):
            features = torch.zeros((1,3))
            features[0,0] = 1
            for word in processed_tweet:
                features[0,1] = bow_word_frequency.get((word, 1), 0) + features[0,1]
                features[0,2] = bow_word_frequency.get((word, 0), 0) + features[0,2]
            return features

        def sigmoid_torch(z):
            h = 1 / (1+ torch.exp(-z))   
            return h

        def predict_tweet_torch(x,theta_toc):
            y_pred = sigmoid_torch(torch.mm(x,theta_toc))
            return y_pred

        text_processor = Preprocess()
        
        bow_word_frequency = joblib.load(open(log_folder + '/bow_word_frequency.pkl','rb'))  
        theta_ns = joblib.load(open(logistic_folder + '/logistic.pkl','rb'))
        clf = joblib.load(open(sklearn_folder + '/sklearn.pkl','rb'))
        theta_toc = joblib.load(open(torch_folder + '/torch.pkl','rb'))
        svm = joblib.load(open(svm_folder + '/svm.pkl','rb'))

        if request.method == 'POST':
            message = request.form['message']
            data = [message]
            data = text_processor.preprocess(data)
            
            data_o = str(data)
            data_o = data_o[2:len(data_o)-2]

            vect = np.zeros((1, 3))
            for index, tweet in enumerate(data):
                vect[index, :] = extract_features(tweet, bow_word_frequency)
            predicted_probs_np = predict_tweet(vect, theta_ns)
            my_prediction_np = np.where(predicted_probs_np > 0.5, 1, 0)

            my_prediction_skl = clf.predict(vect)

            vect_Tensor = torch.zeros((1, 3))
            for index, tweet in enumerate(data):
                vect_Tensor[index, :] = extract_features_torch(
                    tweet, bow_word_frequency)
            predicted_probs_toc = predict_tweet_torch(vect_Tensor, theta_toc)
            my_prediction_toc = torch.where(
                predicted_probs_toc > 0.5, torch.tensor(1), torch.tensor(0))
            
            my_prediction_svm = svm.predict(vect)
            
        return render_template('home.html',
                                message = message,
                                data = data_o,
                                my_prediction_np = my_prediction_np,
                                my_prediction_skl = my_prediction_skl,
                                my_prediction_toc = my_prediction_toc,
                                my_prediction_svm = my_prediction_svm)

    if __name__ == '__main__':
        
        app.run(debug=True,use_reloader=False)


@dsl.pipeline(
    name='Twitter nltk pipeline',
    description='Writing code by the other way.'
)

def nltk_pipeline():
    
    log_folder = '/information'
    pvc_name = "twitter-5000"

    image = "dfm871002/nltk_env:2.4.2"
    
    vop = dsl.VolumeOp(
        name=pvc_name,
        resource_name="twitter-5000",
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWM
    )
    
    dowload_op = func_to_container_op(
        func = twitter_sample_dowload_and_preprocess,
        base_image = image,
    )
    
    numpy_op = func_to_container_op(
        func = numpy_process,
        base_image = image,
    )
    
    sklearn_op = func_to_container_op(
        func = sklearn_logistic,
        base_image = image,
    )
    
    logistic_op = func_to_container_op(
        func = logistic,
        base_image = image,
    )
    
    torch_op = func_to_container_op(
        func = torch_process_logistic,
        base_image = image,
    )
    
    svm_op = func_to_container_op(
        func = svm_process,
        base_image = image,
    )
    
    accuracy_op = func_to_container_op(
        func = accuracy,
        base_image = image,
    )
    
    http_op = func_to_container_op(
        func = http_port,
        base_image = image,
    )
    
    dowload_task = dowload_op(log_folder).add_pvolumes({ log_folder:vop.volume, })
    
    numpy_task = numpy_op(dowload_task.outputs['logdir']).add_pvolumes({ log_folder:vop.volume, })
    
    svm_task = svm_op(numpy_task.outputs['logdir'], numpy_task.outputs['numpydir']).add_pvolumes({ log_folder:vop.volume, })
    
    sklearn_task = sklearn_op(
                                numpy_task.outputs['logdir'],
                                numpy_task.outputs['numpydir']
    ).add_pvolumes({ log_folder:vop.volume, })
    
    logistic_task = logistic_op(
                                numpy_task.outputs['logdir'],
                                numpy_task.outputs['numpydir']
    ).add_pvolumes({ log_folder:vop.volume, })
    
    torch_task = torch_op(
                            dowload_task.outputs['logdir']
    ).add_pvolumes({ log_folder:vop.volume, })
    
    accuracy_task = accuracy_op(
                        sklearn_task.outputs['sklearnscore'],
                        logistic_task.outputs['logisticscore'],
                        torch_task.outputs['torchscore'],
                        svm_task.outputs['svmscore']
    )
        
    http_task = http_op(
                        sklearn_task.outputs['logdir'],
                        sklearn_task.outputs['sklearndir'],
                        logistic_task.outputs['logisticdir'],
                        torch_task.outputs['torchdir'],
                        svm_task.outputs['svmdir']
    ).add_pvolumes({ log_folder:vop.volume, })

kfp.compiler.Compiler().compile(nltk_pipeline, 'twitter-5000.zip')