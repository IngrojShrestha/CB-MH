import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from nltk import word_tokenize, wordpunct_tokenize, RegexpTokenizer, PorterStemmer
from keras.callbacks import Callback, EarlyStopping
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, MaxPooling1D, Conv1D, Bidirectional, GlobalMaxPool1D, Dropout, Activation
from keras.layers import CuDNNLSTM as LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras import regularizers
from keras.models import Model
from gensim.models import KeyedVectors
import keras.backend as K

CRED = '\033[91m'
CEND = '\033[0m'

OUTPUT_PATH = "/path/to/output/"
TEST_GS_FILENAME = "test_gs.csv"
TEST_PRED_FILENAME = "test_preds.csv"
MHA_UNITS = 768
MHA_NUM_HEADS = 12
PRETRAINED_MODEL = "https://archive.org/download/pubmed2018_w2v_400D.tar/pubmed2018_w2v_400D.tar.gz"

porter_stemmer = PorterStemmer()
stop_words = stopwords.words('english')


def stem_sentences(sentence):
    '''
        performs stemming on provided text

        param sentence: user provided text
        type sentence: str

        returns: stemmed sentence
        rtype: str
    '''

    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def remove_stop_word_punctuation(sentence):
    '''
        removes stop word and punctuation from given text

        param sentence: user provided text
        type sentence: str

        returns: preprocessed sentence
        rtype: str
    '''

    stop_words = set(stopwords.words('english'))

    word_tokens = []

    for token in word_tokenize(sentence):
        for t in wordpunct_tokenize(token):
            # remove punctuations
            tokenizer = RegexpTokenizer(r'\w+')
            r = tokenizer.tokenize(t)
            if r:
                word_tokens.append(tokenizer.tokenize(t)[0])

    # remove stop words
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(x for x in filtered_sentence)


def load_preprocess_data():
    '''
        load training, validation and testing data and perform pre-processing (stemming and stop word removal)

        returns: training, validation and testing data
        rtype: dataframes
    '''
    train = pd.read_csv("train.csv", sep='\t')
    val = pd.read_csv("val.csv", sep='\t')
    test = pd.read_csv("test.csv", sep="\t")
    test_raw = pd.read_csv("test.csv", sep="\t")

    # remove stop words from clinical note (HPISection)
    train['HPISection'] = train['HPISection'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    val['HPISection'] = val['HPISection'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    test['HPISection'] = test['HPISection'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # stemming
    train['HPISection'] = train["HPISection"].apply(stem_sentences)
    val['HPISection'] = val["HPISection"].apply(stem_sentences)
    test['HPISection'] = test["HPISection"].apply(stem_sentences)

    return train, val, test, test_raw


def precision(y_true, y_pred):
    micro_precision = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    macro_precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    return micro_precision, macro_precision


def recall(y_true, y_pred):
    micro_recall = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    macro_recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    return micro_recall, macro_recall


def compute_f1_score(y_true, y_pred):
    micro_f1score = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    macro_f1score = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    return micro_f1score, macro_f1score


def compute_micro_f2_score(p, r):
    return (5 * p * r) / (4 * p + r)


def evaluation(y_true, y_pred, target_names, dataset):
    micro_precision, macro_precision = precision(y_true, y_pred)
    micro_recall, macro_recall = recall(y_true, y_pred)
    micro_f1score, macro_f1score = compute_f1_score(y_true, y_pred)
    micro_f2score = compute_micro_f2_score(p=micro_precision.astype(float), r=micro_recall.astype(float))

    if dataset == 'val':
        print(
            "\nval_precision:{} val_recall:{} val_f1:{} val_f2:{} ".format(micro_precision, micro_recall, micro_f1score,
                                                                           micro_f2score))

    elif dataset == 'test':
        print("\ntest_precision:{} test_recall:{} test_f1:{} test_f2:{}".format(micro_precision, micro_recall,
                                                                                micro_f1score, micro_f2score))
    print(classification_report(y_true, y_pred, target_names=target_names))


class rocCallBack(Callback):
    def __init__(self, validation_data, threshold, target_names):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.threshold = threshold
        self.target_names = target_names

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        y_true = self.y_val.astype(float)

        y_pred = self.model.predict(self.x_val)

        if self.threshold is None:
            y_pred = np.around(y_pred.astype(float))
        elif self.threshold > 0 and self.threshold < 1:
            y_pred = y_pred.astype(float)
            y_pred[y_pred >= self.threshold] = 1
            y_pred[y_pred < self.threshold] = 0
        else:
            print(CRED + "Invalid threshold" + CEND)
            sys.exit()

        evaluation(y_true, y_pred, self.target_names, dataset='val')

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def load_pretrained_embedding():
    return KeyedVectors.load_word2vec_format(PRETRAINED_MODEL, binary=True)


def fill_in_missing_words_with_zeros(word2vec, word2idx, EMBEDDING_DIM, NUM_WORDS, MAX_VOCAB_SIZE):
    embedding_matrix = np.zeros((NUM_WORDS, EMBEDDING_DIM))
    for word, i in word2idx.items():
        if i < MAX_VOCAB_SIZE:
            if word in word2vec.vocab:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = word2vec[word]
    return embedding_matrix


# adapted from https://github.com/CyberZHG/keras-multi-head
class MultiHeadAttention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        super(MultiHeadAttention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)

        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')

        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def format_test(df, mlb):
    X_test = df['HPISection']
    df['Dx'] = df['Dx'].str.split(',')
    y_test = df['Dx']
    y_test = mlb.transform(list(y_test))

    df_test_target = pd.DataFrame({'anxiety_disorder': y_test[:, 0],
                                   'bipolar_disorder': y_test[:, 1],
                                   'cluster_b': y_test[:, 2],
                                   'unipolar_depression': y_test[:, 3],
                                   'other': y_test[:, 4],
                                   'PTSD_OCD_PanicDisorder': y_test[:, 5],
                                   'psychotic_disorder': y_test[:, 6],
                                   'substance_use_disorder': y_test[:, 7]
                                   })

    test_HPISection_dict = {'HPISection': X_test}
    df_test_predicate = pd.DataFrame(test_HPISection_dict)
    df_test = pd.concat([df_test_predicate, df_test_target], axis=1)
    df_test['id'] = df['ID']
    return df_test, y_test


def save_gs(test_df):
    '''
    saves gold standards to a file
    '''
    df_test_y = test_df[['id', 'anxiety_disorder',
                         'bipolar_disorder',
                         'cluster_b',
                         'unipolar_depression',
                         'other',
                         'PTSD_OCD_PanicDisorder',
                         'psychotic_disorder',
                         'substance_use_disorder']]
    df_test_y.to_csv(os.path.join(OUTPUT_PATH, TEST_GS_FILENAME), index=False)


def save_predictions(predicted, ids):
    '''
    saves model predictions
    '''
    sub_df = pd.DataFrame(predicted, columns=['anxiety_disorder',
                                              'bipolar_disorder',
                                              'cluster_b',
                                              'unipolar_depression',
                                              'other',
                                              'PTSD_OCD_PanicDisorder',
                                              'psychotic_disorder',
                                              'substance_use_disorder'])
    sub_df['id'] = ids
    sub_df = sub_df[['id', 'anxiety_disorder',
                     'bipolar_disorder',
                     'cluster_b',
                     'unipolar_depression',
                     'other',
                     'PTSD_OCD_PanicDisorder',
                     'psychotic_disorder',
                     'substance_use_disorder']]

    sub_df.to_csv(os.path.join(OUTPUT_PATH, TEST_PRED_FILENAME), index=False)


class CB_MH:
    def __init__(self):
        self.MODEL_NAME = 'CB-MH'
        self.CONV_LAYERS = 1
        self.TARGET_NAMES = []

        self.MAX_VOCAB_SIZE = 40000
        self.NUM_WORDS = 0
        self.MAX_SEQ_LENGTH = 0

        self.BATCH_SIZE = 100
        self.EPOCHS = 10
        self.NUM_LABELS = 0

        self.IS_PRE_TRAINED_EMBEDDING = True
        self.EMBEDDING_DIM = 400
        self.embedding_matrix = []
        self.THRESHOLD = 0.2

        # CNN
        self.CNN_LAYERS = 1
        self.FILTER_SIZES = 3
        self.NUM_FILTERS = 250
        self.STRIDES = 1
        self.POOL_SIZE = 4

        # BiLSTM
        self.HIDDEN_LAYER_SIZE = 128
        self.HIDDEN_LAYERS = 1

        self.MODEL_FILENAME = "{}.h5".format(self.MODEL_NAME)

        a = Input(shape=(32,))
        b = Dense(32)(a)
        model = Model(inputs=a, outputs=b)
        self.model = model

    def set_num_words(self, word2idx):
        self.NUM_WORDS = min(self.MAX_VOCAB_SIZE, len(word2idx) + 1)

    def __repr__(self):
        return '\nModel:{} ' \
               'Pre-trained Embedding:{} ' \
               'Max sequence length:{} ' \
               'Vocabulary size/Num features:{} ' \
               'Labels: {}'.format(self.MODEL_NAME,
                                   self.IS_PRE_TRAINED_EMBEDDING,
                                   self.MAX_SEQ_LENGTH,
                                   self.NUM_WORDS,
                                   self.NUM_LABELS)

    # fit a tokenizer/convert the sentences (strings) into integers
    def create_tokenizer(self, sentences):
        tokenizer = Tokenizer(num_words=self.MAX_VOCAB_SIZE)
        tokenizer.fit_on_texts(sentences)
        return tokenizer

    def build_model(self):
        print("\nBuilding model....")

        model_input = Input(shape=(self.MAX_SEQ_LENGTH,), )

        if self.IS_PRE_TRAINED_EMBEDDING:
            # load pre-trained word embeddings into an Embedding layer
            embedding_layer = Embedding(input_dim=self.NUM_WORDS,
                                        output_dim=self.EMBEDDING_DIM,
                                        weights=[self.embedding_matrix],
                                        input_length=self.MAX_SEQ_LENGTH,
                                        trainable=True)

        else:
            # randomly initialized word embedding
            embedding_layer = Embedding(input_dim=self.NUM_WORDS,
                                        output_dim=self.EMBEDDING_DIM,
                                        input_length=self.MAX_SEQ_LENGTH,
                                        trainable=True)

        z = embedding_layer(model_input)
        z = Dropout(0.5)(z)

        # Convolutional block
        conv = Conv1D(filters=self.NUM_FILTERS,
                      kernel_size=self.FILTER_SIZES,
                      strides=self.STRIDES,
                      padding="valid",
                      activation="relu", kernel_regularizer=regularizers.l2(0.0001))(z)

        # max pooling
        z = MaxPooling1D(pool_size=self.POOL_SIZE)(conv)

        # BiLSTM block
        if len(K.tensorflow_backend._get_available_gpus()) > 0:
            z = Bidirectional(LSTM(self.HIDDEN_LAYER_SIZE,
                                   kernel_regularizer=regularizers.l2(0.0001), return_sequences=True
                                   ))(z)
        else:
            z = Bidirectional(LSTM(self.HIDDEN_LAYER_SIZE, dropout=0.5, recurrent_dropout=0.2,
                                   kernel_regularizer=regularizers.l2(0.0001), return_sequences=True
                                   ))(z)

        # multi head attention (from transformer)
        z = MultiHeadAttention(MHA_NUM_HEADS, MHA_UNITS)([z, z, z])
        z = GlobalMaxPool1D()(z)
        z = Dropout(0.7)(z)
        z = Dense(self.NUM_LABELS)(z)
        model_output = Activation('sigmoid')(z)
        model = Model(model_input, model_output)

        # compile model
        model.compile(loss="binary_crossentropy", optimizer='adam')
        model.summary()

        self.model = model

    def train_model(self, x_train, y_train, x_val, y_val):
        print("\nTraining model....")
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            callbacks=[rocCallBack(
                validation_data=(x_val, y_val),
                threshold=self.THRESHOLD,
                target_names=self.TARGET_NAMES), es]
        )

        self.model.save(OUTPUT_PATH + self.MODEL_FILENAME)
        print("\nSaved model {} to {}".format(self.MODEL_FILENAME, OUTPUT_PATH))
        return history

    def predict(self, x_test):
        print("\n PREDICTION....")
        return self.model.predict(x_test, batch_size=None, steps=None)

    def get_encoded_dataset(self):
        '''
            loads training, validation and testing data, performs preprocessing (stemming, stop word removal, padding/truncation)

            returns: processed train, val and test document along with original test document and corresponding document id,  label encoder, test document ids
            rtype: dataframe, MultiLabelBinarizer, list
        '''

        print("\nLoading data (train, val, test)...")

        mlb = MultiLabelBinarizer()

        train, val, test, test_raw = load_preprocess_data()

        # extract training document (HPISection) and corresponding label (Dx)
        X_train = train['HPISection']
        train['Dx'] = train['Dx'].str.split(',')
        y_train = train['Dx']
        y_train = mlb.fit_transform(list(y_train))

        # set prediction labels
        self.TARGET_NAMES = mlb.classes_

        # extract validation set
        X_val = val['HPISection']
        val['Dx'] = val['Dx'].str.split(',')
        y_val = val['Dx']
        y_val = mlb.transform(list(y_val))

        # extract test data and format
        test, y_test = format_test(test, mlb)

        # save gold standard
        save_gs(test)

        # Extract test set
        X_test = test['HPISection']
        X_test_original = test_raw['HPISection']
        X_test_original_index = test_raw['index']

        self.MAX_SEQ_LENGTH = max([len(s.split()) for s in X_train])

        self.NUM_LABELS = y_train.shape[1]

        # fit a tokenizer
        tokenizer = self.create_tokenizer(X_train)

        # get word -> integer mapping
        word2idx = tokenizer.word_index

        print("\n Setting vocabulary size...")
        # Set vocabulary size
        self.set_num_words(word2idx)

        # save tokenizer
        # with open(OUTPUT_PATH + model_.MODEL_NAME + '_tokenizer.pickle', 'wb') as handle:
        #    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # pad sequences (train)
        x_train = tokenizer.texts_to_sequences(X_train)
        x_train = pad_sequences(x_train, maxlen=self.MAX_SEQ_LENGTH, padding='pre', truncating='pre')

        # pad sequences (test)
        x_test = tokenizer.texts_to_sequences(X_test)
        x_test = pad_sequences(x_test, maxlen=self.MAX_SEQ_LENGTH, padding='pre', truncating='pre')

        # pad sequences (val)
        x_val = tokenizer.texts_to_sequences(X_val)
        x_val = pad_sequences(x_val, maxlen=self.MAX_SEQ_LENGTH, padding='pre', truncating='pre')

        return x_train, y_train, x_val, y_val, x_test, y_test, X_test_original, X_test_original_index, word2idx, mlb, \
               test['id']


def main():
    start_time = time.time()

    # set model parameters
    model_ = CB_MH()

    # fit and text_to_sequence
    x_train, y_train, x_val, y_val, x_test, y_test, _, _, word2idx, mlb, test_ids = model_.get_encoded_dataset()

    if model_.IS_PRE_TRAINED_EMBEDDING:
        # load pre-trained embedding
        word2vec = load_pretrained_embedding()

        # words not found in embedding index will be all-zeros.
        model_.embedding_matrix = fill_in_missing_words_with_zeros(word2vec,
                                                                   word2idx,
                                                                   model_.EMBEDDING_DIM,
                                                                   model_.NUM_WORDS,
                                                                   model_.MAX_VOCAB_SIZE)

    # build model
    model_.build_model()

    # train model
    model_.train_model(x_train, y_train, x_val, y_val)

    # prediction on test data
    prediction = model_.predict(x_test)

    # save predictions
    save_predictions(prediction, test_ids)

    # activation using custom threshold
    if model_.THRESHOLD is None:
        prediction = np.around(prediction.astype(float))
    elif model_.THRESHOLD > 0 and model_.THRESHOLD < 1:
        prediction = prediction.astype(float)
        prediction[prediction >= model_.THRESHOLD] = 1
        prediction[prediction < model_.THRESHOLD] = 0
    else:
        print(CRED + "Invalid threshold" + CEND)
        sys.exit()

    # evaluation on test set
    evaluation(y_true=y_test, y_pred=prediction, target_names=model_.TARGET_NAMES, dataset='test')

    end_time = time.time()

    total_execution_time = (end_time - start_time) / 60

    print("\nTotal Execution time: {}min".format(total_execution_time))


main()
