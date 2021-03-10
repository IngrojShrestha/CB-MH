import pickle
import numpy as np
import nltk
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.models import load_model
from deepexplain.tensorflow import DeepExplain
from CB_MH import stem_sentences, remove_stop_word_punctuation

OUTPUT_PATH = '/path/to/output/'


def get_attribution_score(input_text, label, model_):
    '''
        computes attribution score for provided text given the prediction model

        param input_text: user provided text
        type input_text: str

        param label: predicted label
        type label: list of prediction label for e.g., [1 0 1 1 1 1 1 0]

        param model_: prediction deep learning model
        type model_: keras Model class


        returns: feature (word) and its corresponding importance for prediction label
        rtype: list, list

    '''
    current_session = K.get_session()

    # DeepExplain context
    with DeepExplain(session=current_session) as de:
        input_tensor = model_.layers[1].input

        # embedding layer output vector
        embedding = model_.layers[1].output

        # preprocessing: removal of stop words and punctuations
        sequences = tokenizer.texts_to_sequences([stem_sentences(remove_stop_word_punctuation(input_text))])

        # pad the sequence to a max_seq_length
        x_interpret = pad_sequences(sequences, maxlen=model_.MAX_SEQ_LENGTH, padding='pre', truncating='pre')

        # perform lookup
        get_embedding_output = K.function([input_tensor], [embedding])
        embedding_out = get_embedding_output([x_interpret])[0]

        # attribution using integrated gradient
        attributions = de.explain('intgrad', model_.layers[-2].output * label,
                                  embedding, embedding_out)

        values = np.array((np.sum(attributions, -1)))

        words = []
        values_set = []
        index_word = {v: k for k, v in tokenizer.word_index.items()}  # map back

        for k in range(len(x_interpret[0])):
            word = index_word.get(x_interpret[0][k])
            words.append(word)
            values_set.append(values[0][k])

        return words, values_set


def normalize_score_score(clinical_note, words, scores):
    '''

    normalize the score (-inf, +inf) to [0,1], negative to [0,0.5] and positive to (0.5,1]

    param clinical_note: patient clinical visit note
    type clinical_note: str

    param words: features (word in clinical notes)
    type words: a list of str

    param scores: attribution score
    type scores: list

    returns: important features (words) along with corresponding normalized score
    rtype: dict

    '''
    imp_word_list = {}

    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    word_score_dict = dict(zip(words, normalized_scores))

    for token in nltk.word_tokenize(clinical_note):
        for ts in nltk.wordpunct_tokenize(token):
            if stem_sentences(ts.lower()) in word_score_dict.keys():
                imp_word_list[str(ts)] = word_score_dict.get(stem_sentences(ts.lower()))
            else:
                imp_word_list[str(ts)] = 0.0
    return imp_word_list


def get_attributionget_attribution(clinical_note, model_):
    '''
        obtain attribution score, and normalize it

        param clinical_note: patient clinical visit note
        type clinical_note: str

        param model_: prediction deep learning model
        type model_: keras Model class

        returns: prediction label, normalized feature importance (attribution) from provided clinical note
        rtype: str, dict

    '''
    # tokenization and padding
    sequences = tokenizer.texts_to_sequences([stem_sentences(remove_stop_word_punctuation(clinical_note))])
    data = pad_sequences(sequences, maxlen=model_.MAX_SEQ_LENGTH, padding='pre', truncating='pre')

    # model predictions
    y_pred = model_.predict(data)

    # activation using provided threshold (convert prediction score to 1/0)
    y_pred = y_pred.astype(float)
    y_pred[y_pred >= model_.THRESHOLD] = 1
    y_pred[y_pred < model_.THRESHOLD] = 0

    # obtain corresponding label names
    all_labels = mlb.inverse_transform(y_pred)

    # obtain attribution of features(words) in clinical note
    words, values = get_attribution_score(clinical_note, y_pred, model_)

    # normalize attribution scores
    word_score_dict = normalize_score_score(clinical_note=clinical_note, words=np.array(words),
                                            scores=np.array(values))

    predictions = ''
    for item, labels in zip([clinical_note], all_labels):
        predictions = ', '.join(labels)
    return predictions, word_score_dict


def main(filename, model_name):
    global mlb
    global tokenizer

    # load saved model
    model_ = load_model("{}.h5".format(model_name))

    with open(OUTPUT_PATH + model_name + '_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # fit and text_to_sequence (x_test: padded/truncated, X_test_original: raw test document)
    x_train, y_train, x_val, y_val, x_test, y_test, X_test_original, X_test_original_index, word2idx, mlb, _ = model_.get_encoded_dataset()

    # obtain corresponding label names
    all_labels = mlb.inverse_transform(y_test)

    # save feature importance for a provided clinical note
    with open('post-hoc-results_{}.csv'.format(filename), 'w') as f:
        f.write(
            "index" + "\t" + "notes" + "\t" + "gold_standard_label" + "\t" + "prediction_label" + "\t" + "feature_importance" + "\n")
        for doc_id, note, labels in zip(X_test_original_index, X_test_original, all_labels):
            prediction, word_score_dict = get_attribution(note, model_)
            out = str(doc_id) + "\t" + note + "\t" + str(labels) + "\t" + prediction + "\t" + str(word_score_dict)
            f.write(out + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fetch filename to save result')
    parser.add_argument('-f', help="filename to store posthoc analysis results", dest='filename', type=str)
    parser.add_argument('-f', help="model name", dest='model_name', type=str)
    args = parser.parse_args()

    main(args.filename, args.model_name)
