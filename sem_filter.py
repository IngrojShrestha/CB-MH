import pandas as pd
import ast
from pymetamap import MetaMap

# https://metamap.nlm.nih.gov/Installation.shtml
PATH_TO_METAMAP = '/UMLS/MetaMap/public_mm/bin/metamap18'

mm = MetaMap.get_instance(PATH_TO_METAMAP)

# https://metamap.nlm.nih.gov/SemanticTypesAndGroups.shtml
semantic_types = pd.read_csv("SemanticTypes_2018AB.csv", header=None, sep='|')
semantic_types.columns = ['Abbreviation', 'UniqueIdentifier', 'SemanticType']


def get_concept(sentence):
    '''
        extracts UMLS concepts for the given input text

        param sentence: user provided text
        type sentence: str

        returns: UMLS concept and UMLS semantic type abbreviation
        rtype: str, list
    '''
    max = 0
    index = 0

    # extract UMLS concept = [index,mm,score,preferred_name,cui,semtypes,trigger,location,pos_info,tree_codes]
    concepts, error = mm.extract_concepts([sentence])

    if concepts:
        for index_, concept in enumerate(concepts):

            # score representing relevancy of UMLS concept
            score = float(concept[2])

            # extract the concept with highest score
            if score > max:
                max = score
                index = index_

        # UMLS concept preferred name
        concept = concepts[index][3]

        # semantic type abbreviation
        sem_type_abbreviation = concepts[index][5]

        return concept, sem_type_abbreviation
    else:
        return None, None


def filter_features(df):
    '''
        filters the features with attribution score at least 0.51

        param df: dataframe with clinical notes, labels and important feature along with its importance
        type df: dataframe

        returns: features with attribution score at least 0.51
        rtype: list [document_id, word, attribution_score]

    '''
    important_features = []
    for index, row in df.iterrows():
        doc_id = row['index']
        feature_importance = row['feature_importance']

        # convert string representation of dictionary to a dictionary
        feature_importance = ast.literal_eval(feature_importance)

        # filters feature with importance at least 0.51
        feature_importance = {k: v for k, v in feature_importance.items() if v >= 0.51}

        for k, v in feature_importance.items():
            important_features.append([doc_id, k, v])

    return important_features


def filter_sem_type(important_features):
    '''
        filters features that are related to mental illnesses

        param df: dataframe with clinical notes, labels and important feature along with its importance
        type df: dataframe

        returns: a list of important features (attributions) from a given text
        rtype: list

    '''
    attributions_words = []
    semantic_types = ['Mental or Behavioral Dysfunction', 'Sign or Symptom', 'Disease or Syndrome', 'Mental Process',
                      'Pharmacologic Substance']

    # filter based on semantic type
    for index, features in enumerate(important_features):
        doc_id = features[0]
        feature = features[1]
        score = features[2]
        concept, sem_type = get_concept(feature)
        if concept is not None or sem_type is not None:

            # pick the first semantic type from a list of semantic types
            sem_type_abbreviation = sem_type.strip('][').split(',')[0]

            # extract corresponding semantic type name for the provided semantic-type-abbreviation
            sem_type = semantic_types[semantic_types['Abbreviation'] == sem_type_abbreviation]['SemanticType'].iloc[0]

            # filter mental illness related semantic types
            if sem_type in semantic_types:
                attributions_words.append([doc_id, feature, score, sem_type])
                print("Index: {} {} : {}".format(index, feature, sem_type))
            else:
                print("Index: {} {} : {}".format(index, feature, 'Not relevant semantic type'))
    return attributions_words
