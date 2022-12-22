import re
import json
import statistics
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import spacy
from detoxify import Detoxify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


# Function used to load the raw comment list. TODO
def load_json_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as json_file:
        transformed_json_file = json.load(json_file)
    return transformed_json_file


# Function used to preprocess the comments so they are ready to be used by the neural networks used later.
def preprocess_comments(comment_list, save_path):
    # Acronyms dictionary
    dict_acronyms = {'afk': 'away from keyboard',
                     'brb': 'be right back',
                     'btw': 'by the way',
                     'idk': 'i don\'t know',
                     'imo': 'in my opinion',
                     'irl': 'in real life',
                     'lmk': 'let me know',
                     'lol': 'laughing out loud',
                     'l.o.l.': 'laughing out loud',
                     'l.o.l': 'laughing out loud',
                     'lmao': 'laughing my ass off',
                     'lmfao': 'laughing my freaking ass off',
                     'ofc': 'of course',
                     'omg': 'oh my god',
                     'o.m.g.': 'oh my god',
                     'o.m.g': 'oh my god',
                     'omfg': 'oh my fucking god',
                     'rofl': 'rolling on the floor laughing',
                     'smh': 'shaking my head',
                     'ttyl': 'talk to you later',
                     'yolo': 'you only live once',
                     'wth': 'what the heck',
                     'asap': 'as soon as possible',
                     'diy': 'do it yourself',
                     'faq': 'frequently asked questions',
                     'tba': 'to be announced',
                     'aka': 'also known as',
                     'fyi': 'for your information',
                     'bf': 'boyfriend',
                     'gf': 'girlfriend',
                     'msg': 'message',
                     'tfw': 'that feeling when',
                     'mfw': 'my face when',
                     'jk': 'just kidding',
                     'idc': 'i don’t care',
                     'ily': 'i love you',
                     'imu': 'i miss you',
                     'pov': 'point of view',
                     'tbh': 'to be honest',
                     'ftw': 'for the win',
                     'wtf': 'what the fuck',
                     'dm': 'direct message',
                     'til': 'today i learned',
                     'jsyk': 'just so you know',
                     'nsfw': 'not safe for work',
                     'nsfl': 'not safe for life',
                     'sfw': 'safe for work',
                     'oc': 'original content',
                     'op': 'original poster',
                     'gl': 'good luck',
                     'gratz': 'congratulations',
                     'bfd': 'big fucking deal',
                     'ffs': 'for fuck\'s sake',
                     'stfu': 'shut the fuck up',
                     'gtfo': 'get the fuck out',
                     'idgaf': 'i don\'t give a fuck',
                     'fml': 'fuck my life',
                     'mf': 'motherfucker',
                     'tmi': 'too much information',
                     'roflmao': 'rolling on the floor laughing my ass off'}

    dict_preprocessed_comments = {}
    for i, line in enumerate(comment_list):
        if i % 10000 == 0:
            print(i)
        l = line.split('\t')
        comment_id = l[17]  # Comment id
        comment = l[19]    # Comment

        comment = re.sub(r'(?:https?:\/\/|www[.])\S+', 'URL', comment)  # Replace URLs by "URL"
        comment = re.sub(r'[\w\.-]+@[\w\.-]+[.][\w\.-]+', 'EMAIL', comment)  # Replace email addresses by "EMAIL"

        # Replace acronyms by their long version
        for word in dict_acronyms.keys():
            comment = re.sub(rf'\b{word}\b', dict_acronyms[word], comment, flags=re.I)

        dict_preprocessed_comments[comment_id] = comment

    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(dict_preprocessed_comments, json_file)


# Function used to obtain the Sarcasm score feature.
def sarcasm_analysis_over_time(load_path, save_path):
    print('Loading dictionary of preprocessed comments')
    dict_preprocessed_comments = load_json_file(load_path)
    print('Done loading dictionary of preprocessed comments')

    print('Loading model')
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-irony")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-irony")
    model_pipeline = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True, truncation=True, padding=True, max_length=512, device=0)
    print('Done loading model')

    dict_scores_sarcasm = {}
    for i, comment_id in enumerate(dict_preprocessed_comments.keys()):
        if i % 100 == 0:
            print(i)

        comment_text = dict_preprocessed_comments[comment_id]
        result = model_pipeline(comment_text)
        score_sarcasm = result[0][1]['score']

        dict_scores_sarcasm[comment_id] = score_sarcasm

    with open(save_path + "\\dict_score_sarcasm.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_scores_sarcasm, json_file)


# Function used to obtain the Toxicity score feature.
def toxicity_analysis_over_time(load_path, save_path):
    print('Loading dictionary of preprocessed comments')
    dict_preprocessed_comments = load_json_file(load_path)
    print('Done loading dictionary of preprocessed comments')

    print('Loading model pipeline')
    model_pipeline = Detoxify('multilingual', device='cuda')
    print('Done loading pipeline')

    dict_scores_toxicity = {}
    for i, comment_id in enumerate(dict_preprocessed_comments.keys()):
        if i % 100 == 0:
            print(i)

        comment_text = dict_preprocessed_comments[comment_id]
        result = model_pipeline.predict(comment_text)
        score_toxicity = result['toxicity']

        dict_scores_toxicity[comment_id] = score_toxicity.item()

    with open(save_path + "\\dict_score_toxicity.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_scores_toxicity, json_file)


# Function used to obtain the Sentiment score feature.
def sentiment_analysis_over_time(load_path, save_path):
    print('Loading dictionary of preprocessed comments')
    dict_preprocessed_comments = load_json_file(load_path)
    print('Done loading dictionary of preprocessed comments')

    print('Loading sentiment model')
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model_pipeline = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True, truncation=True, padding=True, max_length=512, device=0)
    print('Done loading sentiment model')

    dict_scores_sarcasm = {}
    for i, comment_id in enumerate(dict_preprocessed_comments.keys()):
        if i % 100 == 0:
            print(i)

        comment_text = dict_preprocessed_comments[comment_id]
        result = model_pipeline(comment_text)
        score_sentiment_pos = result[0][1]['score']

        dict_scores_sarcasm[comment_id] = score_sentiment_pos

    with open(save_path + "\\dict_score_sentiment.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_scores_sarcasm, json_file)


# Function used to obtain the different emotion score features.
def emotion_analysis_over_time(load_path, save_path):
    print('Loading dictionary of preprocessed comments')
    dict_preprocessed_comments = load_json_file(load_path)
    print('Done loading dictionary of preprocessed comments')

    print('Loading sentiment model')
    tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    model_pipeline = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True, truncation=True, padding=True, max_length=512, device=0)
    print('Done loading sentiment model')

    dict_scores_sadness = {}
    dict_scores_joy = {}
    dict_scores_love = {}
    dict_scores_anger = {}
    dict_scores_fear = {}
    dict_scores_surprise = {}
    for i, comment_id in enumerate(dict_preprocessed_comments.keys()):
        if i % 100 == 0:
            print(i)

        comment_text = dict_preprocessed_comments[comment_id]
        result = model_pipeline(comment_text)
        dict_scores_sadness[comment_id] = result[0][0]['score']
        dict_scores_joy[comment_id] = result[0][1]['score']
        dict_scores_love[comment_id] = result[0][2]['score']
        dict_scores_anger[comment_id] = result[0][3]['score']
        dict_scores_fear[comment_id] = result[0][4]['score']
        dict_scores_surprise[comment_id] = result[0][5]['score']

    with open(save_path + "\\dict_score_sadness.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_scores_sadness, json_file)

    with open(save_path + "\\dict_score_joy.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_scores_joy, json_file)

    with open(save_path + "\\dict_score_love.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_scores_love, json_file)

    with open(save_path + "\\dict_score_anger.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_scores_anger, json_file)

    with open(save_path + "\\dict_score_fear.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_scores_fear, json_file)

    with open(save_path + "\\dict_score_surprise.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_scores_surprise, json_file)


# Function that combines the features extracted in the previous functions in a single dictionary.
# Also extracts the remaining features. Outputs a final dictionary containing all the features.
def get_full_comment_features(comment_list, dict_feature_load_path, save_path):
    # List of proper names to ignore
    list_known_names = ['justin trudeau', 'trudeau', 'joe biden', 'biden', 'donald trump',
                        'trump', 'barack obama', 'obama', 'george bush', 'george w bush',
                        'george w. bush', 'bush', 'w bush', 'w. bush', 'bill clinton',
                        'clinton', 'hillary clinton', 'bernie sanders', 'sanders',
                        'doug ford', 'ford', 'françois legault', 'legault', 'francois legault',
                        'tim houston', 'houston', 'blaine higgs', 'higgs', 'heather stefanson',
                        'stefanson', 'john horgan', 'horgan', 'dennis king', 'king',
                        'scott moe', 'moe', 'jason kenney', 'kenney', 'andrew furey',
                        'furey', 'sandy silver', 'silver', 'joe savikataaq', 'savikataaq',
                        'kathleen wynne', 'wynne', 'philippe couillard', 'couillard',
                        'iain rankin', 'rankin', 'brian gallant', 'gallant',
                        'kelvin goertzen', 'goertzen', 'christy clark', 'clark',
                        'wade maclauchlan', 'maclauchlan', 'brad wall', 'wall',
                        'rachel notley', 'notley', 'dwight ball', 'ball', 'darrell pasloski',
                        'pasloski', 'paul quassa', 'quassa', 'elon musk', 'musk',
                        'angela merkel', 'merkel', 'emmanuel macron', 'macron', 'emanuel macron',
                        'xi jinping', 'jinping', 'theresa tam', 'tam', 'anthony fauci', 'fauci',
                        'bill gates', 'gates', 'mark zuckerberg', 'zuckerberg',
                        'jeff bezos', 'bezos']
    list_first_person_singular_pronouns = ['i', 'me', 'my', 'mine', 'myself']
    list_first_person_plural_pronouns = ['we', 'us', 'our', 'ours', 'ourselves']
    list_second_person_pronouns = ['you', 'your', 'yours', 'yourself', 'yourselves']
    list_third_person_singular_pronouns = ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself']
    list_third_person_plural_pronouns = ['they', 'them', 'their', 'theirs', 'themselves']
    list_politeness_gratitude = ['thank you', 'thanks', 'you\'re welcome', 'you are welcome', 'my pleasure', 'hi',
                                 'hello', 'good morning', 'good afternoon', 'good evening', 'good day', 'take care']

    # Load the dictionary for the different features extracted in the previous functions.
    print('Loading feature dicts')
    dict_score_toxicity = load_json_file(dict_feature_load_path + "\\dict_score_toxicity.json")
    dict_score_sarcasm = load_json_file(dict_feature_load_path + "\\dict_score_sarcasm.json")
    dict_score_sentiment = load_json_file(dict_feature_load_path + "\\dict_score_sentiment.json")
    dict_score_anger = load_json_file(dict_feature_load_path + "\\dict_score_anger.json")
    dict_score_fear = load_json_file(dict_feature_load_path + "dict_score_fear.json")
    dict_score_joy = load_json_file(dict_feature_load_path + "dict_score_joy.json")
    dict_score_love = load_json_file(dict_feature_load_path + "dict_score_love.json")
    dict_score_sadness = load_json_file(dict_feature_load_path + "dict_score_sadness.json")
    dict_score_surprise = load_json_file(dict_feature_load_path + "dict_score_surprise.json")
    print('Done loading feature dicts')

    # Combine the previously loaded dictionaries into one.
    dict_score_overall = {}
    print('Combining feature dicts')
    for i, comment_id_key in enumerate(dict_score_toxicity.keys()):
        dict_score_overall[comment_id_key] = {'toxicity_score': dict_score_toxicity[comment_id_key],
                                          'sarcasm_score': dict_score_sarcasm[comment_id_key],
                                          'sentiment_score': dict_score_sentiment[comment_id_key],
                                          'anger_score': dict_score_anger[comment_id_key],
                                          'fear_score': dict_score_fear[comment_id_key],
                                          'joy_score': dict_score_joy[comment_id_key],
                                          'love_score': dict_score_love[comment_id_key],
                                          'sadness_score': dict_score_sadness[comment_id_key],
                                          'surprise_score': dict_score_surprise[comment_id_key]}
    print('Done combining score dicts')

    # Loop which creates a dictionary for every article (conversation).
    # Each dictionary in turn contains one dictionary per comment,
    # which itself contains all the 23 features of that comment.
    ner = spacy.load('en_core_web_sm')  # Loading of the spacy model for named entities recognition
    dict_article = {}
    for i, comment_content in enumerate(comment_list):
        if i % 100 == 0:
            print(i)

        comment_items = comment_content.split('\t')
        comment_text = comment_items[19]    # Comment text
        article_id = comment_items[0]   # Article (conversation) ID
        comment_id = comment_items[17]  # Comment ID
        date_time = comment_items[16]   # Date and time at which the comment was written
        nbr_likes = comment_items[24]   # Number of likes

        # Building a dictionary containing all the features for the comment
        dict_comments = {'comment_id': comment_id,
                         'comment_text': comment_text,
                         'toxicity_score': dict_score_overall[comment_id]['toxicity_score'],
                         'sarcasm_score': dict_score_overall[comment_id]['sarcasm_score'],
                         'sentiment_score': dict_score_overall[comment_id]['sentiment_score'],
                         'anger_score': dict_score_overall[comment_id]['anger_score'],
                         'fear_score': dict_score_overall[comment_id]['fear_score'],
                         'joy_score': dict_score_overall[comment_id]['joy_score'],
                         'love_score': dict_score_overall[comment_id]['love_score'],
                         'sadness_score': dict_score_overall[comment_id]['sadness_score'],
                         'surprise_score': dict_score_overall[comment_id]['surprise_score'],
                         'comment_length_in_words': len(comment_text.split(' ')),
                         'contains_url': False,
                         'contains_email': False,
                         'contains_hashtag': False,
                         'image_only': False,
                         'nbr_likes': nbr_likes,
                         'nbr_first_per_sing_pronouns': 0,
                         'nbr_first_per_plur_pronouns': 0,
                         'nbr_second_per_pronouns': 0,
                         'nbr_third_per_sing_pronouns': 0,
                         'nbr_third_per_plur_pronouns': 0,
                         'nbr_politeness_gratitude': 0,
                         'date_time': date_time,
                         'elapsed_time': None,
                         'starts_w_name': None}

        # Check if the comment starts by a reply or reference to another user (check if it starts with a proper noun).
        if re.search(r'^\s*@', comment_text):
            dict_comments['starts_w_name'] = True
        else:
            tagged_text = ner(re.sub(r'^\s*[\"\']*\s*', '', comment_text))
            if tagged_text.ents and tagged_text.ents[0].label_ == 'PERSON' and tagged_text.ents[0].start_char == 0 and tagged_text.ents[0].text.lower() not in list_known_names:
                dict_comments['starts_w_name'] = True
            else:
                dict_comments['starts_w_name'] = False

        # Check if the comment contains a URL
        if re.search(r'(?:https?:\/\/|www[.])\S+', comment_text):
            dict_comments['contains_url'] = True

        # Check if the comment contains an email address
        if re.search(r'[\w\.-]+@[\w\.-]+[.][\w\.-]+', comment_text):
            dict_comments['contains_email'] = True

        # Check if the comment contains a hashtag
        if re.search(r'(^|[\'\"\(\s,\.:;])#(?![0-9_]+\b)([a-zA-Z0-9_]+)(\b|\r)', comment_text):
            dict_comments['contains_hashtag'] = True

        # Check if the comment is only made up of an image or GIF
        if len(comment_text) == 0:
            dict_comments['image_only'] = True

        # Count the number of first person singular pronouns
        for pronoun in list_first_person_singular_pronouns:
            dict_comments['nbr_first_per_sing_pronouns'] += len(re.findall(rf'\b{pronoun}\b', comment_text, flags=re.I))

        # Count the number of first person plural pronouns
        for pronoun in list_first_person_plural_pronouns:
            dict_comments['nbr_first_per_plur_pronouns'] += len(re.findall(rf'\b{pronoun}\b', comment_text, flags=re.I))

        # Count the number of second person pronouns (both singular and plural)
        for pronoun in list_second_person_pronouns:
            dict_comments['nbr_second_per_pronouns'] += len(re.findall(rf'\b{pronoun}\b', comment_text, flags=re.I))

        # Count the number of third person singular pronouns
        for pronoun in list_third_person_singular_pronouns:
            dict_comments['nbr_third_per_sing_pronouns'] += len(re.findall(rf'\b{pronoun}\b', comment_text, flags=re.I))

        # Count the number of third person plural pronouns
        for pronoun in list_third_person_plural_pronouns:
            dict_comments['nbr_third_per_plur_pronouns'] += len(re.findall(rf'\b{pronoun}\b', comment_text, flags=re.I))

        # Count the number of politeness and gratitude related terms
        for expression in list_politeness_gratitude:
            dict_comments['nbr_politeness_gratitude'] += len(re.findall(rf'\b{expression}\b', comment_text, flags=re.I))

        if article_id not in dict_article.keys():
            dict_article[article_id] = []
        dict_article[article_id].append(dict_comments)

    # Computing the elapsed time between consecutive comments of an article (conversation)
    for i, article_id_key in enumerate(dict_article.keys()):
        if i % 100 == 0:
            print(i)
        dict_article[article_id_key] = [a for a in sorted(dict_article[article_id_key], key=lambda x: datetime.strptime(x['date_time'], '%Y-%m-%d') if re.match(r'^\d{4}\-\d{2}\-\d{2}$', x['date_time']) else datetime.strptime(x['date_time'], '%Y-%m-%d %H:%M:%S'))]

        previous_date_time = datetime.strptime(dict_article[article_id_key][0]['date_time'], '%Y-%m-%d') if re.match(r'^\d{4}\-\d{2}\-\d{2}$', dict_article[article_id_key][0]['date_time']) else datetime.strptime(dict_article[article_id_key][0]['date_time'], '%Y-%m-%d %H:%M:%S')
        for comment in dict_article[article_id_key]:
            current_date_time = datetime.strptime(comment['date_time'], '%Y-%m-%d') if re.match(r'^\d{4}\-\d{2}\-\d{2}$', comment['date_time']) else datetime.strptime(comment['date_time'], '%Y-%m-%d %H:%M:%S')
            time_diff = current_date_time - previous_date_time
            comment['elapsed_time'] = time_diff.days * 86400 + time_diff.seconds
            previous_date_time = current_date_time

    # Save the final dictionary containing all the features for all the comments.
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(dict_article, json_file)


if __name__ == '__main__':
    comment_list_load_path = ""     # # Replace with path name to the "comment_list.json" file
    comment_list = load_json_file(comment_list_load_path)

    preprocessed_comments_save_path = ""    # Replace by the path to the "dict_preprocessed_comments.json" file
    preprocess_comments(comment_list, save_path=preprocessed_comments_save_path)

    feature_save_path = ""      # Replace by the path to the location where the feature dictionaries will be saved
    sarcasm_analysis_over_time(load_path=preprocessed_comments_save_path, save_path=feature_save_path)
    toxicity_analysis_over_time(load_path=preprocessed_comments_save_path, save_path=feature_save_path)
    sentiment_analysis_over_time(load_path=preprocessed_comments_save_path, save_path=feature_save_path)
    emotion_analysis_over_time(load_path=preprocessed_comments_save_path, save_path=feature_save_path)
    # The resulting dictionaries containing the value for these features are in the "dict_score_anger.json",
    # "dict_score_fear.json", "dict_score_joy.json", "dict_score_love.json", "dict_score_sadness.json",
    # "dict_score_surprise.json", "dict_score_sarcasm.json", "dict_score_sentiment.json" and "dict_score_toxicity.json"
    # files

    final_feature_dictionary_save_path = ""     # Replace by the path to the final dictionary containing all the features
    get_full_comment_features(comment_list=comment_list, dict_feature_load_path=feature_save_path, save_path=final_feature_dictionary_save_path)
    # The full final dictionary containing all the features for each comment is in the "final_feature_dict.json" file