import random
import statistics
import math
from scipy.stats import kendalltau
from pomegranate import *
from sklearn.model_selection import KFold
from sklearn.cluster import SpectralCoclustering
import numpy as np
import json
import matplotlib.pyplot as plt
import re
import seaborn as sns
from datetime import datetime


# Function which takes an HMM as input and outputs its emission probabilities
def print_model_emission_distributions_pomegranate(model, nbr_states, feature_list):
    for i in range(nbr_states):
        print(f'State #{i}')
        for j in range(len(feature_list)):
            print(f'\t{feature_list[j]}\t{model.states[i].distribution.parameters[0][j].parameters}')


# Function used to save a trained model as a JSON file (for later use)
def save_pomegranate_model(model, file_path):
    json_model = model.to_json()
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_model, json_file)


# Function used to load an HMM previously saved in JSON form
def load_pomegranate_model(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        json_model = json.load(json_file)
    model = from_json(json_model)
    return model


# Function used to obtain the final dictionary containing all the features for all the comments.
def get_feature_dict(load_path):
    print('Importing full dict')
    with open(load_path, 'r', encoding='utf-8') as json_file:
        feature_dict = json.load(json_file)
    print('Done importing full dict')
    return feature_dict


# Function which splits the dataset (the feature dictionary) in a train and test set, according to the datetime
# at which the comments were written (see paper).
def split_train_test_w_date(feature_dict, train_ratio):
    print('Starting split process')
    list_articles_id_ordered_by_dates = sorted(list(feature_dict.keys()), key=lambda x: datetime.strptime(feature_dict[x][-1]['date_time'], '%Y-%m-%d') if re.match(r'^\d{4}\-\d{2}\-\d{2}$', feature_dict[x][-1]['date_time']) else datetime.strptime(feature_dict[x][-1]['date_time'], '%Y-%m-%d %H:%M:%S'))
    train_stop_index = math.floor(train_ratio * len(list_articles_id_ordered_by_dates))
    train_dict = {art_id: feature_dict[art_id] for art_id in list_articles_id_ordered_by_dates[:train_stop_index]}
    test_dict = {art_id: feature_dict[art_id] for art_id in list_articles_id_ordered_by_dates[train_stop_index:]}
    print('Done with split process')

    return train_dict, test_dict


# Function to analyze one of the 23 features. The outliers are removed and a histogram is shown, as well as a
# smoothed-out version of that histogram. This function is used to determine the threshold for the binarization step.
def analyze_features(load_path, feature_to_analyze):
    feature_dict = get_feature_dict(load_path)

    list_feature = []
    for art_id in feature_dict.keys():
        for comment in feature_dict[art_id]:
            list_feature.append(comment[feature_to_analyze])

    print(f'Max feature value : {max(list_feature)}')
    print(f'Min feature value : {min(list_feature)}')
    print(f'Number of comments : {len(list_feature)}')

    # Remove outliers
    q1, med, q3 = statistics.quantiles(list_feature, n=4)
    iqr = q3 - q1
    lower_bound = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    print(f'Upper bound : {upper_bound}')
    print(f'Lower bound : {lower_bound}')

    # Other method of removing outliers. Uncomment if needed.
    """mean = statistics.mean(list_feature)
    stdev = statistics.stdev(list_feature)
    upper_bound = mean + 3*stdev
    lower_bound = mean - 3*stdev
    print(f'Upper bound : {upper_bound}')
    print(f'Lower bound : {lower_bound}')"""

    list_feature = [elem for elem in list_feature if lower_bound <= elem <= upper_bound]
    print(f'Max feature value : {max(list_feature)}')
    print(f'Min feature value : {min(list_feature)}')
    print(f'Number of comments : {len(list_feature)}')

    # Plot a normal histogram and a smoothed-out version.
    plt.hist(list_feature, bins=36)
    plt.show()
    sns.kdeplot(list_feature)
    plt.show()


# Function which transforms the features to a binary version.
def feature_to_binary(feature_dict):
    for i, art_id in enumerate(feature_dict.keys()):
        for comment in feature_dict[art_id]:

            # Transformation of continuous features
            comment['toxicity_score'] = 0 if comment['toxicity_score'] < 0.5 else 1
            comment['sarcasm_score'] = 0 if comment['sarcasm_score'] < 0.55 else 1
            comment['sentiment_score'] = 0 if comment['sentiment_score'] < 0.5 else 1
            comment['anger_score'] = 0 if comment['anger_score'] < 0.55 else 1
            comment['fear_score'] = 0 if comment['fear_score'] < 0.5 else 1
            comment['joy_score'] = 0 if comment['joy_score'] < 0.5 else 1
            comment['love_score'] = 0 if comment['love_score'] < 0.5 else 1
            comment['sadness_score'] = 0 if comment['sadness_score'] < 0.5 else 1
            comment['surprise_score'] = 0 if comment['surprise_score'] < 0.5 else 1

            comment['elapsed_time'] = 0 if comment['elapsed_time'] < 650 else 1

            # Features that are already binary
            comment['contains_url'] = 1 if comment['contains_url'] else 0
            comment['contains_email'] = 1 if comment['contains_email'] else 0
            comment['contains_hashtag'] = 1 if comment['contains_hashtag'] else 0
            comment['image_only'] = 1 if comment['image_only'] else 0
            comment['starts_w_name'] = 1 if comment['starts_w_name'] else 0

            # Transformation of discrete (non-binary) features
            comment['nbr_likes'] = 0 if comment['nbr_likes'] == '' or float(comment['nbr_likes']) < 1 else 1
            comment['nbr_first_per_sing_pronouns'] = 0 if comment['nbr_first_per_sing_pronouns'] < 1 else 1
            comment['nbr_first_per_plur_pronouns'] = 0 if comment['nbr_first_per_plur_pronouns'] < 1 else 1
            comment['nbr_second_per_pronouns'] = 0 if comment['nbr_second_per_pronouns'] < 1 else 1
            comment['nbr_third_per_sing_pronouns'] = 0 if comment['nbr_third_per_sing_pronouns'] < 1 else 1
            comment['nbr_third_per_plur_pronouns'] = 0 if comment['nbr_third_per_plur_pronouns'] < 1 else 1
            comment['nbr_politeness_gratitude'] = 0 if comment['nbr_politeness_gratitude'] < 1 else 1

            if comment['comment_length_in_words'] <= 5:
                comment['comment_length_in_words'] = 0
            elif comment['comment_length_in_words'] <= 62:
                comment['comment_length_in_words'] = 1
            else:
                comment['comment_length_in_words'] = 2


# Function which takes a trained HMM and associates each comment to a hidden state. This function also
# displays graphs (histograms) to visualize the distribution of feature values in the different states.
def store_comments_by_state(trained_model, dict_usable_data, feature_dict, feature_list, nbr_states):
    # Each comment is associated to a hidden state
    dict_states = {i: [] for i in range(nbr_states)}
    for id_article in dict_usable_data.keys():
        predicted_states = trained_model.predict(dict_usable_data[id_article])
        for i, state in enumerate(predicted_states):
            dict_states[state].append(feature_dict[id_article][i])

    nbr_dim_plots = math.ceil(math.sqrt(len(feature_list)))
    for state in dict_states.keys():
        figs, axs = plt.subplots(nbr_dim_plots, nbr_dim_plots)
        for i, feature in enumerate(feature_list):
            list_feature_values = []
            for comment in dict_states[state]:
                list_feature_values.append(comment[feature])
            axs[math.floor(i/nbr_dim_plots), i % nbr_dim_plots].set_title(f'{feature}')
            axs[math.floor(i/nbr_dim_plots), i % nbr_dim_plots].hist(list_feature_values, bins=5)

        plt.suptitle(f'State #{state}')
        plt.tight_layout()
        plt.show()

    figs, axs = plt.subplots(nbr_dim_plots, nbr_dim_plots)
    for i, feature in enumerate(feature_list):
        list_feature_values = []
        for art_id in dict_usable_data.keys():
            for comment in dict_usable_data[art_id]:
                list_feature_values.append(comment[i])
        axs[math.floor(i/nbr_dim_plots), i % nbr_dim_plots].set_title(f'{feature}')
        axs[math.floor(i/nbr_dim_plots), i % nbr_dim_plots].hist(list_feature_values, bins=5)
    plt.suptitle('Overall')
    plt.tight_layout()
    plt.show()


# Function that extracts the features from the feature dictionary and prepares them so that they are
# ready for usage by the HMMs (Pomegranate library)
def extract_features_from_dict(feature_dict, feature_list, min_comment_length, max_comment_length):
    dict_usable_arrays = {}
    sequences = []
    seq_lengths = []
    num_comments = 0
    temp = 0
    print('Extracting comment features')
    for i, key in enumerate(feature_dict.keys()):
        if min_comment_length <= len(feature_dict[key]) <= max_comment_length:
            temp += 1
            article_seq = []
            for j, comment in enumerate(feature_dict[key]):
                num_comments += 1
                extracted_features = [comment[feature] for feature in feature_list]
                article_seq.append(extracted_features)
            sequences.append(article_seq)
            dict_usable_arrays[key] = article_seq
            seq_lengths.append(len(article_seq))
    print('Done extracting comment features')
    print(f'# of comments : {num_comments}')
    return sequences, seq_lengths, dict_usable_arrays


# Function to initialize the transition, emission and start probabilities of an HMM.
# If random_emission_prob == True, then the emission probabilities are randomly initialized.
# Otherwise, biclustering is used.
def initialize_hmm(nbr_states, feature_list, nbr_val_per_feature, train_sequences, random_emission_prob=True):
    print('Creating model and adding states and transitions')
    states = []
    model = HiddenMarkovModel()
    if random_emission_prob:
        for i in range(nbr_states):
            distributions = []
            for j in range(len(feature_list)):
                prob = np.random.default_rng().dirichlet(np.ones(nbr_val_per_feature[j]), 1)[0]
                distributions.append(DiscreteDistribution({k: prob[k] for k in range(nbr_val_per_feature[j])}))
            states.append(State(IndependentComponentsDistribution(distributions), name=f'state{i}'))
    else:
        dict_feature_per_clusters = biclustering(train_sequences, nbr_val_per_feature, nbr_states)
        for i in range(nbr_states):
            distributions = []
            for j in range(len(feature_list)):
                distributions.append(DiscreteDistribution({k: dict_feature_per_clusters[i][j][k] for k in range(nbr_val_per_feature[j])}))
            states.append(State(IndependentComponentsDistribution(distributions), name=f'state{i}'))
    model.add_states(states)

    prob_start = np.random.default_rng().dirichlet(np.ones(nbr_states), 1)[0]
    for i in range(nbr_states):
        model.add_transition(model.start, states[i], prob_start[i])
        prob = np.random.default_rng().dirichlet(np.ones(nbr_states), 1)[0]
        for j in range(nbr_states):
            model.add_transition(states[i], states[j], prob[j])

    model.bake()
    return model


# Function that performs the biclustering. Returns the necessary information to be used for the initialization
# step (in the initialize_hmm function).
def biclustering(train_sequences, nbr_values_per_feature, nbr_states):
    # Place every comment in a single array, regardless of the article (conversation) it comes from.
    data = []
    for article in train_sequences:
        for comment in article:
            data.append(comment)

    # Remove the comments that have all 0 features. Otherwise, the algorithm does not work.
    data = [row for row in data if sum(row) > 0]
    data = np.array(data)

    # The clustering itself. We choose as many clusters as the number of states.
    model = SpectralCoclustering(n_clusters=nbr_states)
    model.fit(data)

    row_labels = model.row_labels_  # Extract the array that contains numbers indicating to which cluster every comment belongs.
    column_labels = model.column_labels_    # Extract the array that contains the numbers indicating to which cluster every feature belongs.

    # Creating an empty dictionary, where the keys are a number ranging from 0 to n-1, n being the number of row cluster
    # and the values are empty arrays. For every comment, we check which cluster it belongs to, then we add that comment
    # to the array of the corresponding cluster number.
    dict_row_clusters = {v: [] for v in set(row_labels)}
    for i, row_num in enumerate(row_labels):
        dict_row_clusters[row_num].append(data[i])

    # Associate each feature to its corresponding row cluster.
    dict_column_clusters = {v: [] for v in range(nbr_states)}
    for i, column_num in enumerate(column_labels):
        dict_column_clusters[column_num].append(i)

    # For every feature of every bicluster, count how many times each value of that feature appears in the dataset.
    # We can then turn these counts in proportions and use the results to initialize the emission probabilities.
    dict_feature_per_clusters = {v: [] for v in set(row_labels)}
    for cluster_num in dict_row_clusters.keys():
        for i, val_per_feature in enumerate(nbr_values_per_feature):
            if i in dict_column_clusters[cluster_num]:
                dict_feature_per_clusters[cluster_num].append({v: 0 for v in range(val_per_feature)})
                for comment in dict_row_clusters[cluster_num]:
                    dict_feature_per_clusters[cluster_num][i][comment[i]] += 1
            else:
                dict_feature_per_clusters[cluster_num].append({v: len(dict_row_clusters[cluster_num]) / val_per_feature for v in range(val_per_feature)})

    for cluster_num in dict_feature_per_clusters.keys():
        total = len(dict_row_clusters[cluster_num])
        for feature_dict in dict_feature_per_clusters[cluster_num]:
            for val in feature_dict.keys():
                if val == (len(feature_dict) - 1):
                    feature_dict[val] = 1 - sum([feature_dict[k] for k in feature_dict.keys() if k != (len(feature_dict) - 1)])
                else:
                    feature_dict[val] /= total
    return dict_feature_per_clusters


# Function that computes the Silhouette coefficient for a given biclustering.
# Used to determine the optimal number of states.
def silhouette_coefficient(load_path, nbr_states, percentage=1.0):
    # Importing and initializing the necessary data
    feature_dict = get_feature_dict(load_path)
    feature_to_binary(feature_dict)
    train_dict, test_dict = split_train_test_w_date(feature_dict, train_ratio=0.8)
    feature_list = ['toxicity_score', 'sarcasm_score', 'sentiment_score', 'anger_score', 'fear_score', 'joy_score',
                      'love_score', 'sadness_score', 'surprise_score', 'elapsed_time', 'contains_url', 'contains_email',
                      'contains_hashtag', 'image_only', 'starts_w_name', 'nbr_likes', 'nbr_first_per_sing_pronouns',
                      'nbr_first_per_plur_pronouns', 'nbr_second_per_pronouns', 'nbr_third_per_sing_pronouns',
                      'nbr_third_per_plur_pronouns', 'nbr_politeness_gratitude', 'comment_length_in_words']
    nbr_values_per_feature = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
    min_comment_length = 10
    max_comment_length = 200000000000000000
    train_sequences, train_seq_lengths, _ = extract_features_from_dict(train_dict, feature_list, min_comment_length=min_comment_length, max_comment_length=max_comment_length)

    # This part is exactly the same as in the biclustering function. It essentially performs the biclustering.
    # I chose to rewrite it here instead of calling the biclustering function, because some steps in the function
    # are not needed here. Also, the function returns a dictionary which wouldn't be useful here.
    print('Begin biclustering')
    data = []
    for article in train_sequences:
        for comment in article:
            data.append(comment)

    data = [row for row in data if sum(row) > 0]
    data = np.array(data)

    model = SpectralCoclustering(n_clusters=nbr_states)
    model.fit(data)

    row_labels = model.row_labels_  # Extract the array that contains numbers indicating to which cluster every comment belongs.
    column_labels = model.column_labels_  # Extract the array that contains the numbers indicating to which cluster every feature belongs.

    dict_row_clusters = {v: [] for v in set(row_labels)}
    for i, row_num in enumerate(row_labels):
        dict_row_clusters[row_num].append(data[i])

    dict_column_clusters = {v: [] for v in range(nbr_states)}
    for i, column_num in enumerate(column_labels):
        dict_column_clusters[column_num].append(i)
    print('End biclustering')

    # ONLY KEEP A CERTAIN PERCENTAGE OF THE DATA TO ACCELERATE THE COMPUTATIONS
    if percentage < 1:
        for row_num in range(nbr_states):
            num_comments = math.floor(percentage * len(dict_row_clusters[row_num]))
            dict_row_clusters[row_num] = random.sample(dict_row_clusters[row_num], num_comments)
    # END OF THE BICLUSTERING PROCEDURE

    # ACTUAL COMPUTATION OF THE SILHOUETTE INDEX

    # STARTING WITH THE A VALUES
    dict_a_values = {v: [] for v in range(nbr_states)}
    for j in range(nbr_states):
        dict_a_values[j] = np.array([0. for l in range(len(dict_row_clusters[j]))])
        feature_index = dict_column_clusters[j] if len(dict_column_clusters[j]) != 0 else [i for i in range(len(feature_list))]
        for i in range(len(dict_row_clusters[j])):
            for k in range(i + 1, len(dict_row_clusters[j])):
                dist = 1 - np.mean([1 if dict_row_clusters[j][i][feature] == dict_row_clusters[j][k][feature] else 0 for feature in feature_index])
                dict_a_values[j][i] += dist
                dict_a_values[j][k] += dist
        dict_a_values[j] = np.array([v / (len(dict_row_clusters[j]) - 1) for v in dict_a_values[j]])

    # THEN THE B VALUES
    dict_b_values = {v: [[0 for n in range(nbr_states - 1)] for l in range(len(dict_row_clusters[v]))] for v in range(nbr_states)}
    for j in range(nbr_states):
        feature_index_1 = dict_column_clusters[j] if len(dict_column_clusters[j]) != 0 else [m for m in range(len(feature_list))]
        for i in range(len(dict_row_clusters[j])):
            for h in range(j + 1, nbr_states):
                feature_index_2 = dict_column_clusters[h] if len(dict_column_clusters[h]) != 0 else [m for m in range(len(feature_list))]
                for k in range(len(dict_row_clusters[h])):
                    dist_2 = 1 - np.mean([1 if dict_row_clusters[j][i][feature] == dict_row_clusters[h][k][feature] else 0 for feature in feature_index_2])
                    dist_1 = 1 - np.mean([1 if dict_row_clusters[j][i][feature] == dict_row_clusters[h][k][feature] else 0 for feature in feature_index_1])
                    dict_b_values[j][i][h - 1] += dist_2
                    dict_b_values[h][k][j] += dist_1

    for j in range(nbr_states):
        for i in range(len(dict_row_clusters[j])):
            for h in range(nbr_states):
                if j < h:
                    dict_b_values[j][i][h - 1] /= len(dict_row_clusters[h])
                elif j > h:
                    dict_b_values[j][i][h] /= len(dict_row_clusters[h])

    for j in range(nbr_states):
        for i in range(len(dict_row_clusters[j])):
            dict_b_values[j][i] = min(dict_b_values[j][i])

    # COMPUTING THE S VALUES
    dict_s_values = {v: [] for v in range(nbr_states)}
    for j in range(nbr_states):
        for i in range(len(dict_row_clusters[j])):
            a = dict_a_values[j][i]
            b = dict_b_values[j][i]
            if max(a, b) == 0:
                s = 0
            else:
                s = (b - a) / max(a, b)
            dict_s_values[j].append(s)

    # COMPUTING THE Q VALUES PER CLUSTER
    dict_q_values = {v: 0 for v in range(nbr_states)}
    for j in range(nbr_states):
        for i in range(len(dict_row_clusters[j])):
            dict_q_values[j] += dict_s_values[j][i]
        dict_q_values[j] /= len(dict_row_clusters[j])
    print(dict_q_values)

    # COMPUTING THE FINAL Q VALUE FOR THE WHOLE CLUSTERING
    final_q = 0
    for j in range(nbr_states):
        final_q += dict_q_values[j]
    final_q /= nbr_states
    print(final_q)


# Function that computes the distance between two trained HMMs.
def distance_hmm(model_a, model_b, length):
    samples_a = model_a.sample(n=1, length=length)
    log_prob_model_a_sample_a = model_a.log_probability(samples_a[0])
    log_prob_model_b_sample_a = model_b.log_probability(samples_a[0])

    dist_a_b = (log_prob_model_a_sample_a - log_prob_model_b_sample_a) / length

    samples_b = model_b.sample(n=1, length=length)
    log_prob_model_a_sample_b = model_a.log_probability(samples_b[0])
    log_prob_model_b_sample_b = model_b.log_probability(samples_b[0])

    dist_b_a = (log_prob_model_b_sample_b - log_prob_model_a_sample_b) / length

    dist_final = (dist_a_b + dist_b_a) / 2
    return dist_final


# Function to train multiple HMMs and compute a distance matrix between them.
# Used to analyze the differences between HMMs trained with identical parameters.
def test_distance_measure(nbr_models, nbr_states, iter_list, random_emission_prob, length_list):
    dict_articles = get_feature_dict()
    feature_to_binary(dict_articles)
    train_dict, _ = split_train_test_w_date(dict_articles, train_ratio=0.8)
    feature_list = ['toxicity_score', 'sarcasm_score', 'sentiment_score', 'anger_score', 'fear_score', 'joy_score',
                      'love_score', 'sadness_score', 'surprise_score', 'elapsed_time', 'contains_url', 'contains_email',
                      'contains_hashtag', 'image_only', 'starts_w_name', 'nbr_likes', 'nbr_first_per_sing_pronouns',
                      'nbr_first_per_plur_pronouns', 'nbr_second_per_pronouns', 'nbr_third_per_sing_pronouns',
                      'nbr_third_per_plur_pronouns', 'nbr_politeness_gratitude', 'comment_length_in_words']
    nbr_values_per_feature = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
    min_comment_length = 10
    max_comment_length = 200000000000000000
    train_sequences, train_seq_lengths, _ = extract_features_from_dict(train_dict, feature_list, min_comment_length=min_comment_length, max_comment_length=max_comment_length)

    untrained_models = []
    for i in range(nbr_models):
        print(f'Initializing model #{i+1}')
        untrained_models.append(initialize_hmm(nbr_states, feature_list, nbr_values_per_feature, train_sequences, random_emission_prob=random_emission_prob))

    for nbr_iter in iter_list:
        print(f'Training models for {nbr_iter} iteration(s)')
        model_list = []
        for i in range(nbr_models):
            print(f'Training model #{i+1}')
            current_model = untrained_models[i].copy()
            if nbr_iter == 0:
                model_list.append(current_model)
            else:
                model_list.append(current_model.fit(sequences=train_sequences, max_iterations=nbr_iter, n_jobs=-1, verbose=True, inertia=0.1, stop_threshold=0.01))
            print_model_emission_distributions_pomegranate(current_model, nbr_states, feature_list)
            print(current_model.dense_transition_matrix())

        print(f'Results for {nbr_iter} iteration(s)')
        for length in length_list:
            distance_matrix = np.zeros((nbr_models, nbr_models))
            for i in range(nbr_models):
                for j in range(i, nbr_models):
                    if i != j:
                        dist_array = []
                        for k in range(30):
                            dist_array.append(distance_hmm(model_list[i], model_list[j], length))
                        dist_array = [val for val in dist_array if not (math.isnan(val) or math.isinf(val))]
                        if len(dist_array) == 0:
                            distance_matrix[i, j] = -1
                        else:
                            dist_mean = sum(dist_array) / len(dist_array)
                            distance_matrix[i, j] = dist_mean
            print(f'Length {length} : \n{distance_matrix}')


# Function which implements a grid search procedure to find the optimal hyperparameter values for the HMMs.
# Makes use of K-fold cross-validation.
def grid_search_with_cv(nbr_folds, train_sequences, test_sequences, feature_list, nbr_val_per_feature, nbr_states=9):
    # List of tested HMM (hyperparameters)
    hmm_parameters_list = []

    # Liste des valeurs de distance entre les HMMs
    # List containing the values of the distance between HMMs
    hmm_distance_list = []

    # Hyperparameter values to test
    max_iter_array = [i for i in range(2, 11)]
    inertia_array = [0, 0.1, 0.2]
    lr_decay_array = [0, 0.75]

    nbr_permutations = 100

    max_num_models = len(max_iter_array) * len(inertia_array) * len(lr_decay_array)

    num_model = 0
    for max_iter in max_iter_array:
        for inertia in inertia_array:
            for lr_decay in lr_decay_array:
                num_model += 1
                # K-fold cross-validation
                print('####################################################################################################################################################')
                print(f'Performing cross validation for HMM #{num_model}/{max_num_models} with : \n\tMax iter = {max_iter}\n\tInertia = {inertia}\n\tLr decay = {lr_decay}')
                hmm_parameters_list.append({'max_iter': max_iter,
                                            'inertia': inertia,
                                            'lr_decay': lr_decay,
                                            'mean_kendall_tau': 0})
                mean_kendall_tau, mean_distance = cross_validation_pomegranate(nbr_folds=nbr_folds, nbr_states=nbr_states, feature_list=feature_list, nbr_val_per_feature=nbr_val_per_feature, train_sequences=train_sequences, max_iter=max_iter, inertia=inertia, stop_threshold=100, random_emission_prob=False, lr_decay=lr_decay, nbr_permutations=nbr_permutations)
                hmm_distance_list.append(mean_distance)
                hmm_parameters_list[-1]['mean_kendall_tau'] = mean_kendall_tau
                print(f'Mean Kendall Tau for current model : {mean_kendall_tau}')
                print('####################################################################################################################################################')
    print(sorted(hmm_parameters_list, key=lambda x: x['mean_kendall_tau']))
    plt.scatter([ind for ind in range(len(hmm_distance_list))], hmm_distance_list)
    plt.show()


# Fonction qui permet de faire la cross-validation (K-fold) des HMMs avec la librairie Pomegranate.
def cross_validation_pomegranate(nbr_folds, nbr_states, feature_list, nbr_val_per_feature, train_sequences, max_iter, inertia=None, stop_threshold=1000, random_emission_prob=True, lr_decay=0, nbr_permutations=1000):
    # List of Kendall's Tau
    kendalltau_array_folds = []

    # List of models
    model_list = []

    kfolds = KFold(n_splits=nbr_folds, shuffle=True)
    fold_indexes = kfolds.split(train_sequences)
    i = 0
    for train_index, test_index in fold_indexes:
        print(f'\tFold #{i+1}')
        kendalltau_array_seq = []

        train_sequences_for_fold = [train_sequences[e] for e in train_index]    # Training sequences for the (i+1)th fold
        print('\t\tInitializing HMM')
        model = initialize_hmm(nbr_states, feature_list, nbr_val_per_feature, train_sequences_for_fold, random_emission_prob=False)
        print('\t\tTraining HMM')
        model.fit(sequences=train_sequences_for_fold, max_iterations=max_iter, n_jobs=-1, verbose=True, inertia=inertia, stop_threshold=100, lr_decay=lr_decay)
        print('\t\tCalculating Kendall Tau')
        for num, index in enumerate(test_index):
            if num % 200 == 0:
                print(num)
            seq = train_sequences[index].copy()
            original_ordering = [k for k in range(len(seq))]
            max_log_likelihood = model.log_probability(seq)
            ordering_of_max_ll_seq = original_ordering
            temp = list(enumerate(seq))
            for rep in range(nbr_permutations):
                new_ordering = [k for k in range(len(seq))]
                random.shuffle(temp)
                index_of_seq_elem, shuffled_seq = zip(*temp)
                for m, ind in enumerate(index_of_seq_elem):
                    new_ordering[ind] = m
                ll_shuffled_seq = model.log_probability(list(shuffled_seq))
                if ll_shuffled_seq > max_log_likelihood:
                    max_log_likelihood = ll_shuffled_seq
                    ordering_of_max_ll_seq = new_ordering
            kendall_tau, _ = kendalltau(original_ordering, ordering_of_max_ll_seq)
            kendalltau_array_seq.append(kendall_tau)
        kendalltau_array_folds.append(np.mean(kendalltau_array_seq))
        print(f'Kendall Tau of current fold : {np.mean(kendalltau_array_seq)}')

        model_list.append(model.copy())
        i += 1

    print('Computing distance between every model of the current cross-validation procedure.')
    total_dist = 0
    nbr_comparisons = 0
    for i in range(nbr_folds):
        for j in range(i, nbr_folds):
            if i != j:
                nbr_comparisons += 1
                dist_array = []
                for k in range(15):
                    dist_array.append(distance_hmm(model_list[i], model_list[j], 500))
                dist_array = [val for val in dist_array if not (math.isnan(val) or math.isinf(val))]
                if len(dist_array) == 0:
                    print('PROBLEM WITH DISTANCES')
                else:
                    total_dist += sum(dist_array) / len(dist_array)
    mean_dist = total_dist / nbr_comparisons

    return np.mean(kendalltau_array_folds), mean_dist


# Function to train and evaluate the performances of an HMM (compute the Kendall's Tau value).
# If save == True, will save every trained model.
def evaluate_hmm_performances_pomegranate(nbr_models, nbr_permutations, save, save_path):
    dict_articles = get_feature_dict()
    feature_to_binary(dict_articles)
    train_dict, test_dict = split_train_test_w_date(dict_articles, train_ratio=0.8)
    feature_list = ['toxicity_score', 'sarcasm_score', 'sentiment_score', 'anger_score', 'fear_score', 'joy_score',
                      'love_score', 'sadness_score', 'surprise_score', 'elapsed_time', 'contains_url', 'contains_email',
                      'contains_hashtag', 'image_only', 'starts_w_name', 'nbr_likes', 'nbr_first_per_sing_pronouns',
                      'nbr_first_per_plur_pronouns', 'nbr_second_per_pronouns', 'nbr_third_per_sing_pronouns',
                      'nbr_third_per_plur_pronouns', 'nbr_politeness_gratitude', 'comment_length_in_words']
    nbr_values_per_feature = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
    min_comment_length = 10
    max_comment_length = 200000000000000000
    train_sequences, train_seq_lengths, _ = extract_features_from_dict(train_dict, feature_list, min_comment_length=min_comment_length, max_comment_length=max_comment_length)
    test_sequences, test_seq_lengths, dict_test_articles = extract_features_from_dict(test_dict, feature_list, min_comment_length=min_comment_length, max_comment_length=max_comment_length)

    nbr_states = 9
    max_iterations = 8
    inertia = 0.1
    lr_decay = 0

    for model_num in range(nbr_models):
        print(f'Evaluation of model #{model_num}')
        print('Initializing HMM')
        model = initialize_hmm(nbr_states, feature_list, nbr_values_per_feature, train_sequences, random_emission_prob=False)
        print('Training HMM')
        model.fit(sequences=train_sequences, max_iterations=max_iterations, n_jobs=-1, verbose=True, inertia=inertia, stop_threshold=100, lr_decay=lr_decay)

        kendalltau_array = []
        for num, seq_original in enumerate(test_sequences):
            if num % 200 == 0:
                print(f'Seq #{num}')
            seq = seq_original.copy()
            original_ordering = [k for k in range(len(seq))]
            max_log_likelihood = model.log_probability(seq)
            ordering_of_max_ll_seq = original_ordering
            temp = list(enumerate(seq))
            for j in range(nbr_permutations):
                new_ordering = [k for k in range(len(seq))]
                random.shuffle(temp)
                index, shuffled_seq = zip(*temp)
                for i, ind in enumerate(index):
                    new_ordering[ind] = i
                ll_shuffled_seq = model.log_probability(list(shuffled_seq))
                if ll_shuffled_seq > max_log_likelihood:
                    max_log_likelihood = ll_shuffled_seq
                    ordering_of_max_ll_seq = new_ordering
            kendall_tau, _ = kendalltau(original_ordering, ordering_of_max_ll_seq)
            kendalltau_array.append(kendall_tau)
        mean_kendalltau = np.mean(kendalltau_array)
        print(mean_kendalltau)
        if save:
            save_pomegranate_model(model, save_path + f"\\hmm{model_num}.json")


if __name__ == '__main__':
    #np.random.seed(42)
    np.set_printoptions(suppress=True)

    # Step 1 - analyze the distribution of various features
    feature_dict_load_path = ""     # Replace with the location of the "final_feature_dict.json" file
    feature_to_analyze = "toxicity_score"   # Can be replaced with any of the 23 features
    analyze_features(feature_dict_load_path, feature_to_analyze)

    # Step 2 - determine the optimal number of states, using the Silhouette coefficient
    candidate_nbr_biclusters = [i for i in range(2, 16)]
    for nbr_biclusters in candidate_nbr_biclusters:
        print(f'For {nbr_biclusters} states')
        for i in range(5):
            print(f'Results #{i+1}')
            silhouette_coefficient(nbr_states=nbr_biclusters, percentage=0.01)

    # Step 3 - compute the distance between HMMs trained using the same hyperparameter values, to see if they
    # converged to a similar HMM
    nbr_models = 5
    nbr_states = 9
    iter_list = [i for i in range(2, 21)]
    random_emission_prob = False
    length_list = [400, 500, 1000]
    test_distance_measure(nbr_models, nbr_states, iter_list, random_emission_prob, length_list)

    # Step 4 - find the optimal hyperparameter values
    feature_dict = get_feature_dict(feature_dict_load_path)
    feature_to_binary(feature_dict)
    train_dict, test_dict = split_train_test_w_date(feature_dict, train_ratio=0.8)
    feature_list = ['toxicity_score', 'sarcasm_score', 'sentiment_score', 'anger_score', 'fear_score', 'joy_score',
                    'love_score', 'sadness_score', 'surprise_score', 'elapsed_time', 'contains_url', 'contains_email',
                    'contains_hashtag', 'image_only', 'starts_w_name', 'nbr_likes', 'nbr_first_per_sing_pronouns',
                    'nbr_first_per_plur_pronouns', 'nbr_second_per_pronouns', 'nbr_third_per_sing_pronouns',
                    'nbr_third_per_plur_pronouns', 'nbr_politeness_gratitude', 'comment_length_in_words']
    nbr_values_per_feature = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
    min_comment_length = 10
    max_comment_length = 200000000000000000
    train_sequences, train_seq_lengths, _ = extract_features_from_dict(train_dict, feature_list, min_comment_length=min_comment_length, max_comment_length=max_comment_length)
    test_sequences, test_seq_lengths, dict_test_articles = extract_features_from_dict(test_dict, feature_list, min_comment_length=min_comment_length, max_comment_length=max_comment_length)

    nbr_folds = 5
    grid_search_with_cv(nbr_folds, train_sequences, test_sequences, feature_list, nbr_values_per_feature, nbr_states)

    # Step 5 - evaluate the performances of HMMs trained using the optimal number of states and hyperparameter values
    nbr_models = 12
    nbr_permutations = 100
    save = True
    save_path = ""  # Replace with actual path to the location where the trained models will be saved
    evaluate_hmm_performances_pomegranate(nbr_models, nbr_permutations, save, save_path)
    # The 12 trained models used in the paper are in the files "hmm1.json" through "hmm12.json"
