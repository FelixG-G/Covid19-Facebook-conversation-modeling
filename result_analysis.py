from nltk.corpus import stopwords
import pandas as pd
from graphviz import Digraph
from hmm_training import *
from sklearn.feature_extraction.text import TfidfVectorizer


# Draws a state transition graph from a given trained HMM.
def draw_transition_graph(model):
    node_names = ['Positive', 'Images / GIF', 'Negative / toxic', 'COVID-19 & vaccine \nworries or skepticism', 'Misc 1', 'URLs', 'Misc 2', 'Negative - \nsociety & economy', 'Negative - politicians']

    dot = Digraph('HMMTransition')
    dot.node('Start')
    for n in range(9):
        print(n)
        dot.node(node_names[n])

    transition_mat = model.dense_transition_matrix()
    print(transition_mat)

    for i, row in enumerate(transition_mat):
        if i == len(transition_mat) - 1:
            break
        print(f'Row #{i} : ')
        print(row)
        for j, elem in enumerate(row):
            if j == len(row) - 2:
                break
            print(f'\tElem #{j} : ')
            print(f'\t{elem}')
            if elem >= 0.15:
                if i == len(transition_mat) - 2:
                    dot.edge(f'Start', node_names[j], label=f'{np.round(elem, 4)}')
                else:
                    dot.edge(node_names[i], node_names[j], label=f'{np.round(elem, 4)}')
    dot.render(view=True)


# Computes the TF-IDF score of individual words and bigrams using the set of words and bigrams
# from the different states of a trained HMM as documents.
def extract_most_common_words_per_states(model):
    stop_words = set(stopwords.words('english'))
    # Get the necessary dictionaries of comments
    feature_dict = get_feature_dict()
    feature_to_binary(feature_dict)
    train_dict, test_dict = split_train_test_w_date(feature_dict, train_ratio=0.8)
    attribute_list = ['toxicity_score', 'sarcasm_score', 'sentiment_score', 'anger_score', 'fear_score', 'joy_score',
                      'love_score', 'sadness_score', 'surprise_score', 'elapsed_time', 'contains_url', 'contains_email',
                      'contains_hashtag', 'image_only', 'starts_w_name', 'nbr_likes', 'nbr_first_per_sing_pronouns',
                      'nbr_first_per_plur_pronouns', 'nbr_second_per_pronouns', 'nbr_third_per_sing_pronouns',
                      'nbr_third_per_plur_pronouns', 'nbr_politeness_gratitude', 'comment_length_in_words']
    nbr_values_per_feature = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
    min_comment_length = 10
    max_comment_length = 200000000000000000
    train_sequences, train_seq_lengths, _ = extract_features_from_dict(train_dict, attribute_list, min_comment_length=min_comment_length, max_comment_length=max_comment_length)
    test_sequences, test_seq_lengths, dict_test_articles = extract_features_from_dict(test_dict, attribute_list, min_comment_length=min_comment_length, max_comment_length=max_comment_length)

    # Assign each comment to a state (according to the model's transition and emission probabilities)
    nbr_com = 0
    dict_comments_per_states = {i: [] for i in range(9)}
    print(dict_comments_per_states)
    for j, art_id in enumerate(dict_test_articles.keys()):
        if j % 100 == 0:
            print(j)
        predicted_states = model.predict(dict_test_articles[art_id])
        for i, state in enumerate(predicted_states):
            nbr_com += 1
            dict_comments_per_states[state].append(feature_dict[art_id][i]['comment_text'])
    print(nbr_com)

    # Create a list that contains nbr_states items. Each item is a concatenation of
    # every comment in that state. These items will correspond to the documents.
    list_comments_per_state = []
    for i, state in enumerate(dict_comments_per_states.keys()):
        print(f'State #{i}')
        concatenated_comments = ''  # This string will contain a concatenation of every comment from the current state.
        for j, comment in enumerate(dict_comments_per_states[state]):
            if j % 1000 == 0:
                print(f'\tComment #{j}')

            # Replace URLs with '0123url0123' to make every URL count as the same word
            if re.search(r'(?:https?:\/\/|www[.])\S+', comment):
                comment = re.sub(r'(?:https?:\/\/|www[.])\S+', '0123url0123', comment)

            # We perform a similar replacement for email addresses
            if re.search(r'[\w\.-]+@[\w\.-]+[.][\w\.-]+', comment):
                comment = re.sub(r'[\w\.-]+@[\w\.-]+[.][\w\.-]+', '0123email0123', comment)

            # Punctuation is removed
            comment = re.sub(f'[,.!?]', ' ', comment)

            concatenated_comments += comment + ' '  # The actual concatenation

        # The string that contains a concatenation of all the comments from the current state
        # is then added to the list. In total, there should be as many items in the list as
        # there are states in the model.
        list_comments_per_state.append(concatenated_comments)

    # Change the ngram_range parameter to obtain results for bigrams instead of individual words
    vect = TfidfVectorizer(stop_words=list(stop_words), max_df=6, min_df=1, max_features=None, sublinear_tf=True, ngram_range=(1, 1))
    X = vect.fit_transform(list_comments_per_state)
    X = X.toarray()

    dict_tfidf = {i: {word: X[i][j] for j, word in enumerate(vect.get_feature_names_out())} for i in range(9)}

    # Print the 50 words / bigrams that obtained the highest TF-IDF scores
    for state in dict_tfidf.keys():
        print(f'For state #{state}')
        sorted_tfidf_list = sorted(dict_tfidf[state].items(), key=lambda x: x[1], reverse=True)
        print(sorted_tfidf_list[:50])


# Generates a heatmap of the emission probabilities for the different features of every state of a given model
def heatmap_features(excel_file_path):
    # The data to be turned into a heatmap should be stored in an Excel file, accessed here :
    df = pd.read_excel(excel_file_path, index_col=0)
    print(df)
    sns.set(rc={'figure.figsize': (10, 30)})
    heatmap = sns.heatmap(df, vmin=0, vmax=1, annot=True, annot_kws={'fontsize': 'medium', 'fontstretch': 1, 'ha': 'center'}, linewidths=0, cmap='Blues', square=True, fmt='2.0%', xticklabels=['Average', 'Positive', 'Images / GIF', 'Negative / toxic', 'COVID-19 & vaccine \nworries or skepticism', 'URLs', 'Negative - \nsociety & economy', 'Negative - politicians', 'Misc. 1', 'Misc. 2'])
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False, left=False, labeltop=True)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
    heatmap.vlines([1], *heatmap.get_ylim(), color='black', linewidths=2)
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # Display a transition graph for a given trained HMM
    model_path = ""     # Replace with path to the location of the stored trained HMM
    model = load_pomegranate_model(model_path)
    draw_transition_graph(model)

    # Extract the themes of hidden states for a given trained HMM using TF-IDF score of words and bigrams
    model_path = ""  # Replace with path to the location of the stored trained HMM
    model = load_pomegranate_model(model_path)
    extract_most_common_words_per_states(model)

    # Display a heatmap of the emission probabilities of a trained HMM
    excel_file_path = ""    # Replace with actual path for the Excel file. Use the "emission_probabilities_model1.xlsx" file to get the same results as in the paper
    heatmap_features(excel_file_path)
