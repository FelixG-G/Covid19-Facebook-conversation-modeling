import re
import nltk
import statistics
import math
import json
import matplotlib.pyplot as plt
from datetime import datetime


# Function used to load the raw comment list.
def load_comment_list(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        comment_list = json.load(json_file)
    return comment_list


# Number of comments per articles, stats and histograms
def stats_comment_per_article(comment_list):
    # Dictionary with article (convo) IDs as keys and the number of comments in the article as values.
    dict_article = {}
    for line in comment_list:
        l = line.split('\t')
        # The first element (l[0]) corresponds to the ID of the article
        if l[0] not in dict_article.keys():
            dict_article[l[0]] = 1
        else:
            dict_article[l[0]] += 1

    # Statistics on the number of comments per article
    list_article_lengths = []
    for value in dict_article.values():
        list_article_lengths.append(value)
    print(f'Nbr of different articles : {len(list_article_lengths)}')
    print(f'Average : {statistics.mean(list_article_lengths)}')
    print(f'Median (Q2) : {statistics.median(list_article_lengths)}')
    quantiles = statistics.quantiles(list_article_lengths, n=4)
    print(f'Quantiles : {quantiles}')
    print(f'Mode : {statistics.mode(list_article_lengths)} (count : {list_article_lengths.count(statistics.mode(list_article_lengths))})')
    print(f'Max : {max(list_article_lengths)}')
    print(f'Min : {min(list_article_lengths)}')

    # Histogram where each bar correspond to 1 comment. Includes outliers.
    bin_num = [i for i in range(0, math.ceil(max(list_article_lengths)) + 2)]
    plt.hist(list_article_lengths, bins=bin_num)
    plt.show()

    # Histogram where each bar corresponds to 10 comments. Includes outliers.
    bin_num = [i * 10 for i in range(0, math.ceil(max(list_article_lengths) / 10) + 1)]
    plt.hist(list_article_lengths, bins=bin_num)
    plt.show()

    # Removing outliers for better results and better histograms.
    q1 = quantiles[0]
    q3 = quantiles[2]
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    temp_list = list_article_lengths.copy()
    for elem in temp_list:
        if elem > upper or elem < lower:
            list_article_lengths.remove(elem)

    # Histogram where each bar corresponds to 1 comment. Excludes outliers.
    bin_num = [i for i in range(0, math.ceil(max(list_article_lengths)) + 2)]
    plt.hist(list_article_lengths, bins=bin_num)
    plt.show()

    # Histogram where each bar corresponds to 10 comments. Excludes outliers.
    bin_num = [i * 10 for i in range(0, math.ceil(max(list_article_lengths) / 10) + 1)]
    plt.hist(list_article_lengths, bins=bin_num)
    plt.show()


# Stats on the duration of conversations
def stats_duree_convo(comment_list, remove_outliers=False):
    # Dictionary with article (convo) IDs as keys and arrays of comment timestamps as values.
    dict_article_time = {}
    for line in comment_list:
        l = line.split('\t')
        id = l[0]
        time = l[16]
        if re.match(r'^\d{4}\-\d{2}\-\d{2}$', time):
            time = datetime.strptime(time, '%Y-%m-%d')
        else:
            time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

        if id not in dict_article_time.keys():
            dict_article_time[id] = [time]
        else:
            dict_article_time[id].append(time)

    # Remove outliers, if remove_outliers == True
    if remove_outliers:
        prct_keep = 0.9
        for key in dict_article_time.keys():
            nbr_elem = math.ceil(len(dict_article_time[key]) * prct_keep)
            dict_article_time[key].sort()
            dict_article_time[key] = dict_article_time[key][:nbr_elem]

    # For each convo, compute the duration.
    time_list_secs = []
    days_to_secs = 24 * 60 * 60
    for key in dict_article_time.keys():
        max_time = max(dict_article_time[key])
        min_time = min(dict_article_time[key])
        diff_time = max_time - min_time
        convo_duration_sec = diff_time.days * days_to_secs + diff_time.seconds
        time_list_secs.append(convo_duration_sec)
    time_list_days = [i / days_to_secs for i in time_list_secs]

    # Display different statistics
    print(f'Mean in secs : {statistics.mean(time_list_secs)}')
    print(f'Mean in days : {statistics.mean(time_list_days)}')
    quantiles_secs = statistics.quantiles(time_list_secs, n=4)
    quantiles_days = statistics.quantiles(time_list_days, n=4)
    print(f'Quantiles secs : {quantiles_secs}')
    print(f'Quantiles days : {quantiles_days}')
    print(f'Max secs : {max(time_list_secs)}')
    print(f'Max days : {max(time_list_days)}')
    print(f'Min secs : {min(time_list_secs)}')
    print(f'Min days : {min(time_list_days)}')
    print(f'Mode secs : {statistics.mode(time_list_secs)}')
    print(f'Mode days : {statistics.mode(time_list_days)}')

    bin_upper_limit = math.ceil(max(time_list_days))
    # Histogram where each bar = 1 day. Includes outliers.
    bin_num = [i for i in range(0, bin_upper_limit+1)]
    plt.hist(time_list_days, bins=bin_num)
    plt.show()

    # Histogram where each bar = half a day (12 hours). Includes outliers.
    bin_num = [i / 2 for i in range(0, bin_upper_limit*2+1)]
    plt.hist(time_list_days, bins=bin_num)
    plt.show()

    # Histogram where each bar = 6 hours. Includes outliers.
    bin_num = [i / 4 for i in range(0, bin_upper_limit*4+1)]
    plt.hist(time_list_days, bins=bin_num)
    plt.show()

    # Removing outliers.
    q1 = quantiles_days[0]
    q3 = quantiles_days[2]
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    temp_list = time_list_days.copy()
    for elem in temp_list:
        if elem > upper or elem < lower:
            time_list_days.remove(elem)

    bin_upper_limit = math.ceil(max(time_list_days))
    # Histogram where each bar = 1 day. Excludes outliers.
    bin_num = [i for i in range(0, bin_upper_limit+1)]
    plt.hist(time_list_days, bins=bin_num)
    plt.show()

    # Histogram where each bar = half a day (12 hours). Excludes outliers.
    bin_num = [i / 2 for i in range(0, bin_upper_limit*2+1)]
    plt.hist(time_list_days, bins=bin_num)
    plt.show()

    # Histogram where each bar = 6 hours. Excludes outliers.
    bin_num = [i / 4 for i in range(0, bin_upper_limit*4+1)]
    plt.hist(time_list_days, bins=bin_num)
    plt.show()


# Stats on the number of words per comment.
def stats_number_words(comment_list):
    # Extracting the number of words per comment
    word_count = []
    for line in comment_list:
        l = line.split('\t')
        words = l[19].split(' ')
        word_count.append(len(words))
    print(len(word_count))

    # Statistics on the number of words per comment.
    print(f'Average : {statistics.mean(word_count)}')
    quantiles = statistics.quantiles(word_count, n=4)
    print(f'Quartiles : {quantiles}')
    print(f'Max : {max(word_count)}')
    print(f'Min : {min(word_count)}')
    print(f'Mode : {statistics.mode(word_count)} (count : {word_count.count(statistics.mode(word_count))})')

    # Histogram where each bar = 1 word. Includes outliers.
    bin_num = [i for i in range(0, max(word_count)+2)]
    plt.hist(word_count, bins=bin_num)
    plt.show()

    # Histogram where each bar = 10 words. Includes outliers.
    bin_num = [i * 10 for i in range(0, math.ceil(max(word_count) / 10) + 1)]
    plt.hist(word_count, bins=bin_num)
    plt.show()

    # Removing outliers
    q1 = quantiles[0]
    q3 = quantiles[2]
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    word_count.sort()
    index = 0
    for i, nbr in enumerate(word_count):
        if nbr > upper:
            index = i
            break
    word_count = word_count[:index]

    # Histogram where each bar = 1 word. Excludes outliers.
    bin_num = [i for i in range(0, max(word_count) + 2)]
    plt.hist(word_count, bins=bin_num)
    plt.show()

    # Histogram where each bar = 10 words. Excludes outliers.
    bin_num = [i * 10 for i in range(0, math.ceil(max(word_count) / 10) + 1)]
    plt.hist(word_count, bins=bin_num)
    plt.show()


# Stats on the number of likes per comment
def stats_number_likes(comment_list):
    # Extracting the number of likes per comment
    like_count = []
    for line in comment_list:
        l = line.split('\t')
        if re.match(r'^\d+\.0$', l[24]):
            like_count.append(float(l[24]))
        else:
            like_count.append(0)
    print(len(like_count))

    # Statistics on the number of likes per comment (including comments with 0 likes).
    print(f'Average : {statistics.mean(like_count)}')
    quantiles = statistics.quantiles(like_count, n=4)
    print(f'Quartiles : {quantiles}')
    print(f'Max : {max(like_count)}')
    print(f'Min : {min(like_count)}')
    print(f'Mode : {statistics.mode(like_count)} (count : {like_count.count(statistics.mode(like_count))})')

    # Removing comments with 0 likes.
    like_count.sort()
    index = 0
    for i, nbr in enumerate(like_count):
        if nbr > 0:
            index = i
            break
    like_count_wo_zeros = like_count[index:]

    # Statistics on the number of likes per comment (excluding comments with 0 likes)
    print(f'Average : {statistics.mean(like_count_wo_zeros)}')
    quantiles = statistics.quantiles(like_count_wo_zeros, n=4)
    print(f'Quartiles : {quantiles}')
    print(f'Max : {max(like_count_wo_zeros)}')
    print(f'Min : {min(like_count_wo_zeros)}')
    print(f'Mode : {statistics.mode(like_count_wo_zeros)} (count : {like_count_wo_zeros.count(statistics.mode(like_count_wo_zeros))})')

    # Histogram where each bar = 1 like. Includes outliers.
    print(max(like_count))
    bin_num = [i for i in range(0, int(max(like_count)) + 2)]
    plt.hist(like_count, bins=bin_num)
    plt.show()

    # Histogram where each bar = 5 likes. Includes outliers.
    bin_num = [i * 5 for i in range(0, math.ceil(max(like_count) / 5) + 1)]
    plt.hist(like_count, bins=bin_num)
    plt.show()

    # Histogram where each bar = 10 likes. Includes outliers.
    bin_num = [i * 10 for i in range(0, math.ceil(max(like_count) / 10) + 1)]
    plt.hist(like_count, bins=bin_num)
    plt.show()

    # Removing outliers
    q1 = quantiles[0]
    q3 = quantiles[2]
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    like_count.sort()
    index = 0
    for i, nbr in enumerate(like_count):
        if nbr > upper:
            index = i
            break
    like_count = like_count[:index]

    # Histogram where each bar = 1 like. Excludes outliers.
    bin_num = [i for i in range(0, int(max(like_count)) + 2)]
    plt.hist(like_count, bins=bin_num)
    plt.show()

    # Histogram where each bar = 2 likes. Excludes outliers.
    bin_num = [i * 2 for i in range(0, math.ceil(max(like_count) / 2) + 1)]
    plt.hist(like_count, bins=bin_num)
    plt.show()

    # Histogram where each bar = 5 likes. Excludes outliers.
    bin_num = [i * 5 for i in range(0, math.ceil(max(like_count) / 5) + 1)]
    plt.hist(like_count, bins=bin_num)
    plt.show()


# Number of comments that begin with a proper noun
def stats_replies(comment_list):
    total = 0
    for i, line in enumerate(comment_list):
        if i % 1000 == 0:
            print(i)
        l = line.split('\t')
        text = re.sub(r'^@', '', l[19])
        words = nltk.word_tokenize(text)
        tagged_words = nltk.pos_tag(words)
        entities = nltk.ne_chunk(tagged_words)
        if len(entities) > 0 and hasattr(entities[0], 'label') and entities[0].label() == 'PERSON':
            total += 1
    print(total)


# Statistics on the comments that contain swear words, slurs, insults, etc.
def stats_profanities(comment_list):
    swear_words = ['stupid', 'idiot', 'fuck', 'asshole', 'ass hole', 'fuck off', 'fuckoff', 'fag',
                   'pussy', 'puss', 'ass', 'shit', 'bitch', 'asshat', 'ass hat', 'butt', 'crap',
                   'bloody', 'bollocks', 'arse', 'bugger', 'bullshit', 'bull shit', 'chicken shit',
                   'chickenshit', 'cock', 'cock tease', 'cocktease', 'coon ass', 'coonass',
                   'corn hole', 'cornhole', 'cracker', 'cunt', 'damn', 'damnation', 'dick', 'balls',
                   'faggot', 'fagget', 'feck', 'frak', 'fucc', 'fuc', 'son of a bitch', 'ballsucker',
                   'ball sucker', 'fuk', 'git', 'horseshit', 'horse shit', 'mother fucker',
                   'motherfucker', 'nigga', 'nigger', 'niga', 'bitchy', 'paki', 'prick', 'rat fucking',
                   'ratfucking', 'shut the fuck up', 'bastard', 'bitch-ass', 'bitchass', 'bitch ass',
                   'shut the hell up', 'shut up', 'shutup', 'shut it', 'slut', 'spic', 'twat', 'chink',
                   'wanker', 'wank', 'cockhead', 'cock head', 'cocksucker', 'cock sucker', 'cum',
                   'cum dumpster', 'cumslut', 'cum slut', 'dickhead', 'dick head', 'dicksucker',
                   'dick sucker', 'dumbshit', 'dumb shit', 'dumb cunt', 'dumb', 'faggy',
                   'fucked', 'fucker', 'fuckery', 'fucking', 'fucktard', 'fuckwad', 'fuckwit',
                   'gay-ass', 'gay ass', 'gayass', 'gaylord', 'gay lord', 'gay cunt', 'homo', 'jizz',
                   'ho', 'hoe', 'niglet', 'nutsack', 'nut sack', 'nutsucker', 'nut sucker', 'retard',
                   'shitass', 'shit ass', 'shit cunt', 'shitface', 'shit face', 'shithead',
                   'shit head', 'shithole', 'shit hole', 'shithouse', 'shit house', 'shitty', 'whore',
                   'whorehouse', 'whore house']
    print(len(swear_words))
    swear_words_v2 = ['stupid', 'idiot', 'asshole', 'ass hole', 'fuck off', 'fuckoff', 'fag',
                   'pussy', 'puss', 'bitch', 'asshat', 'ass hat', 'chicken shit',
                   'chickenshit', 'cock tease', 'cocktease', 'coon ass', 'coonass',
                   'corn hole', 'cornhole', 'cracker', 'cunt', 'dick',
                   'faggot', 'fagget', 'son of a bitch', 'ballsucker',
                   'ball sucker', 'git', 'mother fucker',
                   'motherfucker', 'nigga', 'nigger', 'niga', 'bitchy', 'paki', 'prick', 'shut the fuck up', 'bastard', 'bitch-ass', 'bitchass', 'bitch ass',
                   'shut the hell up', 'shut up', 'shutup', 'shut it', 'slut', 'spic', 'twat', 'chink',
                   'wanker', 'cockhead', 'cock head', 'cocksucker', 'cock sucker',
                   'cum dumpster', 'cumslut', 'cum slut', 'dickhead', 'dick head', 'dicksucker',
                   'dick sucker', 'dumbshit', 'dumb shit', 'dumb cunt', 'dumb', 'faggy', 'fucker', 'fucktard', 'fuckwad', 'fuckwit',
                   'gay-ass', 'gay ass', 'gayass', 'gaylord', 'gay lord', 'gay cunt', 'homo',
                   'ho', 'hoe', 'niglet', 'nutsack', 'nut sack', 'nutsucker', 'nut sucker', 'retard',
                   'shitass', 'shit ass', 'shit cunt', 'shitface', 'shit face', 'shithead',
                   'shit head', 'shithole', 'shit hole', 'shithouse', 'shit house', 'shitty', 'whore',
                   'whorehouse', 'whore house']

    dict_swear_words = {}
    total = 0
    for i, line in enumerate(comment_list):
        if i % 1000 == 0:
            print(i)
        l = line.split('\t')
        comment = l[19].lower()
        for word in swear_words_v2:
            if re.search(rf'\s+{word}\s+', comment):
                if word not in dict_swear_words.keys():
                    dict_swear_words[word] = 1
                else:
                    dict_swear_words[word] += 1
                total += 1
                break
    print(dict_swear_words)
    print(total)


if __name__ == '__main__':
    path = ""   # Replace with path name to the "comment_list.json" file
    comment_list = load_comment_list(path)

    # Uncomment the desired function

    #stats_comment_per_article(comment_list)
    #stats_duree_convo(comment_list, remove_outliers=True)
    #stats_number_words(comment_list)
    #stats_number_likes(comment_list)
    #stats_replies(comment_list)
    #stats_profanities(comment_list)




