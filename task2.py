import nltk
from nltk.corpus import PlaintextCorpusReader
from collections import Counter
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

# Task 2: Sentiment analysis of movie reviews
# a) Build a sentiment lexicon using a semi-supervised approach by bootstrapping the process, starting from a small
# lexicon of adjectives and corpus 2.

corpus_2_root = '/Users/raduliana/Desktop/git/NLS_cwk_2/data/rt-polaritydata/'
corpus_2 = PlaintextCorpusReader(corpus_2_root, '.*\.txt')
corpus_2_parsed = []
special_chars = ['[', ']', '*', '-', '(', ')']
for file_id in corpus_2.fileids():
    tokens = nltk.word_tokenize(corpus_2.raw(file_id))
    tokens_cleaned = [w for w in tokens if w not in special_chars]
    corpus_2_parsed += nltk.pos_tag(tokens_cleaned)

# Get the list of negative seed adj
f = open("data/neg_adj.txt", "r")
neg_adj = []
for line in f:
    neg_adj.append(line.rstrip())

# Get the list of positive seed adj
f = open("data/pos_adj.txt", "r")
pos_adj = []
for line in f:
    pos_adj.append(line.rstrip())


# ------------------ Methods --------------------------------------------
# method to check the neighbours for and pattern
def check_neighbours_for_and_pattern(current_index, corpus_parsed, polarity_list):
    # check right neighbours
    i = current_index + 1
    while i < len(corpus_parsed):
        if corpus_parsed[i][0] == ',' or corpus_parsed[i][0] == 'and' or 'RB' in corpus_parsed[i][1]:
            i += 1
            continue
        elif corpus_parsed[i][1] == 'JJ':
            polarity_list.append(corpus_parsed[i][0])
            i += 1
        else:
            break

    # check left neighbours
    j = current_index - 1
    while j > 0:
        if corpus_parsed[j][0] == ',' or corpus_parsed[j][0] == 'and' 'RB' in corpus_parsed[j][1]:
            j -= 1
            continue
        elif corpus_parsed[j][1] == 'JJ':
            if corpus_parsed[j][0] == 'stuff':
                here = 1
            polarity_list.append(corpus_parsed[j][0])
            j -= 1
        else:
            break
    return polarity_list


# ---------------------------------------------------------------------------


# Populate lexicon with adjectives
for index in range(len(corpus_2_parsed)):
    current_pair = corpus_2_parsed[index]

    # Consider list of adj conjoined by and or comma or just by space - pattern: (RB)JJ, (RB)JJ, (RB)JJ,.. ,and (RB)JJ
    if current_pair[0] in neg_adj:
        neg_adj = check_neighbours_for_and_pattern(index, corpus_2_parsed, neg_adj)
    if current_pair[0] in pos_adj:
        pos_adj = check_neighbours_for_and_pattern(index, corpus_2_parsed, pos_adj)

    # pattern: JJ but (RB) JJ - this structure give adj with different polarities
    if current_pair[0] == 'but':
        if index + 1 < len(corpus_2_parsed) and index - 1 > 0:
            prev_jj = -1
            next_jj = -1
            if corpus_2_parsed[index - 1][1] == 'JJ':
                prev_jj = corpus_2_parsed[index - 1]

                if corpus_2_parsed[index + 1][1] == 'JJ':
                    next_jj = corpus_2_parsed[index + 1]
                elif index + 2 < len(corpus_2_parsed) and 'RB' in corpus_2_parsed[index + 1][1] and \
                        corpus_2_parsed[index + 2][1] == 'JJ':
                    next_jj = corpus_2_parsed[index + 2]

            if not prev_jj == -1 and not next_jj == -1:
                if prev_jj[0] in pos_adj:
                    neg_adj.append(next_jj[0])
                elif prev_jj[0] in neg_adj:
                    pos_adj.append(next_jj[0])
                elif next_jj[0] in pos_adj:
                    neg_adj.append(prev_jj[0])
                elif next_jj[0] in neg_adj:
                    pos_adj.append(prev_jj[0])

    # pattern: JJ but not (RB) JJ - this structure give adj with same polarities
    if current_pair[0] == 'but':
        if index + 2 < len(corpus_2_parsed) and index - 1 > 0:
            prev_jj = -1
            next_jj = -1
            if corpus_2_parsed[index - 1][1] == 'JJ' and corpus_2_parsed[index + 1][0] == 'not':
                prev_jj = corpus_2_parsed[index - 1]
                if corpus_2_parsed[index + 2][1] == 'JJ':
                    next_jj = corpus_2_parsed[index + 2]
                elif index + 3 < len(corpus_2_parsed) and 'RB' in corpus_2_parsed[index + 2][1] and \
                        corpus_2_parsed[index + 3][1] == 'JJ':
                    next_jj = corpus_2_parsed[index + 3]

            if not prev_jj == -1 and not next_jj == -1:
                if prev_jj[0] in pos_adj:
                    pos_adj.append(next_jj[0])
                elif prev_jj[0] in neg_adj:
                    neg_adj.append(next_jj[0])
                elif next_jj[0] in pos_adj:
                    pos_adj.append(prev_jj[0])
                elif next_jj[0] in neg_adj:
                    neg_adj.append(prev_jj[0])

# Deal with duplicates based on the number of occurances in the list
pos_adj_dict = dict(Counter(pos_adj))
neg_adj_dict = dict(Counter(neg_adj))

adj_both_polarity = dict(Counter(pos_adj) & Counter(neg_adj))

for key in adj_both_polarity:
    pos_adj_occ = pos_adj_dict[key]
    neg_adj_occ = neg_adj_dict[key]

    if pos_adj_occ >= neg_adj_occ:
        del neg_adj_dict[key]
    else:
        del pos_adj_dict[key]

pos_adj = list(pos_adj_dict.keys())
neg_adj = list(neg_adj_dict.keys())

print(pos_adj)
print(neg_adj)


# --------------------------------------------------------
# ------------------ Methods --------------------------------------

def detect_polarity_of_review(pos_count, neg_count):
    if pos_count >= neg_count:
        return 'positive'
    else:
        return 'negative'


def count_polarity_words_of_review(tokens_list):
    pos_count = 0
    neg_count = 0
    for word in tokens_list:
        if word in subjectivity_lexicon.keys():
            if subjectivity_lexicon[word] == 'positive':
                pos_count += 1
            elif subjectivity_lexicon[word] == 'negative':
                neg_count += 1
    return pos_count, neg_count


def calculate_accuracy_for_reviews_classification(reviews_polarity_list, polarity_type):
    count = 0
    for polarity in reviews_polarity_list:
        if polarity == polarity_type:
            count += 1
    return (count / len(reviews_polarity_list)) * 100


def extra_features_added(data_arr, polarity_count_for_data, negation_count_for_data):
    data_arr = np.append(data_arr, polarity_count_for_data, axis=1)
    data_arr = np.append(data_arr, negation_count_for_data, axis=1)
    return csr_matrix(data_arr)


def count_negations_in_review(tokens_list):
    neg_count = 0
    negations_list = ['no', 'not', 'never']
    for tk in tokens_list:
        if tk in negations_list or 'n\'t' in tk:
            neg_count += 1
    return neg_count


# -----------------------------------------------------------------
# b) Implement a classifier that simply counts whether there are more positive or negative words in a review
# Create subjectivity lexicon
subjectivity_lexicon = defaultdict(lambda: None)
file = open("data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff", "r")
for line in file:
    tokens = nltk.word_tokenize(line.rstrip())
    dict_of_tokens = dict(token.split('=', 1) for token in tokens)
    subjectivity_lexicon[dict_of_tokens['word1']] = dict_of_tokens['priorpolarity']

positive_reviews_file = open("data/rt-polaritydata/rt-polarity-pos.txt", "r")
negative_reviews_file = open("data/rt-polaritydata/rt-polarity-neg.txt", "r")
pos_rev_file_polarities = []
neg_rev_file_polarities = []

# Build data for next task
allreviews = []
labels = []
polarity_words_count = []
negations_count = []

# both files have the same number of lines so they can be parsed in the same for loop
for line_pos, line_neg in zip(positive_reviews_file, negative_reviews_file):
    # tokenise positive review line
    pos_rev_tokens = nltk.word_tokenize(line_pos.rstrip())
    # count how many positive words and how many negative words has the line
    pos_words_line, negative_words_line = count_polarity_words_of_review(pos_rev_tokens)
    # save count to be used later for th feature vector
    polarity_words_count.append((pos_words_line, negative_words_line))
    # save polarity type for the line for baseline approach
    pos_rev_file_polarities.append(detect_polarity_of_review(pos_words_line, negative_words_line))
    # add the line to the data array
    allreviews.append(line_pos.rstrip())
    # add the label of the line
    labels.append(0)
    # save number of negations
    negations_count.append(count_negations_in_review(pos_rev_tokens))

    # tokenise positive negative line
    neg_rev_tokens = nltk.word_tokenize(line_neg.rstrip())
    # count how many positive words and how many negative words has the line
    pos_words_line, negative_words_line = count_polarity_words_of_review(neg_rev_tokens)
    # save count to be used later for th feature vector
    polarity_words_count.append((pos_words_line, negative_words_line))
    # save polarity type for the line for baseline approach
    neg_rev_file_polarities.append(detect_polarity_of_review(pos_words_line, negative_words_line))
    # add the line to the data array
    allreviews.append(line_neg.rstrip())
    # add the label of the line
    labels.append(1)
    # save number of negations
    negations_count.append(count_negations_in_review(neg_rev_tokens))

acc_pos = calculate_accuracy_for_reviews_classification(pos_rev_file_polarities, 'positive')
acc_neg = calculate_accuracy_for_reviews_classification(neg_rev_file_polarities, 'negative')

print('Accuracy for positive reviews: %.2f%% ' % acc_pos)
print('Accuracy for negative reviews: %.2f%%' % acc_neg)
print('Overall accuracy of baseline approach: %.2f%%' % ((acc_pos + acc_neg) / 2))

# ---------------------------------------------------------------------------------------------------------------
# Build a machine-learning sentiment classifier using a simple bag-of-words approach.
# Expand the feature set to include other possible features


# Build bow from reviews
vectorizer = CountVectorizer()
data = vectorizer.fit_transform(allreviews)
labels = np.array(labels)
polarity_words_count = np.array(polarity_words_count)
negations_count = np.array(negations_count).reshape(len(negations_count), 1)

kf = KFold(n_splits=5)

acurracies = []
for train_index, test_index in kf.split(data):
    training_data, testing_data = data[train_index], data[test_index]
    labels_training, labels_testing = labels[train_index], labels[test_index]
    training_polarty_counts, testing_polarty_counts = polarity_words_count[train_index], polarity_words_count[
        test_index]
    training_negations_counts, testing_negations_counts = negations_count[train_index], negations_count[test_index]

    training_data_extra_features = extra_features_added(training_data.toarray(), training_polarty_counts,
                                                        training_negations_counts)
    testing_data_extra_features = extra_features_added(testing_data.toarray(), testing_polarty_counts,
                                                       testing_negations_counts)

    # Training model
    model = LogisticRegression(max_iter=1200000)
    model.fit(training_data_extra_features, labels_training)

    # Predict the rest of data
    predicted_labels = model.predict(testing_data_extra_features)

    acc = accuracy_score(predicted_labels, labels_testing)
    acurracies.append(acc)
    print("Accuracy of classifier: %.2f%%" % (acc * 100))

print("Overall accuracy of classifier: %.2f%%" % (sum(acurracies) / len(acurracies) * 100))
