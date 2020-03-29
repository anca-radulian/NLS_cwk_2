import nltk
from nltk.corpus import PlaintextCorpusReader

# Task 2: Sentiment analysis of movie reviews
# a) Build a sentiment lexicon using a semi-supervised approach by bootstrapping the process, starting from a small
# lexicon of adjectives and corpus 2.

corpus_2_root = '/Users/raduliana/Desktop/git/NLS_cwk_2/data/rt-polaritydata/'
corpus_2 = PlaintextCorpusReader(corpus_2_root, '.*\.txt')
corpus_2_parsed = []
for file_id in corpus_2.fileids():
    corpus_2_parsed += nltk.pos_tag(nltk.word_tokenize(corpus_2.raw(file_id)))

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

neg_adj = set(neg_adj)
pos_adj = set(pos_adj)


# ------------------ Methods --------------------------------------------
# method to check the neighbours for and patter
def check_neighbours_for_and_pattern(current_index, corpus_parsed, polarity_list):
    # check right neighbours
    i = current_index + 1
    while i < len(corpus_parsed):
        if corpus_parsed[i][0] == ',' or corpus_parsed[i][0] == 'and':
            i += 1
            continue
        elif corpus_parsed[i][1] == 'JJ':
            polarity_list.add(corpus_parsed[i][0])
            i += 1
        else:
            break

    # check left neighbours
    j = current_index - 1
    while j > 0:
        if corpus_parsed[j][0] == ',' or corpus_parsed[j][0] == 'and':
            j -= 1
            continue
        elif corpus_parsed[j][1] == 'JJ':
            polarity_list.add(corpus_parsed[j][0])
            j -= 1
        else:
            break
    return polarity_list


# ---------------------------------------------------------------------------


# Populate lexicon with adjectives
for index in range(len(corpus_2_parsed)):
    current_pair = corpus_2_parsed[index]

    # Consider list of adj conjoined by and - pattern: JJ, JJ, JJ,.. ,and JJ
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
                elif index + 2 < len(corpus_2_parsed) and 'RB' in corpus_2_parsed[index + 1][1] and corpus_2_parsed[index + 2][1] == 'JJ':
                    next_jj = corpus_2_parsed[index + 2]

            if not prev_jj == -1 and not next_jj == -1:
                if prev_jj[0] in pos_adj:
                    neg_adj.add(next_jj[0])
                if prev_jj[0] in neg_adj:
                    pos_adj.add(next_jj[0])
                if next_jj[0] in pos_adj:
                    neg_adj.add(prev_jj[0])
                if next_jj[0] in neg_adj:
                    pos_adj.add(prev_jj[0])

    # pattern: JJ but not (RB) JJ - this structure give adj with same polarities
    if current_pair[0] == 'but':
        if index + 2 < len(corpus_2_parsed) and index - 1 > 0:
            prev_jj = -1
            next_jj = -1
            if corpus_2_parsed[index - 1][1] == 'JJ' and corpus_2_parsed[index + 1][0] == 'not':
                prev_jj = corpus_2_parsed[index - 1]
                if corpus_2_parsed[index + 2][1] == 'JJ':
                    next_jj = corpus_2_parsed[index + 2]
                elif index + 3 < len(corpus_2_parsed) and 'RB' in corpus_2_parsed[index + 2][1] and corpus_2_parsed[index + 3][1] == 'JJ':
                    next_jj = corpus_2_parsed[index + 3]

            if not prev_jj == -1 and not next_jj == -1:
                if prev_jj[0] in pos_adj:
                    pos_adj.add(next_jj[0])
                if prev_jj[0] in neg_adj:
                    neg_adj.add(next_jj[0])
                if next_jj[0] in pos_adj:
                    pos_adj.add(prev_jj[0])
                if next_jj[0] in neg_adj:
                    neg_adj.add(prev_jj[0])


print(pos_adj)
print(neg_adj)