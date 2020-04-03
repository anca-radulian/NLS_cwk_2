import nltk
from nltk.corpus import PlaintextCorpusReader
from stanfordcorenlp import StanfordCoreNLP
import json
import logging

# Task 1:  Named-entity recognition
# a) NLTK provides a classifier that has been trained to recognise several types of named entities.
# Use the function nltk.ne_chunk( ) to process corpus 1

# create corpus
corpus_1_root = '/Users/raduliana/Desktop/git/NLS_cwk_2/data/inaugural/'
corpus_1 = PlaintextCorpusReader(corpus_1_root, '.*\.txt')

corpus_1_ner_a = []
for file_id in corpus_1.fileids():
    corpus_1_ner_a.append(nltk.tree2conlltags(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(corpus_1.raw(file_id))))))

# tokenize, pos tagged and apply a nltk classifier to perform NER
#  nltk.ne_chunk() returns nested nltk.tree.Tree object,

# b) Use the Stanford named-entity recogniser  to parse same corpus
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 50000
corpus_1_ner_b = []
snlp = StanfordCoreNLP('http://localhost', port=9000, timeout=50000)
for file_id in corpus_1.fileids():
    corpus_1_ner_b.append(snlp.ner((corpus_1.raw(file_id))))

# c) get the organization entities from previous paths
organisations_snlp = []
organisations_nltk = []

for file in corpus_1_ner_a:
    organisations_file = []
    index_tk = 0
    for token, pos, entity in file:
        if entity == 'B-ORGANIZATION':
            organisations_file.append((token, index_tk))
        if entity == 'I-ORGANIZATION':
            organisations_file[len(organisations_file) - 1] = (
                organisations_file[len(organisations_file) - 1][0] + " " + token,
                organisations_file[len(organisations_file) - 1][1])
        index_tk += 1
    organisations_nltk.append(organisations_file)

for file in corpus_1_ner_b:
    organisations_file = []
    index_tk = 0
    for token, entity in file:
        if entity == 'ORGANIZATION':
            organisations_file.append((token, index_tk))
        index_tk += 1
    organisations_snlp.append(organisations_file)

print(organisations_nltk)
print(organisations_snlp)

fully_matches = []
partial_matches = []
for file_index in range(len(organisations_nltk)):
    for word, index in organisations_nltk[file_index]:
        if (word, index) in organisations_snlp[file_index]:
            fully_matches.append(word)
        else:
            tokens = nltk.word_tokenize(word)
            for tk_index in range(len(tokens)):
                if (tokens[tk_index], index + tk_index) in organisations_snlp[file_index]:
                    partial_matches.append((word, tokens[tk_index]))

print(len(fully_matches))
print(len(partial_matches))
