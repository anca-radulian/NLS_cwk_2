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
# https://nlpforhackers.io/named-entity-extraction/
organisations_snlp = []
organisations_nltk = []

for file in corpus_1_ner_a:
    for token, pos, entity in file:
        if 'ORGANIZATION' in entity:
            organisations_nltk.append((entity, token))

for file in corpus_1_ner_b:
    for token, entity in file:
        if entity == 'ORGANIZATION':
            organisations_snlp.append(token)


print(organisations_nltk)
print(organisations_snlp)
