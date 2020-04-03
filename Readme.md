#### Archive:
The coursework2 archive contains the following: 
* a data folder where the corpus are held 
* The formatted output from the two tools from Part1: organisations_nltk.txt and organisations_snlp.txt
* The output from Part 2a with the bootstrapped lexicon: positive_lexicon.txt and negative_lexicon.txt
* file task1.py which contains Part 1 of the coursework 
* file task2.py which contains Part 2 of the coursework 

#### Dependencies:
* python 3
* nltk library
* sklearn library
* stanford NLP tool

#### How to run the coursework:
* go into coursework1 folder: `cd coursework2`
* Before running part 1, go to stanford-corenlp-full-2018-10-05 folder and run the following command to start the server:

``java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 50000`
``
* to run Part 1: `python3 task1.py `
* to run Part 2:  `python3 task02.py `