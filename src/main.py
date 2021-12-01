import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer = WordNetLemmatizer()
english_stopwords = stopwords.words('english')


def clean_sentence(sentence):
    sentence = sentence.lower()
    for char in sentence:
        if (not char.isalpha()) and char != ' ':
            sentence = sentence.replace(char, ' ')
    tokens = sentence.split()
    good_tokens = []
    for token in tokens:
        if token not in english_stopwords:
            good_tokens.append(lemmatizer.lemmatize(token))
    sentence = ' '.join(good_tokens)
    return sentence


def read_fie_rows(file_path):
    rows = []
    tsv_file = open(file_path)
    tsv_reader = csv.reader(tsv_file, delimiter="\t")

    for row in tsv_reader:
        rows.append(row)
    return rows


def build_examples():
    pcl_rows = read_fie_rows('../dataset/dontpatronizeme_pcl.tsv')[2000:]
    categories_rows = read_fie_rows('../dataset/dontpatronizeme_categories.tsv')[2000:]

    positive_examples = []
    negative_examples = []
    corpus = []
    for curr_row in pcl_rows:
        if int(curr_row[-1]) == 0:
            negative_examples.append(curr_row[-2])
        corpus.append(curr_row[-2])

    # ['disabled', 'immigrant', 'homeless', 'poor-families', 'in-need', 'migrant', 'women', 'vulnerable', 'refugee', 'hopeless']

    for curr_row in categories_rows:
        weight = int(curr_row[-1])
        for _ in range(weight):
            positive_examples.append(curr_row[-3])
        corpus.append(curr_row[-3])

    return (positive_examples, negative_examples, corpus)


(positive_examples, negative_examples, corpus) = build_examples()

from gensim.models import word2vec

tokenized_sentences = [clean_sentence(sentence).split() for sentence in corpus]
model = word2vec.Word2Vec(tokenized_sentences)


def create_model():
    model.wv.save('./model.bin')


create_model()
model.wv.load('./model.bin')

def get_sentence_score(sentence):
    sentence_tokens = sentence.split(" ")
    sentence_score = None
    for token in sentence_tokens:
        try:
            if sentence_score is None:
                sentence_score = model.wv[token]
            else:
                sentence_score = sentence_score + model.wv[token]
        except:
            pass

    sentence_score = sentence_score / len(sentence_tokens)
    return sentence_score


def find_dist(sent1, sent2):
    dist = 0
    for i in range(0, sent1.size):
        dist += (sent1[i] - sent2[i]) * (sent1[i] - sent2[i])
    return dist


test_sentence_score = get_sentence_score(clean_sentence(
    "Mr Little said they would provide better and quicker treatment for those in need"))



train_data = []
train_labels = []
for negative_example in negative_examples:
    try:
        train_data.append(get_sentence_score(clean_sentence(negative_example)))
        train_labels.append(0)
    except:
        pass



for positive_example in positive_examples:
    try:
        train_data.append(get_sentence_score(clean_sentence(positive_example)))
        train_labels.append(1)
    except:
        pass

train_data = np.array(train_data)
train_labels = np.array(train_labels)

print(np.shape(train_data))
print(np.shape(train_labels))
