# TODO: shuffle
import tensorflow as tf
import csv
import nltk
import numpy as np
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold

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

    return positive_examples, negative_examples, corpus


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



train_examples = []
train_labels = []
for negative_example in negative_examples:
    try:
        train_examples.append(negative_example)
        train_labels.append(0)
    except:
        pass


for positive_example in positive_examples:
    try:
        train_examples.append(positive_example)
        train_labels.append(1)
    except:
        pass

def get_model():
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

    for example, label in train_dataset.take(1):
        print('text: ', example.numpy())
        print('label: ', label.numpy())

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    for example, label in train_dataset.take(1):
        print('texts: ', example.numpy()[:3])
        print()
        print('labels: ', label.numpy()[:3])

    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    return model

kfold = KFold(n_splits=5, shuffle=True)
kfold_split = kfold.split(train_examples, train_labels)

train_examples = np.array(train_examples)
train_labels = np.array(train_labels)

step = 0
accuracies = []
# iterating through the different splits
for curr_train, curr_test in kfold_split:
    step += 1
    model = get_model()
    # training the model
    model.fit(train_examples[curr_train], train_labels[curr_train], batch_size=32, epochs=15)
    # getting the evaluation results
    results = model.evaluate(train_examples[curr_test], train_labels[curr_test])
    # printing the loss and accuracy
    print(f'Step {step}: Loss - {results[0]}, Accuracy - {results[1]}')
    accuracies.append(results[1])
    # writing the results to an output file
with open('results.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    for accuracy in accuracies:
        writer.writerow(str(accuracy))

"""
import matplotlib.pyplot as plt


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

for example, label in train_dataset.take(1):
  print('text: ', example.numpy())
  print('label: ', label.numpy())

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
for example, label in train_dataset.take(1):
  print('texts: ', example.numpy()[:3])
  print()
  print('labels: ', label.numpy()[:3])

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())
vocab[:20]

encoded_example = encoder(example)[:3].numpy()

print(encoded_example)

for n in range(3):
  print("Original: ", example[n].numpy())
  print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
  print('\n')


model = get_model()

print([layer.supports_masking for layer in model.layers])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.show()

real_predictions = model.predict(train_examples)
predictions = [int(real_prediction >= 0.0) for real_prediction in real_predictions]
print(predictions)
zeros = 0
ones = 0
for prediction in predictions:
    ones += prediction
    zeros += (1-prediction)

print(f"Zeros={zeros}, Ones={ones}")
zeros = 0
ones = 0
for label in train_labels:
    ones += int(label)
    zeros += (1-int(label))

print(f"Real zeros={zeros}, Real ones={ones}")
results = model.evaluate(test_dataset)
# printing the loss and accuracy
print(f'Results: Loss - {results[0]}, Accuracy - {results[1]}')
"""
