import csv
from keras import layers, Sequential
import numpy as np
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


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

    train_data = []
    train_labels = []
    for curr_row in pcl_rows:
        if int(curr_row[-1]) == 0:
            train_data.append(curr_row[-2])
            train_labels.append(0)

    # ['disabled', 'immigrant', 'homeless', 'poor-families', 'in-need', 'migrant', 'women', 'vulnerable', 'refugee', 'hopeless']

    for curr_row in categories_rows:
        weight = int(curr_row[-1])
        for _ in range(weight):
            train_data.append(curr_row[-3])
            train_labels.append(1)

    return (train_data, train_labels)



(train_data, train_labels) = build_examples()
train_data = np.array(train_data)
train_labels = np.array(train_labels)
y_train = train_labels
y_test = train_labels


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data)


X_train = tokenizer.texts_to_sequences(train_data)
X_test = tokenizer.texts_to_sequences(train_data)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserve
embedding_dim = 50
maxlen = 100

embedding_matrix = create_embedding_matrix('data/glove_word_embeddings/glove.6B.50d.txt',
                                           tokenizer.word_index, embedding_dim)

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=50,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
