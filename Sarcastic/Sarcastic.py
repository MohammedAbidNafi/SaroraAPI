import json
import pickle
import random

import nltk
import numpy
from nltk.stem import LancasterStemmer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, model_from_json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')

stemmer = LancasterStemmer()

with open("Sarcastic/intents.json") as file:
    data = json.load(file)

try:
    with open("Sarcastic.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)

except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("Sarcastic.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)

try:
    json_file = open('Sarcastic.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    myChatModel = model_from_json(loaded_model_json)
    myChatModel.load_weights("Sarcastic.h5")
    print("Loaded model from disk")

except FileNotFoundError:
    # Make our neural network
    myChatModel = Sequential()
    myChatModel.add(Dense(8, input_shape=[len(words)], activation='relu'))
    myChatModel.add(Dense(len(labels), activation='softmax'))

    # optimize the model
    myChatModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    myChatModel.fit(training, output, epochs=5000, batch_size=20)

    # serialize model to JSON and save it to disk
    model_json = myChatModel.to_json()
    with open("Sarcastic.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    myChatModel.save_weights("Sarcastic.h5")
    print("Saved model to disk")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chatWithSarcastic(inputText):
    currentText = bag_of_words(inputText, words)
    currentTextArray = [currentText]
    numpyCurrentText = numpy.array(currentTextArray)

    if numpy.all((numpyCurrentText == 0)):
        return "I didn't get that, try again"

    result = myChatModel.predict(numpyCurrentText.reshape(1, -1))  # Reshape to represent a batch of size 1
    result_index = numpy.argmax(result, axis=-1)
    tag = labels[result_index[0]]

    if result[0][result_index] > 0.5:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return random.choice(responses)

    else:
        return "I didn't get that, try again"
    
def chat():
    print("Start talking with the chatbot (try quit to stop)")

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        print(chatWithSarcastic(inp))

#chat()
