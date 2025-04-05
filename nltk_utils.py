import nltk
from nltk.stem.porter import PorterStemmer
import json
import numpy as np

# Initialize the stemmer
stemmer = PorterStemmer()

# Load the car names from the cars.json file to recognize car models
with open("cars.json") as f:
    car_data = json.load(f)

car_names = list(car_data["cars"].keys())  # Extract car names from JSON data


def tokenize(sentence):
    """
    Tokenize sentence and check for car names in the sentence.
    Also, return recognized car names separately for further handling.
    """
    words = nltk.word_tokenize(sentence)
    sentence_words = [word.lower() for word in words]  # Convert words to lowercase
    recognized_cars = [
        car for car in car_names if car.lower() in sentence_words
    ]  # Detect car names in sentence
    return sentence_words, recognized_cars


def stem(word):
    """
    Stemming = find the root form of the word.
    Example: words = ["organize", "organizes", "organizing"] -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())  # Stem each word


def bag_of_words(tokenized_sentence, words):
    """
    Return a bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.

    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag_of_words = [0, 1, 0, 1, 0, 0, 0]
    """
    sentence_words = [
        stem(word) for word in tokenized_sentence
    ]  # Stem each word in the sentence
    bag = np.zeros(len(words), dtype=np.float32)  # Initialize bag with 0 for each word
    for idx, w in enumerate(words):
        if w in sentence_words:  # Check if the word is in the sentence
            bag[idx] = 1  # Mark it as present
    return bag
