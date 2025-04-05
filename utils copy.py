import nltk
from nltk.stem.porter import PorterStemmer
import json
import numpy as np
from nltk.tokenize import word_tokenize
import string

stemmer = PorterStemmer()

# Load the car names from the cars.json file to recognize car models
with open("cars.json") as f:
    car_data = json.load(f)

car_names = list(car_data["cars"].keys())

# this should match your actual car names

print(car_names)


def tokenize(sentence):
    """
    Tokenize sentence and check for car names in the sentence
    """
    # Tokenize the sentence using word_tokenize from nltk
    words = word_tokenize(sentence)

    # Debug: Print tokenized words before filtering
    #print("Tokenized words:", words)

    # Remove punctuation from the tokenized words
    words = [word.lower() for word in words if word not in string.punctuation]

    # Debug: Print filtered words (after removing punctuation)
    #print("Filtered words:", words)

    # Check for car names in the filtered words
    recognized_cars = [
        car for car in car_names if car.lower() in " ".join(words).lower()
    ]

    # Debug: Print the recognized cars
    #print("Recognized cars:", recognized_cars)

    return words, recognized_cars


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())  # Stem each word


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


# Testing tokenize function with a test sentence
sentence = "Tell me about the Toyota Corolla."
tokenized_sentence, recognized_cars = tokenize(sentence)

# Output the tokenized words and recognized cars
#print("Tokenized Sentence:", tokenized_sentence)
#print("Recognized Cars:", recognized_cars)
