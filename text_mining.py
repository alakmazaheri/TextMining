"""
File: text_mining.py
Name: Ava Lakmazaheri
Date: 10/11/17
Desc: Reads in text of books from a .pickle file, runs word frequency analysis for determining cosine similarity between them.
      Also uses compilation of all books for Markov Text Synthesis!
"""
import math
import random
import pickle
import doctest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Global variables:
all_names = ['Lao-Tze', 'Plato', 'Aristotle', 'Machiavelli', 'Spinoza', 'Locke', 'Hume', 'Kant', 'Marx', 'Mill', 'Cousin', 'Nietzsche']
num = len(all_names)
all_texts = [' '] * num

suffix_map = {}        # map prefixes to list of suffixes
prefix = ()

def load_texts(filename):
    """
    Load in all books from a .pickle file and store each as a string element in a list.

    Note that instead of returning the list, the function saves it to a global variable.

    Args:
        filename: name of .pickle file as a string
    """
    input_file = open(filename, 'rb')
    reloaded_copy_of_texts = pickle.load(input_file)

    for i in range(num):
        all_texts[i] = clean(reloaded_copy_of_texts[i])

def clean(text):
    """
    Cleans text of Project Gutenberg header and footer

    Args:
        text: Gutenberg text as a string

    Returns:
        Book-only text as a string
    """
    #startidx = text.find(" ***")
    startidx = text.find(' ***')
    endidx = text.rfind("*** END")
    return text[startidx:endidx]


def run_similarity():
    """
    Run linguistic similarity analysis on on philosophy texts and plot their relationships spatially using MDS
    """
    wordbank = all_unique_words(all_texts)

    vecs = [[]] * num
    for i in range(num):
        vecs[i] = gen_vector(all_texts[i], wordbank)

    sim = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            sim[i][j] = comp_cos(vecs[i], vecs[j])
            #print(sim[i][j])

    dissimilarities = 1 - sim
    coord = MDS(dissimilarity='precomputed').fit_transform(dissimilarities)

    plt.scatter(coord[:,0], coord[:,1])

    # Label the points
    for i in range(coord.shape[0]):
        plt.annotate(all_names[i], (coord[i,:]))

    plt.show()

def histogram(text):
    """
    Counts occurrences of each word in text
    Args:
        text: book text as string

    Returns:
        dict (key: words in the text, value: frequency of occurence)
    """
    d = dict()

    # break giant string of text into list of words
    words = text.split();
    for word in words:
        d[word] = d.get(word, 0) + 1
    return d

def all_unique_words(all_texts):
    """
    Compiles all unique words that appear in all loaded texts
    Args:
        all_texts: list of all book-texts as strings

    Returns:
        List of all unique words (strings) in all loaded texts -- no repeats!
    """
    allwords = []

    for text in all_texts:
        wordlist = text.split()
        for word in wordlist:
            if(word not in allwords):
                allwords.append(word)
    return allwords

def gen_vector(text, wordbank):
    """
    Generate an n-dimensional vector for word count (where n is the total number of unique words)
    Args:
        text: book text as string
        wordbank: single list of all unique words in all texts

    Returns:
        List of frequencies (numbers >= 0)
    """
    v = []
    h = histogram(text)

    for word in wordbank:
        # If a word does not appear in the text, store it as 0 frequency
        v.append(h.get(word, 0))

    return v

def comp_cos(vec1, vec2):
    """
    Compute the cosine similarity between two vectors
    Args:
        vec1: list of floats (word frequencies) for text 1
        vec2: list of floats (word frequencies) for text 2

    Returns:
        Cosine similarity as float between texts 1 and 2

    >>> comp_cos([1,0], [-1,0])
    -1.0
    """
    dot_product = np.dot(vec1, vec2)
    norm_1 = np.linalg.norm(vec1)
    norm_2 = np.linalg.norm(vec2)

    prod = norm_1 * norm_2

    if prod == 0:    # avoid dviding by 0
        return 0
    return dot_product/prod


def run_sentiments(all_texts):
    """
    Print names of text and their corresponding sentiment analysis
    Args:
        all_texts: list of all book-texts as strings
    """
    for i in range(num-1):
        print(all_names[i])
        print(sentiment(all_texts[i]))

def sentiment(text):
    """
    Run valence sentiment analysis on text
    Args:
        text: book contents as string

    Returns:
        dict of negative, neutral, and positive valence scores
    """
    analyzer = SentimentIntensityAnalyzer()
    f = analyzer.polarity_scores(text)
    return f



def run_markov(n=100):
    """
    Create a giant text file that combines all of the books together. Split it
    up and process each word individually. Then generate a random string of text
    using this processing!
    """
    concatall = ' '.join(all_texts)

    for word in concatall.rstrip().split():
        process_word(word)

    gen_text(n)

def process_word(word, order=2):
    """
    Take each word and add corresponding entries to the Markov dictionary

    Args:
        word: string
        order: integer length of tuple

    """
    global prefix

    if len(prefix) < order:                 # keep adding words until order length is fulfilled
        prefix += (word,)
        return

    try:
        suffix_map[prefix].append(word)     # if there is no entry for this prefix, make one
    except KeyError:
        suffix_map[prefix] = [word]

    prefix = shift(prefix, word)            # increment tuple

def shift(t, word):
    """
    Creates tuple by removing the existing head and adding word to the tail

    Args:
        t: tuple of strings
        word: string

    Returns:
        Tuple of strings
    """
    return t[1:] + (word,)

def gen_text(n):
    """
    Generate sentence(s) of n random words based on the analyzed text.

    Start with a random prefix from the dictionary!

    Args:
        n: number of words to generate
    """
    # Start with a random prefix
    start = random.choice(list(suffix_map.keys()))

    # While there are still words to be written...
    for i in range(n):
        suffixes = suffix_map.get(start, None)      # find an appropriate suffix
        if suffixes == None:
            gen_text(n-i)                           # if it isn't in map, start again!
            return

        word = random.choice(suffixes)              # choose a random suffix
        print(word, end=' ')                        # print the next word
        start = shift(start, word)


if __name__ == "__main__":
    doctest.testmod()

    load_texts('philtexts3.pickle')
    run_markov(100)
    run_similarity()

    # Sentiment analysis didn't end up being as interesting as I hoped, so I left it out of the final analysis!
    #run_sentiments(all_texts)
