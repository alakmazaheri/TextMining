"""
File: text_similarity.py
Name: Ava Lakmazaheri
Date: 10/11/17
Desc: Load, pickle texts from Project Gutenberg
"""
import pickle
import numpy as np
import math
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

all_names = ['tao', 'analects', 'plato', 'aristotle', 'machiavelli', 'spinoza',
'locke', 'hume', 'kant', 'marx', 'mill', 'cousin', 'nietzsche']

num = len(all_names)

all_texts = [' '] * num

def clean(text):
    """
    Removes header and footer text from Gutenberg document
    Input: string
    Output: string
    """
    startidx = text.find(" ***")
    endidx = text.rfind("*** ")
    return text[startidx:endidx]

def load_texts(filename):
    """
    Loads in all books from a .pickle file and stores each as a string element in a list
    Input: none (change to .pickle file name?)
    Output: list of strings
    """
    input_file = open(filename, 'rb')
    reloaded_copy_of_texts = pickle.load(input_file)

    for i in range(num-1):
        all_texts[i] = clean(reloaded_copy_of_texts[i])

def histogram(text):
    """
    Counts occurrences of each word in text
    Input: string
    Output: dict
    """
    d = dict()

    # break giant string of text into list of words
    words = text.split();
    for word in words:
        d[word] = d.get(word, 0) + 1
    return d

def all_unique_words(all_texts):
    """
    Accounts for all unique words in all texts provided, to assist with similarity analysis
    Input: list of strings
    Output: list of strings
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
    Inputs: string, list of strings
    Output: list of values (in this case, floats >= 0)
    """
    v = []
    h = histogram(text)

    for word in wordbank:
        v.append(h.get(word, 0))

    return v

def comp_cos(vec1, vec2):
    """
    Compute the cosine similarity between two vectors
    Inputs: list of floats
    Output: float
    """
    dot_product = np.dot(vec1, vec2)
    norm_1 = np.linalg.norm(vec1)
    norm_2 = np.linalg.norm(vec2)
    cos_val = dot_product / (norm_1 * norm_2)
    if math.isnan(cos_val):
        cos_val = 0
    return cos_val

def similarity():
    """
    Run linguistic similarity analysis on on philosophy texts. Print the raw
    similarity comparisons and plot their relationships spatially.
    """
    wordbank = all_unique_words(all_texts)

    vecs = [[]] * num
    for i in range(num-1):
        vecs[i] = gen_vector(all_texts[i], wordbank)

    sim = np.zeros((num, num))
    for i in range(num-1):
        for j in range(num-1):
            sim[i][j] = comp_cos(vecs[i], vecs[j])
            #print(sim[i][j])

    dissimilarities = 1 - sim
    coord = MDS(dissimilarity='precomputed').fit_transform(dissimilarities)

    plt.scatter(coord[:,0], coord[:,1])

    # Label the points
    for i in range(coord.shape[0]):
        plt.annotate(str(i), (coord[i,:]))

    plt.show()

def sentiment(text):
    """
    Run valence sentiment analysis on text
    Input: string
    Output: dict
    """
    analyzer = SentimentIntensityAnalyzer()
    f = analyzer.polarity_scores(text)
    return f

if __name__ == "__main__":
    load_texts('philtexts2.pickle')
    similarity()
    # for i in range(num-1):
    #     print(all_names[i])
    #     print(sentiment(all_texts[i]))
