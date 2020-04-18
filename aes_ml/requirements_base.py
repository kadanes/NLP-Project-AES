#Utils
import re
import pprint
import itertools
import numpy as np
from numpy.linalg import svd
import pandas as pd
from time import time 
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt

#NLP
import nltk
from nltk.corpus import stopwords
import language_check
from spellchecker import SpellChecker
from nltk.tag.perceptron import PerceptronTagger
from gensim.models import word2vec
from nltk.stem.porter import PorterStemmer

#ML
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')