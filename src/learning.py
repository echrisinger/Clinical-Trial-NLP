import nltk
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from dataset import get_Xy
from nltk.stem import *

from utils import stem_tokens

def tfidf_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    filtered = [w for w in tokens if not w in nltk.corpus.stopwords.words('english')]
    stems = stem_tokens(filtered, PorterStemmer())
    return stems

def apply_tfidf():
    pfiles, y = get_Xy()
    tfidf = TfidfVectorizer(tokenizer=tfidf_tokenizer, stop_words='english')
    tfs = tfidf.fit_transform([pfiles[0].entries[0].text])

def main():
    apply_tfidf()

if __name__ == "__main__":
    main()
