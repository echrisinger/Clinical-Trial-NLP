import nltk
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from dataset import get_Xy
from nltk.stem import *

from utils import stem_tokens
import config

def tfidf_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    filtered = [w for w in tokens if not w in nltk.corpus.stopwords.words('english')]
    stems = stem_tokens(filtered, PorterStemmer())
    return stems

def apply_tfidf():
    pfiles, y = get_Xy()
    tfidf = TfidfVectorizer(tokenizer=tfidf_tokenizer, stop_words='english')
    for label in range(config.NUM_LABELS):
        pos_pfiles, pos_y = zip(*[(pfile, y[i]) for i, pfile in enumerate(pfiles) if y[i][label] == 1])
        neg_pfiles, neg_y = zip(*[(pfile, y[i]) for i, pfile in enumerate(pfiles) if y[i][label] == -1])
        
        pos_text = [entry.text for entry in pfile.entries for pfile in pos_pfiles]
        pos_tfs = tfidf.fit_transform(pos_text)

        neg_text = [entry.text for entry in pfile.entries for pfile in neg_pfiles]
        neg_tfs = tfidf.fit_transform(neg_text)

        
def main():
    apply_tfidf()

if __name__ == "__main__":
    main()
