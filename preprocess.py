# ----------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def stopwds(stopword_filename):
    """read a stopword file into a list, as well as including nltk.corpus.stopwords
       return a list of stop words to remove"""
    with open(stopword_filename,"r") as f:
        sw = f.read().split("\n")
    stop_words = stopwords.words('english')
    stop_words.extend(sw)
    return stop_words

def tokenize(sentences, remove_stpwd = True):
    """tokenize, remove stop words, and punctuations
       input: raw text
       return: a list of preprocessed tokens, like ['human', 'interface', 'computer']"""
    if type(sentences) == str:
        sentences = [sentences]
    token_sent = []
    stopwordlist = stopwds("./data/stopwords.txt")
    for sentence in sentences: 
        tokens = word_tokenize(sentence)

        # remove tokens that are neither word nor numbers
        tokens = [token for token in tokens if token[0].isalnum() and token[-1].isalnum()]
    
        # remove stop words: must
        if remove_stpwd:
            tokens = [token.lower() for token in tokens if token.lower() not in stopwordlist]
        token_sent.append(tokens)
    assert(len(sentences) == len(token_sent))
    if len(token_sent) == 1:
        return token_sent[0]
    else:
        return token_sent