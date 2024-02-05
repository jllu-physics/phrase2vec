from nltk.tokenize import word_tokenize
from tqdm import tqdm
import string
import math
from multiprocessing import Pool
from LossyCounter import LossyCounter
from gensim.models import Word2Vec, KeyedVectors

def tokenize_single_process(corpus, skip_words = None, uncased = True):
    """
    Tokenize a corpus, omitting words (and symbols) in skip_words,
    optionally lowering the case of the text

    Parameters:
    -----------

        corpus: iterable
            Iterable of strings, with each string being a text.
            For example, ['Hello, world!','This is an example.']
            
        skip_words: list-like, string.punctuation by default
            List or list-like of words and symbols to be omitted
            from tokenization. The default value contains all
            the punctuation marks, like comma.
            It is recommended to use a set for faster look-up.

        uncased: boolean, True by default
            Whether to lower the case for each string in corpus
            before tokenization, making the tokenization case
            insensitive.

    Returns:
    --------

        token_corpus: list
            A list of list, with each sub-list containing the
            tokens from tokenizing the corresponding string in
            corpus
    """
    token_corpus = []
    if skip_words is None:
        skip_words = set(string.punctuation) # set for faster look up
    if uncased: 
        # if case insensitive
        for raw_text in corpus:
            uncased_text = raw_text.lower() 
            # then put string in lower case
            word_list = [word for word in word_tokenize(uncased_text)
                         if word not in skip_words]
            token_corpus.append(word_list)
    else:
        for raw_text in corpus:
            word_list = [word for word in word_tokenize(raw_text)
                         if word not in skip_words]
            token_corpus.append(word_list)
    return token_corpus
    
def tokenize_single_process_wrapper(args):
    """
    Wrapped function for tokenize_single_process. The arguments
    are put into a dictionary.
    
    This is done so that it can be called from imap in 
    multiprocessing. Another option is to use starmap. But
    imap has the advantage of being iterative, and thus can
    work with tqdm to produce a progress bar.
    """
    corpus = args['corpus']
    skip_words = args['skip_words']
    uncased = args['uncased']
    # below should be exactly the same as tokenize_single_process
    # I copied the lines directly
    # This saves one function call
    # For easier maintenance, we could also do
    # return tokenize_single_process(corpus, skip_words, uncased)
    token_corpus = []
    if skip_words is None:
        skip_words = set(string.punctuation)
    if uncased:
        for raw_text in corpus:
            uncased_text = raw_text.lower()
            word_list = [word for word in word_tokenize(uncased_text)
                         if word not in skip_words]
            token_corpus.append(word_list)
    else:
        for raw_text in corpus:
            word_list = [word for word in word_tokenize(raw_text)
                         if word not in skip_words]
            token_corpus.append(word_list)
    return token_corpus

def tokenize(corpus, skip_words = None, uncased = True, chunk_size = 100):
    """
    Parallel tokenization of a corpus.

    Parameters:
    -----------

        corpus: iterable
            Iterable of strings, with each string being a text.
            For example, ['Hello, world!','This is an example.']
            
        skip_words: list-like, string.punctuation by default
            List or list-like of words and symbols to be omitted
            from tokenization. The default value contains all
            the punctuation marks, like comma.
            It is recommended to use a set for faster look-up.

        uncased: boolean, True by default
            Whether to lower the case for each string in corpus
            before tokenization, making the tokenization case
            insensitive.

        chunk_size: int, 100 by default
            Number of strings to be passed to single-process
            function at each time.

    Returns:
    --------

        token_corpus: list
            A list of list, with each sub-list containing the
            tokens from tokenizing the corresponding string in
            corpus
    """
    print("Tokenizing text:")
    L = len(corpus)
    N = math.ceil(L/chunk_size)
    # N is the number of chunks
    result = []
    
    args_list = []
    for i in range(N):
        args = {}
        # put arguments in args dictionary so that we can pass
        # it as one single argument in imap
        args['corpus'] = corpus[i*chunk_size:(i+1)*chunk_size]
        args['skip_words'] = skip_words
        args['uncased'] = uncased
        args_list.append(args)
        # each argument is a job, args_list is like a job queue
        # to be passed to imap

    pool = Pool()
    with tqdm(total = L, position=0, leave=True) as pbar:
        for r in pool.imap(tokenize_single_process_wrapper,args_list):
        # imap is slower than map, but we can have a nice progress bar
        # another option is to distribute tasks to workers manually
            result += r
            pbar.update(len(r))
    pool.close()
    
    return result
    
def get_word_freq(word_corpus, threshold = 1e-6):
    """
    Get the frequent words and their relative frequency.
    Frequent words are those with relative frequency
    beyond threshold

    Parameters:
    -----------

        word_corpus: list
            corpus stored in the following format:
            The corpus itself is a list of lists. 
            Each document in the corpus is in turn 
            a list (or a sub-list), the elements
            of the sublist are tokens, either in
            string or by index.

        threshold: float
            probability threshold (or more exactly, 
            relative frequency threshold) for a word 
            (or token) to be considered frequent word. 
            Only words with frequency higher than 
            this threshold is counted and reported.

    return:
    -------

        result: dictionary
            a dictionary mapping frequent words to
            their relative frequency

    NOTE: In this function, as in counting get_ngram_freq,
    the frequencies are counted approximately, using
    lossy counting algorithm.
    """
    print("Finding frequent words:")
    
    eps = threshold/2 
    # error bound for lossy counter
    w = math.ceil(1/eps)
    # bucket size
    flush_limit = 5*w
    # cache size
    lc = LossyCounter(eps, flush_limit = 5*w, 
                      prune_limit = 1)
    # prune_limit = 1 means prune immediately
    # after flushing

    # ---counting---
    
    for text in tqdm(word_corpus):
        lc.cache(text)
    lc.flush() # in case there are elements in cache
    #lc.prune() # this is not needed as prune_limit=1

    # ---get frequent items---
    
    word_list = lc.getFreqItems(threshold, 'median')
    # word_list is the list of frequent words

    # ---get frequency---
    approx_count = lc.getCounts(word_list, 'median')
    # approximate counts of frequent words
    result = {}
    total = lc.total
    sorted_word_list = sorted(word_list, 
                        key = approx_count.get, 
                        reverse = True)
    for word in sorted_word_list:
        result[word] = approx_count[word]/total
    return result
    
def get_ngram_freq(corpus, n, threshold = 1e-6):
    """
    Get the frequent n-grams and their relative frequency.
    Frequent n-grams are those with relative frequency
    beyond threshold

    Parameters:
    -----------

        word_corpus: list
            corpus stored in the following format:
            The corpus itself is a list of lists. 
            Each document in the corpus is in turn 
            a list (or a sub-list), the elements
            of the sublist are tokens, either in
            string or by index.

        n: int
            length n of n-grams

        threshold: float
            probability threshold (or more exactly, 
            relative frequency threshold) for a word 
            (or token) to be considered frequent word. 
            Only words with frequency higher than 
            this threshold is counted and reported.

    return:
    -------

        result: dictionary
            a dictionary mapping frequent words to
            their relative frequency

    NOTE: In counting the frequent n-grams, the 
    frequencies are counted approximately, using
    lossy counting algorithm.
    """
    print("Finding frequent", n, "grams:")
    eps = threshold/2 
    # error bound for lossy counter
    w = math.ceil(1/eps)
    # bucket size
    flush_limit = 5*w
    # cache size
    lc = LossyCounter(eps, flush_limit = 5*w, 
                      prune_limit = 1)
    # prune_limit = 1 means prune immediately
    # after flushing
    L = len(corpus)
    for i in tqdm(range(L)):
        text = corpus[i]
        T = len(text)
        ngrams = ['_'.join(text[i:i+n]) for i in range(T-n+1)]
        lc.cache(ngrams)
    lc.flush()
    #lc.prune()
    ngram_list = lc.getFreqItems(threshold, 'median')
    approx_count = lc.getCounts(ngram_list, 'median')
    result = {}
    total = lc.total
    sorted_ngram_list = sorted(ngram_list, 
                        key = approx_count.get, 
                        reverse = True)
    for ngram in sorted_ngram_list:
        result[ngram] = approx_count[ngram]/total
    return result
    
def get_baseline_prob(phrase, ngram_freq, e):
    """
    Find baseline probability of a phrase candidate.
    Baseline probability is the maximum value one 
    can get if one split the phrase into two and
    multiply the probability of the two parts. i.e.
    
    max_{k} Prob(phrase_{:k})*Prob(phrase_{k:})

    Parameters:
    -----------

        phrase: string
            phrase candidate, in the format of tokens
            split by "_" (i.e. underline)

        ngram_freq: dictionary
            dictionary mapping ngram to relative freq

        e: float
            the default probability if a n-gram is
            missing from ngram_freq

    Returns:
    --------

        baseline: float
            baseline probability

    Comment:
    --------

        Baseline probability can be understood as the
        maximum possible probability if the phrase
        candidate is actually not a phrase, but more
        than two phrases or words concatenated together.
        In this case, the probability they appear
        together is roughly the product of their
        probability.

        There are multiple choices for e. One possibility
        is to choose the probability of the phrase, since
        the probability observing a part of the phrase
        cannot be lower than observing the whole phrase.
        However, in finite dataset, this may not hold
        exactly, because there are slightly less long
        n-grams than shorter ones.
        Another possibility is to use the threshold for
        a n-gram to be frequent. If a n-gram is missing,
        it is likely that it is slightly lower in prob
        than threshold.
    """
    words = phrase.split('_')
    # list of words (or tokens) in the phrase
    n = len(words)
    # len of n-gram
    baseline = 0
    # baseline probability
    # we will update whenever we find a split can lead
    # to higher value, so we will set it to 0 in the
    # beginning
    for i in range(1,n):
        part1 = '_'.join(words[:i])
        # first half of the split
        if part1 not in ngram_freq and i > 1:
            continue
        prob1 = ngram_freq.get(part1, e)
        part2 = '_'.join(words[i:])
        # second half
        if part2 not in ngram_freq and i < n-1:
            continue
        prob2 = ngram_freq.get(part2, e)
        prob = prob1*prob2
        if prob > baseline:
            baseline = prob
    return baseline
    
def expand_vocab(vocab, ngram_freq, e, threshold = 100):
    """
    Expand vocabulary based on scores of the n-grams.
    A n-gram is added to the vocabulary, if its score
    if above threshold.

    Parameters:
    -----------

        vocab: set
            set of vocabulary to be expanded

        ngram_freq: dictionary
            Dictionary mapping frequent n-grams to their
            frequencies

        e: float
            default probability (or more exactly, relative
            frequency) if ngram is not in ngram_freq

        threshold: int
            minimum score for a n-gram to be considered
            a phrase and added to the vocabulary

    Returns:
    --------

        Nothing. 
        This function modifies vocab argument directly
    """
    print("Building vocabulary:")
    for ngram in tqdm(ngram_freq):
        if ngram in vocab:
            continue
        baseline = get_baseline_prob(ngram, ngram_freq, e)
        if baseline == 0:
            continue
        if ngram_freq[ngram] > threshold * baseline:
            vocab.add(ngram)
            
def build_vocabulary(word_corpus, max_n = 8, 
                     freq_threshold = 1e-6, 
                     score_threshold = 100):
    """
    Build a vocabulary of words and phrases 
    from word-tokenized corpus

    Parameters:
    -----------

        word_corpus: list
            corpus stored in the following format:
            The corpus itself is a list of lists. 
            Each document in the corpus is in turn 
            a list (or a sub-list), the elements
            of the sublist are tokens, either in
            string or by index.

        max_n: int, default 8
            maximal length of n-grams to be considered
            for phrase

        freq_threshold: float, default 1e-6
            relative frequency threshold for a n-gram
            to be considered for being a phrase 
            Only words and n-grams with frequency 
            higher than this threshold would appear
            in vocabulary.

        score_threshold: float, default 100
            minimal score for a frequent n-gram to be
            considered a phrase and added to vocabulary

    Returns:
    --------

        vocab: set
            a set of words and phrases learned from
            corpus
    """
    word_freq = get_word_freq(word_corpus, 
                              freq_threshold)
    ngram_freq_pool = word_freq.copy()
    for n in range(2,max_n+1):
        ngram_freq = get_ngram_freq(word_corpus, 
                                    n, freq_threshold)
        ngram_freq_pool = ngram_freq_pool | ngram_freq
    vocab = set([k for k in word_freq])
    expand_vocab(vocab, ngram_freq_pool, 
                 freq_threshold, score_threshold)
    return vocab
    
def learn_embedding(phrase_corpus, dim = 300, window = 5, 
                    epochs = 20, workers = 12):
    """
    Learn word2vec-like CBOW embeddings for the phrases

    Parameters:
    
        phrase_corpus: list
            corpus stored in the following format:
            The corpus itself is a list of lists. 
            Each document in the corpus is in turn 
            a list (or a sub-list), the elements
            of the sublist are tokens, either in
            words or phrases.

        dim: int
            the dimensionality of vector embedding

        window: int
            window length for CBOW

        epochs: int
            number of epochs to run training

        workers: int
            number of workers to use for parallel
            training

    Returns:

        p2v: gensim.KeyedVectors
            mapping phrases (or words) to embedding
            vectors
    """
    model = Word2Vec(sentences=phrase_corpus, vector_size=dim, 
                     window=window, min_count=1, 
                     workers=workers, epochs = epochs)
    phrase_vectors = model.wv
    phrase_vectors.save("phrase2vec.wordvectors")
    p2v = KeyedVectors.load("phrase2vec.wordvectors", mmap='r')
    return p2v
