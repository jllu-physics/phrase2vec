import pygtrie
from tqdm import tqdm

class Prefix_Tokenizer:
    """
    Prefix tokenizer parses text into phrases and words
    in the vocabulary, or symbol for unknown.

    Arguments:
    ----------

        vocab: set
            set of words and phrases within vocabulary,
            assumes phrases are in the format of wors
            separated by underlines ('_')

        unknown_symbol: string, default 'UNKNOWN'
            what to put if the next word in not in the 
            vocabulary

    Attributes:
    -----------

        t: pygtrie.Trie
            a trie (or prefix tree) for prefix parsing

        unknown_symbol: string
            memorizes unknown_symbol in argument

    Methods:
    --------

        tokenize(self, text):
            group a list of words into phrases

        tokenize_dataset(self, dataset):
            group multiple lists of words into phrases
    """
    def __init__(self, vocab, unknown_symbol = 'UNKNOWN'):
        self.t = pygtrie.Trie()
        for c in vocab:
            self.t[c.split('_')] = c
        self.unknown_symbol = unknown_symbol
    def tokenize(self, text):
        """
        Group a list of words into phrases

        Parameters:
        -----------

            text: list
                a list of words, each word represented by
                a string

        Returns:
        --------

            result: list
                a list of words and phrases
        """
        result = []
        i = 0
        L = len(text)
        while i < L:
            key, value = self.t.longest_prefix(text[i:])
            if key is None:
                #parsed_text.append(text[i])
                i += 1
                result.append(self.unknown_symbol)
            else:
                #parsed_text.append(key)
                i += len(key)
                result.append(value)
        return result
    def tokenize_dataset(self, dataset):
        """
        Group multiple lists of words into phrases

        Parameters:
        -----------

            dataset: list
                a list of lists, each sub-list being
                a list of words, each word represented by
                a string

        Returns:
        --------

            result: list
                a list of lists, each sub-list being
                a list of words and phrases
        """
        result = []
        longest_prefix = self.t.longest_prefix
        unknown = self.unknown_symbol
        for text in tqdm(dataset):
            tokenized = []
            i = 0
            L = len(text)
            while i < L:
                key, value = longest_prefix(text[i:])
                if key is None:
                    i += 1
                    tokenized.append(unknown)
                else:
                    i += len(key)
                    tokenized.append(value)
            result.append(tokenized)
        return result
