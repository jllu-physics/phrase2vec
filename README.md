## Phrase2Vec

Word embedding methods like Word2Vec and GloVe are extremely fast and efficient tools for many NLP tasks. However, they cannot account for phrases. The meaning of a phrase is not equal to simple sum of component words. For example, "baby back ribs" is a dish, it has nothing to do with infant (for "baby"), or turning around (for "back").

This repository implements an extension to Word2Vec to account for phrases. It can automatically identify phrases without annotation and subsequently learn vector embeddings for the phrases.

### Example

Consider learning phrases (including various dishes) from Yelp review dataset, which contains 650k reviews.

```python
from datasets import load_dataset
import format_yelp
import phrase2vec as pv
from PrefixTokenizer import Prefix_Tokenizer

# loading data and tokenization
dataset = load_dataset("yelp_review_full")
raw_train_corpus = dataset['train']['text']
train_corpus = format_yelp.format_dataset(raw_train_corpus)
train_word_corpus = pv.tokenize(train_corpus)

# build vocabulary
vocab = pv.build_vocabulary(train_word_corpus, max_n=8, 
                            freq_threshold = 1e-6, 
                            score_threshold = 100)
                            
# re-parse text into words and phrases
phrase_tokenizer = Prefix_Tokenizer(vocab)
train_phrase_corpus = phrase_tokenizer.tokenize_dataset(train_word_corpus)

# learn embeddings
phrase_vectors = learn_embedding(train_phrase_corpus)

# use embedding for similarity search
phrase_vectors.most_similar('lamb_vindaloo', topn=10)
```

### Features

#### Improved Semantic Understanding

Extending Word2Vec to phrases greatly improves the semantic understanding of the model. This can be seen by comparing the results below from similarity search.

| query                            | phrase2vec           | word2vec                    |
|----------------------------------|----------------------|-----------------------------|
| Sam's Club                       | Costco               | Player's Club               |
| Michael Jackson                  | Britney Spears       | Michael Kors                |
| lamb vindaloo                    | chicken tikka masala | lamb                        |
| avoid this place like the plague | stay away            | this place like the plague  |


This increased semantic capability is also reflected in sentiment analysis. On Yelp reviews, Phrase2Vec leads to 2\% higher accuracy than Word2Vec.

For more demonstration on the semantic understanding, as well as the linear substructure within region-dish pair, please refer to phrase2vec.pdf.

#### High Quality Phrases

By introducing a new scoring criterion based on association mining, we can now learn much higher quality phrases (such as "avoid this place like the plague"). This is especially true for long phrases, which previous methods usually fail to discover.

#### Efficient Implementation

By using Lossy Counting, we can now search all n-grams for potential phrases. The exact choice of n is no longer a bottleneck.
