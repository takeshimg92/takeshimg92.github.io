```python
from nltk.corpus import inaugural
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

rows = []

for fileid in inaugural.fileids():
  year = int(fileid[:4])
  president = fileid[5:-4]
  words = ' '.join(list(inaugural.words(fileid)))
  rows.append([year, president, words])

speech_df = pd.DataFrame(rows, columns=['year', 'president', 'speech'])

vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tf_idf_speeches = vectorizer.fit_transform(speech_df['speech'])
tf_idf_speeches_df = pd.DataFrame(tf_idf_speeches.todense(), columns=vectorizer.get_feature_names())

nmf = NMF(n_components=5)
nmf_speeches = nmf.fit_transform(tf_idf_speeches_df.values)

word_component_df = pd.DataFrame(nmf.components_.T)
word_component_df['words'] = vectorizer.get_feature_names()
word_component_df['component_max'] = np.argmax(nmf.components_.T, axis=1)

distance_matrix = cosine_similarity(tf_idf_speeches)
```
# This the new president usually sounds like the old president

Look at an example

Look at distance president-over-president

# ?

# How should we interpret these results, exactly?

## This is not a demonstration that specific latent factors definitely exist

You interpret these at your own risk

## This does tell us about which covariates are correlated


## Factors may have testable implications
