
# coding: utf-8

#harshel Jain
#hxj170009

import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from string import punctuation


stop_words = stopwords.words('english')+ list(punctuation)
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
#print(df.target_names.unique())


def preprocess(d):
    d = [re.sub('\S*@\S*\s?', '', sent) for sent in d]
    d = [re.sub('\s+', ' ', sent) for sent in d]
    d = [re.sub("\'", "", sent) for sent in d]
    return d

data = preprocess(df.content.values.tolist())

#pprint(data[:1])

def sentence2word(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

words = list(sentence2word(data))

#print(words[:1])

#bigram and trigram models
bigram = gensim.models.Phrases(words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

words_nostops = remove_stopwords(words)
words_bigrams = make_bigrams(words_nostops)
nlp = spacy.load('en', disable=['parser', 'ner'])
words_lemmatized = lemmatization(words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print(words_lemmatized[:1])

# Vocab
vocab = corpora.Dictionary(words_lemmatized)

# Term Document Frequency
bow_data = [vocab.doc2bow(text) for text in words_lemmatized]


# View
#print(bow_data[:1])

# Gensim LDA model
gensim_lda = gensim.models.ldamodel.LdaModel(corpus=bow_data,id2word=vocab,num_topics=10,random_state=50,
                                           update_every=1,chunksize=300,passes=5,alpha='auto',per_word_topics=True,
                                           minimum_phi_value=0.03)

#Mallet LDA Model
mallet_path = r"/Users/harshel/Downloads/mallet/bin/mallet"
mallet_lda = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_data, num_topics=10, id2word=vocab,iterations=5)

#Topics for Gensim LDA
#pprint(gensim_lda.print_topics())


#Gensim LDA
gensim_perplexity = gensim_lda.log_perplexity(bow_data)
#print('\nPerplexity: ', gensim_perplexity )

#Coherence Score
coherence_gensim_lda = CoherenceModel(model=gensim_lda, texts=words_lemmatized, dictionary=vocab, coherence='c_v')
coherence_gensim = coherence_gensim_lda.get_coherence()
#print('\nCoherence Score: ', coherence_gensim)


#Mallet LDA
#Coherence Score
coherence_mallet_lda = CoherenceModel(model=mallet_lda, texts=words_lemmatized, dictionary=vocab, coherence='c_v')
coherence_mallet = coherence_mallet_lda.get_coherence()
#print('\nCoherence Score: ', coherence_mallet)


def gensim_lda_dom_topic(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def mallet_lda_dom_topic(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

#Gensim dominant topics
df_topic_keywords_gensim = gensim_lda_dom_topic(ldamodel=gensim_lda, corpus=bow_data, texts=data)

# Format
df_dominant_topic_gensim = df_topic_keywords_gensim.reset_index()
df_dominant_topic_gensim.columns = ['Document Number', 'Dominant Topic', 'Percent Contribution', 'Keywords', 'Text']

# Show
df_dominant_topic_gensim.head(100)


#Mallet dominant topics
df_topic_keywords_mallet = mallet_lda_dom_topic(ldamodel=mallet_lda, corpus=bow_data, texts=data)

# Format
df_dominant_topic_mallet = df_topic_keywords_mallet.reset_index()
df_dominant_topic_mallet.columns = ['Document Number', 'Dominant Topic', 'Percent Contribution', 'Keywords', 'Text']

# Show
df_dominant_topic_mallet.head(10)


#GENSIM
# Group top 5 sentences under each topic
sent_topics_sorteddf_gensim = pd.DataFrame()

sent_topics_outdf_grpd_gensim = df_topic_keywords_gensim.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd_gensim:
    sent_topics_sorteddf_gensim = pd.concat([sent_topics_sorteddf_gensim,
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                            axis=0)

# Reset Index
sent_topics_sorteddf_gensim.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_gensim.columns = ['Topic Number', "Percent Contribution", "Topic Keywords", "Document"]



# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_keywords_mallet.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                            axis=0)

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic Number', "Percent Contribution", "Topic Keywords", "Document"]



writer_orig = pd.ExcelWriter('top5sent_mallet.xlsx', engine='xlsxwriter')
sent_topics_sorteddf_mallet.to_excel(writer_orig, index=False, sheet_name='top5sent_mallet')
writer_orig.save()

writer_orig = pd.ExcelWriter('top5sent_gensim.xlsx', engine='xlsxwriter')
sent_topics_sorteddf_gensim.to_excel(writer_orig, index=False, sheet_name='top5sent_gensim')
writer_orig.save()

writer_orig = pd.ExcelWriter('dominant_topic_gensim.xlsx', engine='xlsxwriter')
df_dominant_topic_gensim.to_excel(writer_orig, index=False, sheet_name='dominant_topic')
writer_orig.save()

writer_orig = pd.ExcelWriter('dominant_topic_mallet.xlsx', engine='xlsxwriter')
df_dominant_topic_mallet.to_excel(writer_orig, index=False, sheet_name='dominant_topic')
writer_orig.save()


#Mallet
topic_counts_m = df_topic_keywords_mallet['Dominant_Topic'].value_counts()

total_m = sum(topic_counts_m)

new_topic_count_m = topic_counts_m/total_m

topic_contribution_m = pd.DataFrame(new_topic_count_m)

topic_contribution_m.reset_index(inplace=True)
topic_contribution_m.set_axis(['Topic','Share in corpus'], axis=1)
topic_contribution_m.set_index('Topic')

#Gensim
topic_counts_s = df_topic_keywords_gensim['Dominant_Topic'].value_counts()

total_s = sum(topic_counts_s)

new_topic_count_s = topic_counts_s/total_s

topic_contribution_s = pd.DataFrame(new_topic_count_s)

topic_contribution_s.reset_index(inplace=True)
topic_contribution_s.set_axis(['Topic','Share in corpus'], axis=1)
topic_contribution_s.set_index('Topic')

print("The topic share for the corpus for Gensim Model:")
print("\n")
print('\nPerplexity: ', gensim_perplexity )
print("\n")
print('\nCoherence Score: ', coherence_gensim)
print("\n")
print(topic_contribution_s)


print("\n")
print("\n")

print("The topic share for the corpus for Mallet Model:")
print("\n")
print('\nCoherence Score: ', coherence_mallet)
print("\n")
print(topic_contribution_m)
