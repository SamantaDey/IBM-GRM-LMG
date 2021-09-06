#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,argparse
import numpy as np
import math
import sys
from fractions import Fraction
import warnings
from collections import Counter
from nltk import ngrams
from __future__ import unicode_literals
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
from builtins import object
from collections import defaultdict
import math
import re

def myngrams(n, sent):
        """Given a sentence, return n-grams of nodes for the given N. Lowercases
        everything if the measure should not be case-sensitive.
        @param n: n-gram 'N' (1 for unigrams, 2 for bigrams etc.)
        @param sent: the sent in question
        @return: n-grams of nodes, as tuples of tuples (t-lemma & formeme)
        """
        return(list(ngrams(sent.split(), n)))

def get_ngram_counts(n, sents):
        """Returns a dictionary with counts of all n-grams in the given sentences.
        @param n: the "n" in n-grams (how long the n-grams should be)
        @param sents: list of sentences for n-gram counting
        @return: a dictionary (ngram: count) listing counts of n-grams attested in any of the sentences
        """
        merged_ngrams = {}

        for sent in sents:
            ngrams1 = defaultdict(int)

            for ngram in myngrams(n, sent):
                ngrams1[ngram] += 1
            for ngram, cnt in ngrams1.items():
                merged_ngrams[ngram] = max((merged_ngrams.get(ngram, 0), cnt))
        return merged_ngrams

def info(ref_ngrams, ngram):
        """Return the NIST informativeness of an n-gram."""
        if ngram not in ref_ngrams[len(str(ngram).split())]:
            return 0.0
        return math.log(ref_ngrams[len(str(ngram).split()) - 1][ngram[:-1]] /
                        float(ref_ngrams[len(str(ngram).split())][ngram]), 2)

def nist_length_penalty(lsys, avg_lref):
        """Compute the NIST length penalty, based on system output length & average reference length.
        @param lsys: total system output length
        @param avg_lref: total average reference length
        @return: NIST length penalty term
        """
        BETA = old_div(- math.log(0.5), math.log(0.67) ** 2)
        ratio = lsys / float(avg_lref)
        if ratio >= 1:
            return 1
        if ratio <= 0:
            return 0
        return math.exp(-BETA * math.log(ratio) ** 2)

def mydiv(x,y):
    if y!=0:
        return x/y
    else:
        return 0

def nist(pred_sent, ref_sents, max_ngram=5):
        """Return the current NIST score, according to the accumulated counts."""
        #BETA = old_div(- math.log(0.5), math.log(1.5) ** 2)
        ref_ngrams = [defaultdict(int) for _ in range(max_ngram + 1)]  # has 0-grams
        # these two don't have 0-grams
        hit_ngrams = [[] for _ in range(max_ngram)]
        cand_lens = [[] for _ in range(max_ngram)]
        avg_ref_len = 0.0
        #print(cand_lens)
        # collect ngram matches
        for n in range(max_ngram):
            cand_lens[n].append(len(pred_sent.split()) - n)  # keep track of output length
            #print(cand_lens)
            merged_ref_ngrams = get_ngram_counts(n + 1, ref_sents)
            pred_ngrams = get_ngram_counts(n + 1, [pred_sent])
            # collect ngram matches
            hit_ngrams1 = defaultdict(int)
            for ngram in pred_ngrams:
                hits = min(pred_ngrams[ngram], merged_ref_ngrams.get(ngram, 0))
                if hits:
                    hit_ngrams1[ngram] = hits
            hit_ngrams[n].append(hit_ngrams1)
            # collect total reference ngram counts
            for ref_sent in ref_sents:
                for ngram in myngrams(n + 1, ref_sent):
                    ref_ngrams[n + 1][ngram] += 1
        
        #print(cand_lens)
        #print(hit_ngrams)
        # ref_ngrams: use 0-grams for information value as well
        ref_len_sum = sum(len(ref_sent.split()) for ref_sent in ref_sents)
        ref_ngrams[0][()] += ref_len_sum
        # collect average reference length
        avg_ref_len += ref_len_sum / float(len(ref_sents))
        
        #print(ref_ngrams)
        # 1st NIST term
        hit_infos = [0.0 for _ in range(max_ngram)]
        for n in range(max_ngram):
            for i in hit_ngrams[n]:
                hit_infos[n] += sum(info(ref_ngrams, ngram) * hits for ngram, hits in i.items())
        
        #print(hit_infos)
        total_lens = [sum(cand_lens[n]) for n in range(max_ngram)]
        #print(total_lens)
        nist_sum = sum(mydiv(hit_info, total_len) for hit_info, total_len in zip(hit_infos, total_lens))
        #print(nist_sum)
        #print(total_lens)
        # length penalty term
        bp = nist_length_penalty(sum(cand_lens[0]), avg_ref_len)
        #print(bp)
        return bp * nist_sum

def get_nist(ref_path, gen_path, is_sentence=False):
    gen_sentence_lst = open(gen_path).read().split("\n")
    ref_sentence_lst = open(ref_path).read().split("\n")
    nist_lst = [nist([ref_sentence], gen_sentence) for ref_sentence, gen_sentence in zip(ref_sentence_lst, gen_sentence_lst)]
    nist_score = np.mean(nist_lst)
    return nist_score*100

if __name__ == "__main__":

    ##### get parameters #####
    parser = argparse.ArgumentParser(description='calculate NIST')

    parser.add_argument("-r", "--ref_path", metavar="test.ref.txt",
                        help='the path of the reference\'s file', required = True)
    parser.add_argument("-g", "--gen_path", metavar="test.gen.txt",
                        help='the path of the generation\'s file', required = True)

    args = parser.parse_args()

    if os.path.exists(args.ref_path) and os.path.exists(args.gen_path):
        print(get_nist(args.ref_path, args.gen_path))
    else:
        print("File not exits")


# In[ ]:




