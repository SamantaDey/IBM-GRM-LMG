#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os,argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

w1=0.1
w2=0.2
w3=0.3
w4=0.4

def get_bleu_moses(ref_path, gen_path):
    gen_sentence_lst = open(gen_path).read().split("\n")
    ref_sentence_lst = open(ref_path).read().split("\n")
    bleu_moses_lst = [sentence_bleu([ref_sentence.split()], gen_sentence.split(),weights= (w1,w2,w3,w4)) for ref_sentence, gen_sentence in zip(ref_sentence_lst, gen_sentence_lst)]
    bleu_moses_score = np.mean(bleu_moses_lst)
    return bleu_moses_score*100

if __name__ == "__main__":

    ##### get parameters #####
    parser = argparse.ArgumentParser(description='calculate BLEU-Moses')

    parser.add_argument("-r", "--ref_path", metavar="test.ref.txt",
                        help='the path of the reference\'s file', required = True)
    parser.add_argument("-g", "--gen_path", metavar="test.gen.txt",
                        help='the path of the generation\'s file', required = True)

    args = parser.parse_args()

    if os.path.exists(args.ref_path) and os.path.exists(args.gen_path):
        print(get_bleu_moses(args.ref_path, args.gen_path))
    else:
        print("File not exits")


# In[ ]:




