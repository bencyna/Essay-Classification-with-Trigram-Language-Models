import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2025 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    if n < 1:
      return []
    
    result = []
    seq = sequence + ["STOP"]
    cur_tup = ['START' for _ in range(n)]
    sequence_idx = 0 
    while sequence_idx < len(seq):
      tup_idx = n-1
      next_word = seq[sequence_idx]
      
      while tup_idx >= 0:
          prev_string = cur_tup[tup_idx]
          cur_tup[tup_idx] = next_word
          tup_idx -= 1
          next_word = prev_string
      
      sequence_idx += 1
      result.append(tuple(cur_tup))
    
    return result
      
    


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        self.total_words = 0
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) 
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        
        self.total_words = 0

        ##Your code here
        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)

            for unigram in unigrams:
                self.unigramcounts[unigram] += 1
                self.total_words += 1
            
            for bigram in bigrams:
                self.bigramcounts[bigram] += 1
                
            for trigram in trigrams:
                self.trigramcounts[trigram] += 1
            
    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        
        numerator = self.trigramcounts[trigram]
        denominator = self.bigramcounts[trigram[0:2]]
        if denominator == 0:
            return 1/(len(self.lexicon)-1)
        
        return numerator/denominator


    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        numerator = self.bigramcounts[bigram]
        denominator = self.unigramcounts[(bigram[0],)]
        if denominator == 0:
            return 1/(len(self.lexicon)-1)
        
        return numerator/denominator
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts[unigram]/self.total_words

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return        

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        unigram_raw = self.raw_unigram_probability((trigram[2],))
        bigram_raw = self.raw_bigram_probability(trigram[1:])
        trigram_raw = self.raw_trigram_probability(trigram)
        
        P_smoothed = lambda1*unigram_raw + lambda2*bigram_raw + lambda3*trigram_raw
        
        return P_smoothed
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        trigram_p_sum = 0
        for trigram in trigrams:
            p = self.smoothed_trigram_probability(trigram)
            log_space_p = math.log2(p)
            trigram_p_sum += log_space_p
        
        return trigram_p_sum


    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        total_log_probability = 0.0
        total_tokens = 0
        
        for sentence in corpus:
            log_probability = self.sentence_logprob(sentence)
            total_log_probability += log_probability
            total_tokens += len(sentence)+1
        
        l = (1/total_tokens) * total_log_probability
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1 < pp2:
                correct += 1
            total += 1 
    
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp2 < pp1:
                correct += 1
            total += 1 
        
        return correct/total

if __name__ == "__main__":

    # model = TrigramModel(sys.argv[1]) 

    # print(model.trigramcounts[('START','START','the')], "Expect 5478")
    # print(model.bigramcounts[('START','the')], "Expect 5478")
    # print( model.unigramcounts[('the',)], "Expect 61428")
    # s = ["natural", "language", "processing"]
    
    # print(model.sentence_logprob(s))
    # print(model.sentence_logprob(s+s))
    ## expect lower probabilities for longer sentence
    
    
    # print(model.perplexity(corpus_reader(sys.argv[1], model.lexicon)))
    
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', "test_high", "test_low")
    print(acc)

