import sys
import timeit
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# txt file loader
def file_loader(filename, rw_mode):
    try:
        file = open(filename, rw_mode)
        lines = [line[:-1] for line in file.readlines()] # rm '\n'
        file.close()
    except OSError as err:
        print('OS Error {0}'.format(err))
    except ValueError:
        print('Value Error')
    except:
        print('Unexpected Error:', sys.exc_info()[0])
    return lines

# counter fro positive and negative attitudes
def freq_counter(labels, reviews):
    pos_counts = Counter()
    neg_counts = Counter()
    for i in range(len(reviews)):
        for word in reviews[i].split(' '):
            if(labels[i] == 'positive'):
                pos_counts[word] += 1
            else:
                neg_counts[word] += 1
    return pos_counts, neg_counts

# return full_words_set
def create_full_words_set(pos_counts, neg_counts):
    output_set = set(pos_counts.keys()).union(neg_counts.keys())
    return output_set, len(output_set)

# calculate pos/neg ratios
def pos2neg_ratio(pos_counts, neg_counts, all_words, count_threshold=100):
    pos2neg_ratio = Counter()
    for word in all_words:
        if((pos_counts[word] > count_threshold) and (neg_counts[word] >count_threshold)):
            pos2neg_ratio[word] = pos_counts[word]/(neg_counts[word] + 1.0)
    return pos2neg_ratio

# print out most common words beautifully
def print_most_common(pos2neg_ratio):
    COMMON_LEN = 10
    # COMMON_LEN most commmon
    print('-'*40 + '\nPOSITIVE: \n' + '-'*40)
    for block in pos2neg_ratio.most_common()[:10]:
        print('{0:15s} | {1: 3.2f}'.format(block[0], block[1]))
    # COMMON_LEN least common
    print('-'*40 + '\nNEGATIVE:\n' + '-'*40)
    for block in pos2neg_ratio.most_common()[-COMMON_LEN:]:
        print('{0:15s} | {1: 3.2f}'.format(block[0], block[1]))

# plot out histgram of of pos/neg ratios
def plot_hist(pos2neg_ratio, BINS=100):
    plt.hist(list(pos2neg_ratio.values()), BINS)
    plt.xlabel('pos/neg ratio')
    plt.ylabel('counts')
    plt.title('histgram of [pos/neg]')
    plt.show()

def word_to_index(all_words, number_of_words):
    word_index = {}
    for word, i in enumerate(all_words):
        word_index[word] = i
    return word_index
# TODO: Complete the implementation of update_input_layer. It should count how many times each word is used in the given review, and then store those counts at the appropriate indices inside layer_0.
def update_input_layer():
    # TODO
    pass
    # global layer_0
    
    # # clear out previous state, reset the layer to be all 0s
    # layer_0 *= 0
    
    # # count how many times each word is used in the given review and store the results in layer_0 
    # for word in review.split(" "):
    #     layer_0[0][word2index[word]] += 1



def run():
    # define a run function
    # TODO
    pass

# load files with specific read/write modes
reviews = file_loader('reviews.txt', 'r')
labels = file_loader('labels.txt', 'r')
pos_counts, neg_counts = freq_counter(labels, reviews)
all_words, number_of_words = create_full_words_set(pos_counts, neg_counts)
pos2neg_ratio = pos2neg_ratio(pos_counts, neg_counts, all_words, count_threshold=100)
word_index = word_to_index(all_words, number_of_words)
print(word_index)



