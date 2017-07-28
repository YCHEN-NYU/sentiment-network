import sys
import timeit
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def load_file(filename, rw_mode):
    try:
        file = open(filename, rw_mode)
        output = [line[:-1] for line in file.readlines()] # rm '\n'
        file.close()
    except OSError as err:
        print('OS Error {0}'.format(err))
    except ValueError:
        print('Value Error')
    except:
        print('Unexpected Error:', sys.exc_info()[0])
    return output

reviews = load_file('reviews.txt', 'r')
labels = load_file('labels.txt', 'r')

pos_counts = Counter()
neg_counts = Counter()
for i in range(len(reviews)):
    for word in reviews[i].split(' '):
        if(labels[i] == 'positive'):
            pos_counts[word] += 1
        else:
            neg_counts[word] += 1

all_words = set(pos_counts.keys()).union(neg_counts.keys())
count_threshold = 100
pos2neg_ratio = Counter()
for word in all_words:
    if((pos_counts[word] > count_threshold) and (neg_counts[word] >count_threshold)):
        pos2neg_ratio[word] = pos_counts[word]/(neg_counts[word] + 1.0)

COMMON_LEN = 10
# COMMON_LEN most commmon
print('-'*40 + '\nPOSITIVE: \n' + '-'*40)
for block in pos2neg_ratio.most_common()[:10]:
    print('{0:15s} | {1: 3.2f}'.format(block[0], block[1]))
# COMMON_LEN least common
print('-'*40 + '\nNEGATIVE:\n' + '-'*40)
for block in pos2neg_ratio.most_common()[-COMMON_LEN:]:
    print('{0:15s} | {1: 3.2f}'.format(block[0], block[1]))

ratio_dist = list(pos2neg_ratio.values())
# print(ratio_dist)
plt.hist(ratio_dist, 100)
plt.xlabel('pos/neg ratio')
plt.ylabel('probability')
plt.title('P(pos/neg)')
plt.show()
