import pandas as pd 
# from fim import fpgrowth,fim 
import numpy as np
import math
from itertools import chain, combinations
import itertools
from numpy.random import random
from scipy import sparse
from bisect import bisect_left
from random import sample
import random
from sklearn.metrics import accuracy_score
from scipy.stats.distributions import poisson, gamma, beta, bernoulli, binom
import time
import scipy
from sklearn.preprocessing import binarize
import operator
from collections import Counter, defaultdict
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def extract_rules(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]     

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = 'neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        #           lineage.append((parent, split, threshold[parent], features[parent]))
        lineage.append((features[parent].strip()+suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)   
    rules = []
    for child in idx:
        rule = []
        for node in recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules

def code_categorical(df,colnames):
    for col in colnames:
#         print(col)
        values = df[col].unique()
        for val in values:
            df[col+'_'+str(val)] = (df[col]==val).astype(int)
    df.drop(colnames, axis = 1, inplace = True)

def code_multiple_categorical(df,colnames):
    for col in colnames:
        values = []
        for value in df[col]:
            try:
                values = values + [int(x) for x in value.split(',')]
            except:
                continue
        values = np.unique(values)
        for value in values:
            df[col + '_' + str(value)] = 0
        for i in range(df.shape[0]):
            try:
                vals = [int(x) for x in df.iloc[i][col].split(',')]
                for val in vals:
                    df.iloc[i][col+'_'+str(val)] = int(val in values)
            except:
                continue
    df.drop(colnames, axis = 1, inplace = True)
    
def code_continuous(df,collist,Nlevel):
    for col in collist:
        for q in range(1,Nlevel,1):
            threshold = df[~np.isnan(df[col])][col].quantile(float(q)/Nlevel)
            df[col+'_geq'+str(int(q))+'q'+str(threshold)] = (df[col] >= threshold).astype(float)
    df.drop(collist,axis = 1, inplace = True)

def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    return int(i-1) if i else 0

def fairness_eval(Yhat,Y, z, name = 'equal_opp'):
    if name == 'equal_opp':
        output = np.abs(np.dot(Yhat, np.multiply(z, Y))/np.dot(Y,z) - np.dot(Yhat, np.multiply(1-z,Y))/np.dot(Y,1-z))
        return output
    if name == 'demographic_parity':
        return np.abs(np.dot(Yhat,z)/np.sum(z) - np.dot(Yhat, 1-z)/np.sum(1-z))

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]