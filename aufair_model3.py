import pandas as pd 
# from fim import fpgrowth,fim # you can comment this out if you do not use fpgrowth to generate rules
import numpy as np
from itertools import chain, combinations
import itertools
from random import sample
import random
from scipy import sparse
from bisect import bisect_left
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import binarize
import operator
from collections import Counter, defaultdict
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from util import *

class aufair(object):
    def __init__(self, binary_data,Ytrue, Yb, z_col,alpha = 0,beta = 0.01,budget = 1,metric = 'equal_opp'):
        self.df = binary_data  
        self.N = len(Ytrue)
        self.Yb = Yb.copy()
        self.Z = np.array(binary_data[z_col])
        self.Y = Ytrue 
        self.metric = metric
        self.alpha = alpha
        self.beta = beta
        self.budget = budget
        h = RandomForestClassifier(n_estimators = 500, random_state = 42)
        h.fit(binary_data, Yb)
        self.Yb_prob = h.predict_proba(binary_data)[:,0]
        self.inconfidence = np.abs(self.Yb_prob - 0.5)

    def generate_rulespace(self,supp,maxlen,N, need_negcode = False,method = 'fpgrowth'):
        if method == 'fpgrowth': # generate rules with fpgrowth
            if need_negcode:
                df = 1-self.df 
                df.columns = [name.strip() + 'neg' for name in self.df.columns]
                df = pd.concat([self.df,df],axis = 1)
            else:
                df = 1 - self.df
            pindex = np.where(self.Y==1)[0]
            nindex = np.where(self.Y!=1)[0]
#            itemMatrix = [[item for item in df.columns if row[item] ==1] for i,row in df.iterrows() ]  
            itemMatrix = [[item for item in df.columns if row[item]] for i,row in df.iterrows() ]   
            prules= fpgrowth([itemMatrix[i] for i in pindex],supp = supp,zmin = 1,zmax = maxlen)
            prules = [np.sort(x[0]).tolist() for x in prules]
            nrules= fpgrowth([itemMatrix[i] for i in nindex],supp = supp,zmin = 1,zmax = maxlen)
            nrules = [np.sort(x[0]).tolist() for x in nrules]
        else: # if you cannot install the package fim, then use random forest to generate rules
            print('Using random forest to generate rules ...')
            prules = []
            for length in range(2,maxlen+1,1):
                n_estimators = 250*length
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length)
                clf.fit(self.df,self.Y)
                for n in range(n_estimators):
                    prules.extend(extract_rules(clf.estimators_[n],self.df.columns))
            prules = [list(x) for x in set(tuple(np.sort(x)) for x in prules)] 
            nrules = []
            for length in range(2,maxlen+1,1):
                n_estimators = 250*length# min(5000,int(min(comb(df.shape[1], length, exact=True),10000/maxlen)))
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length)
                clf.fit(self.df,1-self.Y)
                for n in range(n_estimators):
                    nrules.extend(extract_rules(clf.estimators_[n],self.df.columns))
            nrules = [list(x) for x in set(tuple(np.sort(x)) for x in nrules)]   
            df = 1-self.df 
            df.columns = [name.strip() + 'neg' for name in self.df.columns]
            df = pd.concat([self.df,df],axis = 1)
        self.prules, self.pRMatrix, self.psupp, self.pprecision, self.perror = self.screen_rules(prules,df,self.Y,N,supp)
        self.nrules, self.nRMatrix, self.nsupp, self.nprecision, self.nerror = self.screen_rules(nrules,df,1-self.Y,N,supp)

    def screen_rules(self,rules,df,y,N,supp,criteria = 'precision'):
        itemInd = {}
        for i,name in enumerate(df.columns):
            itemInd[name] = int(i)
        len_rules = [len(rule) for rule in rules]
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        indptr =list(accumulate(len_rules))
        indptr.insert(0,0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(df.columns),len(rules)))
        # mat = sparse.csr_matrix.dot(df,ruleMatrix)
        mat = np.matrix(df)*ruleMatrix
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
        Z =  (mat ==lenMatrix).astype(int)

        Zpos = [Z[i] for i in np.where(y>0)][0]
        TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
        supp_select = np.where(TP>=supp*sum(y)/100)[0]
        FP = np.array(np.sum(Z,axis = 0))[0] - TP
        precision = TP.astype(float)/(TP+FP)


        supp_select = np.array([i for i in supp_select if precision[i]>np.mean(y)])
        select = np.argsort(precision[supp_select])[::-1][:N].tolist()
        ind = list(supp_select[select])
        rules = [rules[i] for i in ind]
        RMatrix = np.array(Z[:,ind])
        rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules]
        supp = np.array(np.sum(Z,axis=0).tolist()[0])[ind]
        return rules, RMatrix, supp, precision[ind], FP[ind]


    def query(self,Yhat,q):
        I_left = [i for i in range(len(self.Y)) if i not in self.I_querried]
        # if np.dot(yhat[self.I_querried], np.multiply(self.Z[self.I_querried], self.Y[self.I_querried]))/np.dot(self.Y[self.I_querried],self.Z[self.I_querried]) > np.dot(yhat[self.I_querried], np.multiply(1-self.Z[self.I_querried],self.Y[self.I_querried]))/np.dot(self.Y[self.I_querried],1-self.Z[self.I_querried]):
        #     ind = np.where(self.Z[self.I_querried] != yhat[self.I_querried])[0]
        # else:
        #     ind = np.where(self.Z[self.I_querried] == yhat[self.I_querried])[0]
        # I_left = list(set(ind).intersection(I_left))

        # # --------------- random ---------------
        # draw = sample(I_left, min(len(I_left),q))

        # -------- sample based on bias first and then random --------
        # first deciding which group to sample from and randomly sample from that group
        # save to group.pkl
        # notes: experiments show this is not doing better than random query
        # if self.metric == 'equal_opp':
        #     if np.dot(Yhat[self.I_querried], np.multiply(self.Z[self.I_querried], self.Y[self.I_querried]))/np.dot(self.Y[self.I_querried],self.Z[self.I_querried]) > np.dot(Yhat[self.I_querried], np.multiply(1-self.Z[self.I_querried],self.Y[self.I_querried]))/np.dot(self.Y[self.I_querried],1-self.Z[self.I_querried]):
        #         group = 0
        #     else:
        #         group = 1
        # elif self.metric == 'demographic_parity':
        #     if np.dot(Yhat[self.I_querried],self.Z[self.I_querried])/np.sum(self.Z[self.I_querried]) > np.dot(Yhat[self.I_querried], 1-self.Z[self.I_querried])/np.sum(1-self.Z[self.I_querried]):
        #         group = 0
        #     else:
        #         group = 1
        # ind = np.where(self.Z == group)[0]
        # candidate = list(set(ind).intersection(I_left))
        # if q <= len(candidate):
        #     draw = sample(candidate, q)
        # else:
        #     ind = np.where(self.Z == 1 - group)[0]
        #     candidate2 = list(set(ind).intersection(I_left))
        #     draw = candidate + sample(candidate2, q - len(candidate))

        # # -------- sample based on bias first and then inconfidence --------
        # first deciding which group to sample from and sample from that group, pick those with smallest confidence
        # save to inconfidence.pkl
        if self.metric == 'equal_opp':
            if np.dot(Yhat[self.I_querried], np.multiply(self.Z[self.I_querried], self.Y[self.I_querried]))/np.dot(self.Y[self.I_querried],self.Z[self.I_querried]) > np.dot(Yhat[self.I_querried], np.multiply(1-self.Z[self.I_querried],self.Y[self.I_querried]))/np.dot(self.Y[self.I_querried],1-self.Z[self.I_querried]):
                group = 0
            else:
                group = 1
        elif self.metric == 'demographic_parity':
            if np.dot(Yhat[self.I_querried],self.Z[self.I_querried])/np.sum(self.Z[self.I_querried]) > np.dot(Yhat[self.I_querried], 1-self.Z[self.I_querried])/np.sum(1-self.Z[self.I_querried]):
                group = 0
            else:
                group = 1
        ind = np.where(self.Z == group)[0]
        I_left = np.array(list(set(ind).intersection(I_left)))
        ind = np.argsort(self.inconfidence[I_left])
        draw = list(I_left[ind[:min(len(I_left),q)]])


        # # -------- sample based on bias first and then weight --------
        # first deciding which group to sample from and sample from that group, pick those with smallest confidence
        # save to test_weight.pkl
        # if self.metric == 'equal_opp':
        #     if np.dot(Yhat[self.I_estimate], np.multiply(self.Z[self.I_estimate], self.Y[self.I_estimate]))/np.dot(self.Y[self.I_estimate],self.Z[self.I_estimate]) > np.dot(Yhat[self.I_estimate], np.multiply(1-self.Z[self.I_estimate],self.Y[self.I_estimate]))/np.dot(self.Y[self.I_estimate],1-self.Z[self.I_estimate]):
        #         group = 0
        #     else:
        #         group = 1
        # elif self.metric == 'demographic_parity':
        #     if np.dot(Yhat[self.I_estimate],self.Z[self.I_estimate])/np.sum(self.Z[self.I_estimate]) > np.dot(Yhat[self.I_estimate], 1-self.Z[self.I_estimate])/np.sum(1-self.Z[self.I_estimate]):
        #         group = 0
        #     else:
        #         group = 1        
        # weight = 0.5 - self.inconfidence + self.alpha * np.multiply(Yhat == 0, self.Z == group)
        # ind = np.argsort(weight[I_left])[::-1]
        # draw = list(np.array(I_left)[ind[:min(len(I_left),q)]])

        self.I_querried = self.I_querried + draw

    def fit(self, Niteration = 1000, metric = 'equal_opp', print_message=False):
        ############# initialization #############
        self.maps = []
        self.metric = metric
        T0 = 0.001
        # generate random initial solutions with 3 positive rules and 3 negative rules
        prs_curr = sample(list(range(len(self.prules))),3) 
        nrs_curr = sample(list(range(len(self.nrules))),3)
        group0 = list(np.where(self.Z==0)[0])
        group1 = list(np.where(self.Z==1)[0])
        self.I_querried = sample(group0, min(len(group0), int(len(group0) * self.budget * 0.25))) + sample(group1, min(len(group1), int(len(group1) * self.budget * 0.25)))
        # self.I_estimate = list(range(len(self.Y)))
        # self.I_querried = list(range(len(self.Y)))
        Yhat_curr,covered_curr = self.get_yhat(prs_curr,nrs_curr)
        error_tmp,bias_tmp,obj_curr = self.compute_obj(Yhat_curr)
        obj_curr = obj_curr + self.beta * (len(prs_curr)+len(nrs_curr))
        self.tmp = Yhat_curr
        self.maps.append([-1,obj_curr,prs_curr,nrs_curr,[]])
        print('starting from obj = {}, error = {}, bias = {}'.format(obj_curr,error_tmp, bias_tmp))
        ntimes = 10
        q = int((len(self.Y)*self.budget - len(self.I_querried))/ntimes)

        for iter in range(Niteration):
            # if iter >0.75 * Niteration:
            #     prs_curr,nrs_curr,pcovered_curr,ncovered_curr,overlap_curr,covered_curr, Yhat_curr = prs_opt[:],nrs_opt[:],pcovered_opt[:],ncovered_opt[:],overlap_opt[:],covered_opt[:], Yhat_opt[:] 
            if iter % int((Niteration*0.4)/ntimes)==0 and len(self.I_querried)< int(self.budget * len(self.Y)):
                self.query(Yhat_curr,q)
                error_curr, bias_curr, obj_curr= self.compute_obj(Yhat_curr)
                obj_curr = obj_curr + self.beta * (len(prs_curr)+len(nrs_curr))
                self.maps.append([iter,obj_curr,prs_curr,nrs_curr,Yhat_curr])
                print('iteration: {}, objective updated to: {}(obj) = {}(error) + {} * {}(bias) on {} data'.format(iter, round(obj_curr,3),error_curr, self.alpha, bias_curr, len(self.I_querried)/len(self.Y)))
            prs_new,nrs_new = self.propose_rs(prs_curr,nrs_curr,Yhat_curr,covered_curr,print_message)
            Yhat_new,covered_new = self.get_yhat(prs_new,nrs_new)
            error_new, bias_new, obj_new= self.compute_obj(Yhat_new)
            obj_new = obj_new + self.beta * (len(prs_new)+len(nrs_new))
            T = T0**(iter/Niteration)
            alpha = np.exp(float(-obj_new +obj_curr)/T) # minimize
            if obj_new < self.maps[-1][1]:
                print('\n**  max at iter = {} ** \n {}(obj) = {}(error) + {} * {}(bias) + {} * {} (rules) on {} data'.format(iter,round(obj_new,3),error_new, self.alpha, bias_new, self.beta, len(prs_new)+len(nrs_new),len(self.I_querried)/len(self.Y)))
                print('prs = {}, nrs = {}'.format(prs_new,nrs_new))
                self.maps.append([iter,obj_new,prs_new,nrs_new,Yhat_new])
            
            # if print_message:
            #     perror, nerror, oerror, berror = self.diagnose(pcovered_new,ncovered_new,overlap_new,covered_new,Yhat_new)
            #     print('\niter = {}, alpha = {}, {}(obj) = {}(error) + {}(bias)'.format(iter,round(alpha,2),round(obj_new,3),np.mean(Yhat_new !=self.Y), fairness_eval(Yhat_new,self.Y,self.Z,metric)  ))
            #     print('prs = {}, nrs = {}'.format(prs_new, nrs_new))
            if random.random() <= alpha:
                prs_curr,nrs_curr,obj_curr,covered_curr,Yhat_curr =  prs_new[:],nrs_new[:],obj_new, covered_new[:],Yhat_new[:]


        self.positive_rule_set = self.maps[-1][2]
        self.negative_rule_set = self.maps[-1][3]

    # def diagnose(self, pcovered, ncovered, overlapped, covered, Yhat):
    #     perror = sum(self.Y[pcovered]!=Yhat[pcovered])
    #     nerror = sum(self.Y[ncovered]!=Yhat[ncovered])
    #     oerror = sum(self.Y[overlapped]!=Yhat[overlapped])
    #     berror = sum(self.Y[~covered]!=Yhat[~covered])
    #     return perror, nerror, oerror, berror

    def get_yhat(self,prs, nrs):
        p = np.sum(self.pRMatrix[:,prs],axis = 1)>0 # instances covered by positive rules
        n = np.sum(self.nRMatrix[:,nrs],axis = 1)>0 # instances covered by negative rules
        # overlap = np.multiply(p,n) # instances covered by both
        covered = np.logical_xor(p,n) # instances covered by either

        pcovered = np.where(p)[0]
        Yhat = self.Yb.copy()
        Yhat[covered] = 0
        Yhat[pcovered] = 1
        return Yhat, np.where(covered)[0]        

    def compute_obj(self,Yhat):
        error = np.mean(Yhat[self.I_querried]!=self.Y[self.I_querried])
        bias = fairness_eval(Yhat[self.I_querried],self.Y[self.I_querried],self.Z[self.I_querried],self.metric) 
        return error, bias, error + self.alpha * bias 
    
    # def propose_rs(self, prs,nrs,Yhat,covered,print_message = False):
    #     query_covered = [i for i in covered if i in self.I_querried]
    #     query_noncovered = [i for i in self.I_querried if i not in query_covered]
    #     incorr = np.where(Yhat[query_covered]!=self.Y[query_covered])[0]# correct interpretable models
    #     incorrb = np.where(Yhat[query_noncovered]!=self.Y[query_noncovered])[0]
    #     # p = np.sum(self.pRMatrix[self.I_querried,prs],axis = 1)
    #     # n = np.sum(self.nRMatrix[self.I_querried,nrs],axis = 1)
    #     ex = -1
    #     # if sum(covered) ==self.N: # covering all examples.
    #     #     move = ['cut']
    #     #     self.actions.append(0)
    #     #     if len(prs)==0:
    #     #         sign = [0]
    #     #     elif len(nrs)==0:
    #     #         sign = [1]
    #     #     else:
    #     #         sign = [int(random.random()<0.5)]  
    #     # elif len(incorr) ==0 and (len(incorrb)==0 or len(overlapped) ==self.N) or sum(overlapped) > sum(covered):
    #     #     self.actions.append(1)
    #     #     move = ['cut']
    #     #     sign = [int(random.random()<0.5)]           
    #     # else: 
    #     # self.actions.append(3)
    #     incorr_total = list(incorr) + list(incorrb)
    #     if len(incorr_total)>0:
    #         ex = sample(incorr_total,1)[0] 
    #     if ex in incorr: # incorrectly classified by the interpretable model
    #         rs_indicator = Yhat[ex]
    #         move = ['cut']
    #         sign = [rs_indicator]
    #     else: # incorrectly classified by the black box model
    #         move = ['add']
    #         sign = [1-self.Y[ex]]
    #     for j in range(len(move)):
    #         if sign[j]==1:
    #             prs = self.action(move[j],sign[j],ex,prs,Yhat)
    #         else:
    #             nrs = self.action(move[j],sign[j],ex,nrs,Yhat)

    #     # p = np.sum(self.pRMatrix[:,prs],axis = 1)>0
    #     # n = np.sum(self.nRMatrix[:,nrs],axis = 1)>0
    #     # o = np.multiply(p,n)
    #     return prs, nrs 

    def propose_rs(self, prs,nrs,Yhat,covered,print_message = False):
        if random.random()>0.5:
            move = 'cut'
        else:
            move = 'add'
        if move == 'cut' and len(prs)==0:
            sign = 0
        elif move == 'cut' and len(nrs)==0:
            sign = 1
        else:
            sign = round(random.random())

        if sign==1:
            prs = self.action(move,sign,prs,Yhat)
        else:
            nrs = self.action(move,sign,nrs,Yhat)

        return prs, nrs 

    def action(self,move, rs_indicator, rules,Yhat):
        if rs_indicator==1:
            RMatrix = self.pRMatrix
            # error = self.perror
            supp = self.psupp
        else:
            RMatrix = self.nRMatrix
            # error = self.nerror
            supp = self.nsupp
        Y = self.Y if rs_indicator else 1- self.Y
        if move=='cut' and len(rules)>0:
            """ cut """
            p = []
            all_sum = np.sum(RMatrix[:,rules],axis = 1)
            for index,rule in enumerate(rules):
                Yhat= ((all_sum - np.array(RMatrix[:,rule]))>0).astype(int)
                TP,FP,TN,FN  =confusion_matrix(Yhat,Y).ravel()
                p.append(TP.astype(float)/(TP+FP+1))
                # p.append(log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2))
            p = [x - min(p) for x in p]
            p = np.exp(p)
            p = np.insert(p,0,0)
            p = np.array(list(accumulate(p)))
            if p[-1]==0:
                cut_rule = sample(rules,1)[0]
            else:
                p = p/p[-1]
                index = find_lt(p,random.random())
                cut_rule = rules[index]
            rules.remove(cut_rule)
        elif move == 'add' : 
            """ add """
            select = [i for i in range(RMatrix.shape[1]) if i not in rules]
            if len(select)>0:
                add_rule = sample(select,1)[0]
                if add_rule not in rules:
                    rules.append(add_rule)
        return rules

    # def action(self,move, rs_indicator, ex, rules,Yhat):
    #     if rs_indicator==1:
    #         RMatrix = self.pRMatrix
    #         # error = self.perror
    #         supp = self.psupp
    #     else:
    #         RMatrix = self.nRMatrix
    #         # error = self.nerror
    #         supp = self.nsupp
    #     # Y = self.Y if rs_indicator else 1- self.Y
    #     if move=='cut' and len(rules)>0:
    #         """ cut """
    #         if random.random()<1 and ex >=0:
    #             candidate = list(set(np.where(RMatrix[ex,:]==1)[0]).intersection(rules))
    #             if len(candidate)==0:
    #                 candidate = rules
    #             cut_rule = sample(candidate,1)[0]
    #         else:
    #             p = []
    #             all_sum = np.sum(RMatrix[:,rules],axis = 1)
    #             for index,rule in enumerate(rules):
    #                 Yhat= ((all_sum - np.array(RMatrix[:,rule]))>0).astype(int)
    #                 TP,FP,TN,FN  =confusion_matrix(Yhat,Y).ravel()
    #                 p.append(TP.astype(float)/(TP+FP+1))
    #                 # p.append(log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2))
    #             p = [x - min(p) for x in p]
    #             p = np.exp(p)
    #             p = np.insert(p,0,0)
    #             p = np.array(list(accumulate(p)))
    #             if p[-1]==0:
    #                 cut_rule = sample(rules,1)[0]
    #             else:
    #                 p = p/p[-1]
    #                 index = find_lt(p,random.random())
    #                 cut_rule = rules[index]
    #         rules.remove(cut_rule)
    #     elif move == 'add' and ex>=0: 
    #         """ add """
    #         score_max = -self.N *10000000
    #         if self.Y[ex]*rs_indicator + (1 - self.Y[ex])*(1 - rs_indicator)==1:
    #             # select = list(np.where(RMatrix[ex] & (error +self.alpha*self.N < self.beta * supp))[0]) # fix
    #             select = list(np.where(RMatrix[ex])[0])
    #         else:
    #             # select = list(np.where( ~RMatrix[ex]& (error +self.alpha*self.N < self.beta * supp))[0])
    #             select = list(np.where( ~RMatrix[ex])[0])
    #         self.select = select
    #         if len(select)>0:
    #             add_rule = sample(select,1)[0]
    #             # if random.random()<0.25:
    #             #     add_rule = sample(select,1)[0]
    #             # else: 
    #             #     # cover = np.sum(RMatrix[(~covered)&(~covered2), select],axis = 0)
    #             #     # =============== Use precision as a criteria ===============
    #             #     # Yhat_neg_index = np.where(np.sum(RMatrix[:,rules],axis = 1)<1)[0]
    #             #     # mat = np.multiply(RMatrix[Yhat_neg_index.reshape(-1,1),select].transpose(),Y[Yhat_neg_index])
    #             #     # TP = np.sum(mat,axis = 1)
    #             #     # FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1,1),select],axis = 0) - TP)
    #             #     # TN = np.sum(Y[Yhat_neg_index]==0)-FP
    #             #     # FN = sum(Y[Yhat_neg_index]) - TP
    #             #     # p = (TP.astype(float)/(TP+FP+1)) + self.alpha * supp[select]
    #             #     # add_rule = select[sample(list(np.where(p==max(p))[0]),1)[0]]
    #             #     # =============== Use objective function as a criteria ===============
    #             #     for ind in select:
    #             #         z = np.logical_or(RMatrix[:,ind],Yhat)
    #             #         TN,FP,FN,TP = confusion_matrix(z,self.Y).ravel()
    #             #         score = FP+FN 
    #             #         if score > score_max:
    #             #             score_max = score
    #             #             add_rule = ind
    #             if add_rule not in rules:
    #                 rules.append(add_rule)
    #     else: # expand
    #         candidates = [x for x in range(RMatrix.shape[1])]
    #         if rs_indicator:
    #             select = list(set(candidates).difference(rules))
    #         else:
    #             select = list(set(candidates).difference(rules))
    #         # self.error = error
    #         self.supp = supp
    #         self.select = select
    #         self.candidates = candidates
    #         self.rules = rules
    #         add_rule = sample(select, 1)[0]
    #         if add_rule not in rules:
    #             rules.append(add_rule)
    #     return rules


    def predict(self, df, Y,Yb ):
        prules = [self.prules[i] for i in self.positive_rule_set]
        nrules = [self.nrules[i] for i in self.negative_rule_set]
        # if isinstance(self.df, scipy.sparse.csc.csc_matrix)==False:
        dfn = 1-df #df has negative associations
        dfn.columns = [name.strip() + 'neg' for name in df.columns]
        df_test = pd.concat([df,dfn],axis = 1)
        if len(prules):
            p = [[] for rule in prules]
            for i,rule in enumerate(prules):
                p[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
            p = (np.sum(p,axis=0)>0).astype(int)
        else:
            p = np.zeros(len(Y))
        if len(nrules):
            n = [[] for rule in nrules]
            for i,rule in enumerate(nrules):
                n[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
            n = (np.sum(n,axis=0)>0).astype(int)
        else:
            n = np.zeros(len(Y))
        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        covered = [x for x in range(len(Y)) if x in pind or x in nind]
        Yhat = Yb.copy()
        Yhat[nind] = 0
        Yhat[pind] = 1
        return Yhat,covered

def accumulate(iterable, func=operator.add):
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
    if i:
        return int(i-1)
    else:
        return 0


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