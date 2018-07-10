# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from hmmlearn import hmm
import math
import MySQLdb
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics.pairwise import pairwise_distances_argmin
import warnings
import time
import sys

# states = ["box 1", "box 2", "box3"]
# n_states = len(states)
#
# observations = ["red", "white"]
# n_observations = len(observations)
# data = np.loadtxt('hmmodletest.data', dtype=int, delimiter=',', skiprows=7, usecols=(0, 1, 2))
# up_down = data[:, 1]
# n_state = data[:, 2]
# oberserve = data[:, 0].reshape(-1,1)
# print oberserve
#
# model2 = hmm.MultinomialHMM(n_components=6, n_iter=200, tol=0.01)
# X2 = np.array([[0,1,0,1],[0,0,0,1],[1,0,1,1]])
# print type(oberserve)
# model2.fit(oberserve)
# print model2.startprob_
# print model2.transmat_
# print model2.emissionprob_
# print model2.score(oberserve)
# model2.fit(oberserve)
# print model2.startprob_
# print model2.transmat_
# print model2.emissionprob_
# print model2.score(oberserve)
# model2.fit(oberserve)
# print model2.startprob_
# print model2.transmat_
# print model2.emissionprob_
# print model2.score(oberserve)

# -*-coding:UTF-8-*-
def find_result():
    try:
        conn = MySQLdb.connect(host='192.168.5.63', user='root', passwd='wwc@icinfo', port=3306, charset='utf8')
        cur = conn.cursor()

        conn.select_db('sandp')

        # count = cur.execute('select issue,expertname,redthree,lasthit,lastscore,formax,oddeven,smallbig,primecom,p51,p52,p53,'
        #                     'p101,p102,p103,p201,p202 ,p203,p301,p302,p303,p501,p502,p503,p1001,p1002,p1003,nowhit,nowscore '
        #                     'from ssqthred WHERE expertname = "一码当先"')

        sqlqry= "SELECT baihund,mtstat FROM sandp.sandhistory";

        count = cur.execute(sqlqry)
        print 'there has %s rows record' % count

        results = cur.fetchall()
        results = list(results)
        conn.commit()
        cur.close()
        conn.close()
        return results
    except MySQLdb.Error, e:
        print "Mysql Error %d: %s" % (e.args[0], e.args[1])

def observ_result():
    try:
        conn = MySQLdb.connect(host='192.168.5.63', user='root', passwd='wwc@icinfo', port=3306, charset='utf8')
        cur = conn.cursor()

        conn.select_db('sandp')

        # count = cur.execute('select issue,expertname,redthree,lasthit,lastscore,formax,oddeven,smallbig,primecom,p51,p52,p53,'
        #                     'p101,p102,p103,p201,p202 ,p203,p301,p302,p303,p501,p502,p503,p1001,p1002,p1003,nowhit,nowscore '
        #                     'from ssqthred WHERE expertname = "一码当先"')

        sqlqry= "SELECT baihund FROM sandp.sandhistory";

        count = cur.execute(sqlqry)
        print 'there has %s rows record' % count

        results1 = cur.fetchall()
        results1 = list(results1)
        conn.commit()
        cur.close()
        conn.close()
        return results1
    except MySQLdb.Error, e:
        print "Mysql Error %d: %s" % (e.args[0], e.args[1])
def generate_index_map(lables):
        index_label = {}
        label_index = {}
        i = 0
        for l in lables:
            index_label[i] = l
            label_index[l] = i
            i += 1
        return label_index, index_label

def convert_observations_to_index(observations, label_index):
        list = []
        for o in observations:
            list.append(label_index[o])
        return list


def convert_map_to_vector(map, label_index):
        v = np.empty(len(map), dtype=float)
        for e in map:
            v[label_index[e]] = map[e]
        return v


def convert_map_to_matrix(map, label_index1, label_index2):
        m = np.empty((len(label_index1), len(label_index2)), dtype=float)
        for line in map:
            for col in map[line]:
                m[label_index1[line]][label_index2[col]] = map[line][col]
        return m


class HMM:
    """
    Order 1 Hidden Markov Model

    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector
    """

    def __init__(self, Ann=[[0]], Bnm=[[0]], pi1n=[0]):
        self.A = np.array(Ann)
        self.B = np.array(Bnm)
        self.pi = np.array(pi1n)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]



    def printhmm(self):
        print "=================================================="
        print "HMM content: N =", self.N, ",M =", self.M
        for i in range(self.N):
            if i == 0:
                print "hmm.A ", self.A[i, :], " hmm.B ", self.B[i, :]
            else:
                print "      ", self.A[i, :], "       ", self.B[i, :]
        print "hmm.pi", self.pi
        print "=================================================="

    def simulate(self, T):

            def draw_from(probs):
                return np.where(np.random.multinomial(1,probs) == 1)[0][0]

            observations = np.zeros(T, dtype=int)
            states = np.zeros(T, dtype=int)
            states[0] = draw_from(self.pi)
            observations[0] = draw_from(self.B[states[0],:])
            for t in range(1, T):
                states[t] = draw_from(self.A[states[t-1],:])
                observations[t] = draw_from(self.B[states[t],:])
            return observations, states

    def _forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        F = np.zeros((N,T))
        F[:,0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                F[n,t] = np.dot(F[:, t-1], (self.A[:,n])) * self.B[n, obs_seq[t]]

        return F

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((N,T))
        X[:,-1:] = 1

        for t in reversed(range(T-1)):
            for n in range(N):
                X[n,t] = np.sum(X[:,t+1] * self.A[n,:] * self.B[:, obs_seq[t+1]])

        return X

    def state_path(self, obs_seq):
        """
        Returns
        -------
        V[last_state, -1] : float
            Probability of the optimal state path
        path : list(int)
            Optimal state path for the observation sequence
        """
        V, prev = self.viterbi(obs_seq)

        # Build state path with greatest probability
        last_state = np.argmax(V[:,-1])
        last_state1 = np.argsort(V[:,-1])
        path = list(self.build_viterbi_path(prev, last_state))

        return V[last_state,-1], reversed(path)

    def state_pathDIY(self, obs_seq, prev_state):
        """
        Returns
        -------
        V[last_state, -1] : float
            Probability of the optimal state path
        path : list(int)
            Optimal state path for the observation sequence
        """
        V, prev = self.viterbi(obs_seq)

        # Build state path with greatest probability
        last_state = np.argsort(-V[:, -1]).tolist()
        lists = [[] for i in range(7)]
        lists[0].append(prev_state[0])
        lists[1].append(prev_state[1])
        lists[2].append(int(last_state[0]))
        lists[2].append(V[last_state[0], -1])
        lists[3].append(int(last_state[1]))
        lists[3].append(V[last_state[1], -1])
        lists[4].append(int(last_state[2]))
        lists[4].append(V[last_state[2], -1])
        lists[5].append(int(last_state[3]))
        lists[5].append(V[last_state[3], -1])
        lists[6].append(int(last_state[4]))
        lists[6].append(V[last_state[4], -1])

        return lists

    def baum_welch_train(self, observations, criterion=0.05):
        n_states = self.A.shape[0]
        n_samples = len(observations)

        done = False
        while not done:
            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha
            alpha = self._forward(observations)

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = self._backward(observations)

            xi = np.zeros((n_states, n_states, n_samples-1), np.float)
            for t in range(n_samples-1):
                denom = np.dot(np.dot(alpha[:, t].T, self.A) * self.B[:, observations[t+1]].T, beta[:, t+1])
                for i in range(n_states):
                    numer = alpha[i,t] * self.A[i, :] * self.B[:, observations[t+1]].T * beta[:, t+1].T
                    xi[i, :, t] = numer / denom

            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.sum(xi,axis=1)
            # Need final gamma element for new B
            prod =  (alpha[:, n_samples-1] * beta[:, n_samples-1]).reshape((-1,1))
            gamma = np.hstack((gamma,  prod / np.sum(prod))) #append one more to gamma!!!

            newpi = gamma[:,0]
            newA = np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
            newB = np.copy(self.B)

            num_levels = self.B.shape[1]
            sumgamma = np.sum(gamma,axis=1)
            for lev in range(num_levels):
                mask = observations == lev
                newB[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma

            if np.max(abs(self.pi - newpi)) < criterion and \
                            np.max(abs(self.A - newA)) < criterion and \
                            np.max(abs(self.B - newB)) < criterion:
                done = 1

            self.A[:],self.B[:],self.pi[:] = newA, newB, newpi

    def build_viterbi_path(self, prev, last_state):
        """Returns a state path ending in last_state in reverse order."""
        T = len(prev)
        yield(last_state)
        for i in range(T-1, -1, -1):
            yield(prev[i, last_state])
            last_state = prev[i, last_state]


    def viterbi(self, obs_seq):
        """
        Returns
        -------
        V : numpy.ndarray
            V [s][t] = Maximum probability of an observation sequence ending
                       at time 't' with final state 's'
        prev : numpy.ndarray
            Contains a pointer to the previous state at t-1 that maximizes
            V[state][t]
        """
        N = self.A.shape[0]
        T = len(obs_seq)
        prev = np.zeros((T - 1, N), dtype=int)

        # DP matrix containing max likelihood of state at a given time
        V = np.zeros((N, T))
        V[:,0] = self.pi * self.B[:,obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                seq_probs = V[:,t-1] * self.A[:,n] * self.B[n, obs_seq[t]]
                prev[t-1,n] = np.argmax(seq_probs)
                V[n,t] = np.max(seq_probs)

        return V, prev

    def train(self, I, O, num_state, num_obserivation, init_observation=-1, init_state=-1):
            """
            function: training HMM
            :param I: state list like I = np.array([[0,1,2],[1,0,1],[1,2,0],])
            :param O: observation list like O =     O = np.array([[0,1,1],[1,0,1],[1,1,0],])
            :param num_state: the number of state, lke 3
            :param num_obserivation: the number of observation, like 2
            :param init_observation: the index of init observation, like 1
            :param init_state: the index of init starw, like 2
            """
            print "statr training HMM..."
            self.N = num_state
            self.M = num_obserivation

            # count num_A[i,j] standing for the numbers of state i translating to state j/
            #0RR/1RQ/2RP/3QP/4QR/5QQ
            num_A = np.zeros((num_state, num_state), np.float)
            for i in range(self.N):
                for j in range(self.N):
                    num_i2j = 0
                    for i_I in range(I.shape[0]-1):
                            if I[i_I] == i and I[i_I+ 1] == j:
                                num_i2j += 1
                    num_A[i, j] = num_i2j

            # count num_B[i,j] standing for the numbers of state i translating to obsrtvation j
            num_B = np.zeros((num_state, num_obserivation), np.float)
            for i in range(self.N):
                for j in range(self.M):
                    num_i2j = 0
                    for i_I in range(I.shape[0]):
                        for j_I in range(I.shape[1]):
                            if I[i_I, j_I] == i and O[i_I, j_I] == j:
                                num_i2j += 1
                    num_B[i, j] = num_i2j


                self.A = num_A / np.sum(np.mat(num_A), axis=1).A
                self.B = num_B / np.sum(np.mat(num_B), axis=1).A


            # calculate pi according init_observation or init_state
            if init_state != -1:
                print "init pi with init_state!"
                pi_temp = np.zeros((self.N,), np.float)
                self.pi = pi_temp[init_state] = 1.0
            elif init_observation != -1:
                print "init pi with init_observation!"
                self.pi = self.B[:, init_observation] / np.sum(self.B[:, init_observation])
            else:
                print "init pi with state list I!"
                self.pi = np.zeros((self.N,), np.float)
                for i in range(self.N):
                    num_state_i = 0
                    for line in I:
                        if line[0] == i:
                            num_state_i += 1
                    self.pi[i] = num_state_i
                self.pi = self.pi/np.sum(self.pi, axis=0)

            print "finished train successfully! the hmm is:"
            # self.printhmm()

######################新版Baum_Welch############
class HMM_SEC:
    def __init__(self, Ann, Bnm, Pi, O):
        self.A = np.array(Ann, np.float)
        self.B = np.array(Bnm, np.float)
        self.Pi = np.array(Pi, np.float)
        self.O = np.array(O, np.float)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]

    def viterbi(self):
        # given O,lambda .finding I

        T = len(self.O)
        I = np.zeros(T, np.float)

        delta = np.zeros((T, self.N), np.float)
        psi = np.zeros((T, self.N), np.float)

        for i in range(self.N):
            delta[0, i] = self.Pi[i] * self.B[i, self.O[0]]
            psi[0, i] = 0

        for t in range(1, T):
            for i in range(self.N):
                delta[t, i] = self.B[i,self.O[t]] * np.array( [delta[t-1,j] * self.A[j,i]
                    for j in range(self.N)] ).max()
                psi[t,i] = np.array( [delta[t-1,j] * self.A[j,i]
                    for j in range(self.N)] ).argmax()

        P_T = delta[T-1, :].max()
        I[T-1] = delta[T-1, :].argmax()

        for t in range(T-2, -1, -1):
            I[t] = psi[t+1, I[t+1]]

        return I

    def forward(self):
        T = len(self.O)
        alpha = np.zeros((T, self.N), np.float)

        for i in range(self.N):
            alpha[0, i] = self.Pi[i] * self.B[i, self.O[0]]

        for t in range(T-1):
            for i in range(self.N):
                summation = 0   # for every i 'summation' should reset to '0'
                for j in range(self.N):
                    summation += alpha[t, j] * self.A[j, i]
                alpha[t+1, i] = summation * self.B[i, self.O[t+1]]

        summation = 0.0
        for i in range(self.N):
            summation += alpha[T-1, i]
        Polambda = summation
        return Polambda,alpha

    def backward(self):
        T = len(self.O)
        beta = np.zeros((T, self.N), np.float)
        for i in range(self.N):
            beta[T-1, i] = 1.0

        for t in range(T-2, -1, -1):
            for i in range(self.N):
                summation = 0.0     # for every i 'summation' should reset to '0'
                for j in range(self.N):
                    summation += self.A[i, j] * self.B[j, self.O[t+1]] * beta[t+1, j]
                beta[t, i] = summation

        Polambda = 0.0
        for i in range(self.N):
            Polambda += self.Pi[i] * self.B[i, self.O[0]] * beta[0, i]
        return Polambda, beta

    def compute_gamma(self, alpha, beta):
        T = len(self.O)
        gamma = np.zeros((T, self.N), np.float)       # the probability of Ot=q
        for t in range(T):
            for i in range(self.N):
                gamma[t, i] = alpha[t,i] * beta[t,i] / sum(
                    alpha[t, j] * beta[t, j] for j in range(self.N) )
        return gamma

    def compute_xi(self, alpha, beta):
        T = len(self.O)
        xi = np.zeros((T-1, self.N, self.N), np.float)  # note that: not T
        for t in range(T-1):   # note: not T
            for i in range(self.N):
                for j in range(self.N):
                    numerator = alpha[t, i] * self.A[i, j] * self.B[j, self.O[t+1]] * beta[t+1, j]
                    # the multiply term below should not be replaced by 'nummerator'，
                    # since the 'i,j' in 'numerator' are fixed.
                    # In addition, should not use 'i,j' below, to avoid error and confusion.
                    denominator = sum( sum(
                        alpha[t, i1] * self.A[i1, j1] * self.B[j1, self.O[t+1]] * beta[t+1, j1]
                        for j1 in range(self.N) )   # the second sum
                            for i1 in range(self.N) )   # the first sum
                    xi[t, i, j] = numerator / denominator
        return xi

    def Baum_Welch(self):
        # given O list finding lambda model(can derive T form O list)
        # also given N, M,
        T = len(self.O)
        V = [k for k in range(self.M)]

        # initialization - lambda
        # self.A = np.array(([[0,1,0,0],[0.4,0,0.6,0],[0,0.4,0,0.6],[0,0,0.5,0.5]]), np.float)
        # self.B = np.array(([[0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2]]), np.float)

        # mean value may not be a good choice
        # self.Pi = np.array(([1.0 / self.N] * self.N), np.float)  # must be 1.0 , if 1/3 will be 0
        # self.A = np.array([[1.0 / self.N] * self.N] * self.N) # must array back, then can use[i,j]
        # self.B = np.array([[1.0 / self.M] * self.M] * self.N)

        x = 1
        delta_lambda = x + 1
        times = 0
        # iteration - lambda
        while delta_lambda > x:  # x
            Polambda1, alpha = self.forward()           # get alpha
            Polambda2, beta = self.backward()           # get beta
            gamma = self.compute_gamma(alpha, beta)     # use alpha, beta
            xi = self.compute_xi(alpha, beta)

            lambda_n = [self.A, self.B, self.Pi]


            for i in range(self.N):
                for j in range(self.N):
                    numerator = sum(xi[t, i, j] for t in range(T-1))
                    denominator = sum(gamma[t, i] for t in range(T-1))
                    self.A[i, j] = numerator / denominator

            for j in range(self.N):
                for k in range(self.M):
                    numerator = sum(gamma[t, j] for t in range(T) if self.O[t] == V[k])  # TBD
                    denominator = sum(gamma[t, j] for t in range(T))
                    self.B[i, k] = numerator / denominator

            for i in range(self.N):
                self.Pi[i] = gamma[0, i]

            # if sum directly, there will be positive and negative offset
            delta_A = map(abs, lambda_n[0] - self.A)  # delta_A is still a matrix
            delta_B = map(abs, lambda_n[1] - self.B)
            delta_Pi = map(abs, lambda_n[2] - self.Pi)
            delta_lambda = sum([sum(sum(delta_A)), sum(sum(delta_B)), sum(delta_Pi) ])
            times += 1
            print times

        return self.A, self.B, self.Pi

if __name__ == "__main__":

    states = ('FZL', 'A', 'B', 'C', 'D', 'E', 'AB', 'AC', 'AD', 'AE', 'BC', 'BD', 'BE', 'CD', 'CE', 'DE', 'ABC', 'ABD',
              'ABE', 'ACD', 'ACE', 'ADE', 'BCD', 'BCE', 'BDE', 'CDE', 'ABCD', 'ABCE', 'ABDE', 'ACDE', 'BCDE', 'ABCDE', 'ZZZ')

    observations = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
    start_probability = {'zero': 0.1, 'one': 0.1, 'two': 0.1, 'three': 0.1, 'four': 0.1, 'five': 0.1, 'six': 0.1,
                         'seven': 0.1, 'eight': 0.1, 'nine': 0.1}

    # 已知观测序列与对应的状态序列, 训练得到hmm模型
    results = find_result()

    print results
    head = ["obseq", "mtstate"]
    data = pd.DataFrame(results, columns=head, dtype=np.int32)

    print data


    observe_seq, states_data = np.split(data.values, (1,), axis=1)
    states_label_index, states_index_label = generate_index_map(states)
    observations_label_index, observations_index_label = generate_index_map(observations)
    observations_index = convert_observations_to_index(observations, observations_label_index)

    pi = convert_map_to_vector(start_probability, observations_label_index)


#  通过监督学习获得模型参数
    hmm2 = HMM()
    hmm2.train(states_data.reshape(-1, 1), observe_seq.reshape(-1, 1), 33, 10)  # 未知初始状态或观测值

    where_are_nan = np.isnan(hmm2.A)
    hmm2.A[where_are_nan] = 0
    A = hmm2.A
    # print "A is --------------"
    # print A
    where_are_nan = np.isnan(hmm2.B)
    hmm2.B[where_are_nan] = 0
    B = hmm2.B
    pi = hmm2.pi

    # origin = sys.stdout
    # f = open('E:/3Dstate.txt', 'a')
    # sys.stdout = f
    # print "\n 参数 is The most possible states and probability are:"
    # print "A is -------------------"
    # np.set_printoptions()
    # A = np.array(A)
    # print A
    # print "B is -------------------"
    # print B
    # print "pi is -------------------"
    # print pi
    #
    # sys.stdout = origin
    # f.flush()
    # f.close()

    data2 = data.loc[:, ["obseq"]]
    X = -2
    data2 = np.array(data2.iloc[X:])

    data3 = data.loc[:, ["mtstate"]]
    data3 = np.array(data3.iloc[X:])
    prev_state = np.array(data3.reshape(1, len(data3)).tolist()[0])

    # observations_indexp = convert_observations_to_index(observations_pre, observations_label_index)
    for i in range(0, 10):
        observations_pre = data2.reshape(1, len(data2)).tolist()[0]
        observations_pre.append(i)

        V, p = hmm2.viterbi(observations_pre)
        print " " * 7, " ".join(("%10s" % observations_index_label[i]) for i in observations_pre)
        for s in range(1, 33):
            print "%7s: " % states_index_label[s] + " ".join("%10s" % ("%f" % v) for v in V[s])
        origin = sys.stdout
        f = open('E:/3Dstate.txt', 'a')
        sys.stdout = f
        print "\n Observe is " + str(i) + " The most possible states and probability are:"
        ss = hmm2.state_pathDIY(observations_pre, prev_state)
        print states_index_label[ss[0][0]]+"  "+states_index_label[ss[1][0]]+ "  概率最大:"+states_index_label[ss[2][0]]+\
               "  评估值为: " +"%7s " % ss[2][1]+ "  概率第二大:"+states_index_label[ss[3][0]]+\
               "  评估值为: " +"%7s " % ss[3][1]+ "  概率第三大:"+states_index_label[ss[4][0]]+\
               "  评估值为: " +"%7s " % ss[4][1]+ "  概率第四大:"+states_index_label[ss[5][0]]+\
               "  评估值为: " +"%7s " % ss[5][1]+ "  概率第五大:"+states_index_label[ss[6][0]]+\
               "  评估值为: " +"%7s " % ss[6][1]

        sys.stdout = origin
        del observations_pre[:]
        f.flush()
        f.close()



#  通过无监督学习Baum_Welch获得模型参数


    pi_baum =np.array([0, 0.082095387, 0.049257232, 0.106724003, 0.085613761, 0.045738858, 0.034010946, 0.071540266,
                       0.05629398, 0.021110242, 0.039874902, 0.02814699, 0.012900704, 0.080922596, 0.015246286,
                       0.00938233, 0.024628616, 0.017591869, 0.012900704, 0.051602815, 0.014073495, 0.008209539,
                       0.023455825, 0.003518374, 0.007036747, 0.008209539, 0.012900704, 0.007036747, 0.007036747,
                       0.003518374, 0.002345582, 0.002345582, 0.054730258])
    A_baum = np.array([[0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    B_baum = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

    observ_sec = np.array(data.loc[:, ["obseq"]]).reshape(1, len(data)).tolist()[0]
    model = hmm.MultinomialHMM(n_components=33, n_iter=20, tol=0.01)
    model.startprob_ = pi_baum
    model.transmat_ = A_baum
    model.emissionprob_ = B_baum

    model.fit(np.array(observ_sec).reshape(-1, 1))
    hmm_Baum_Welch = HMM(model.transmat_, model.emissionprob_, model.startprob_)
    print "\n 无监督学习统计模式预测"
    for i in range(0, 10):
        observations_baum = data2.reshape(1, len(data2)).tolist()[0]
        observations_baum.append(i)

        V, p = hmm_Baum_Welch.viterbi(observations_baum)
        print " " * 7, " ".join(("%10s" % observations_index_label[i]) for i in observations_baum)
        for s in range(1, 33):
            print "%7s: " % states_index_label[s] + " ".join("%10s" % ("%f" % v) for v in V[s])
        origin = sys.stdout
        f = open('E:/3Dstate.txt', 'a')
        sys.stdout = f

        print "百位数为 "+ str(i) + " 的概率为: " + str(math.exp(model.score(np.array(observations_baum).reshape(-1, 1))))
        print "\n Observe is " + str(i) + " The most possible states and probability are:"
        ss = hmm_Baum_Welch.state_pathDIY(observations_baum, prev_state)
        print states_index_label[ss[0][0]]+"  "+states_index_label[ss[1][0]]+ "  概率最大:"+states_index_label[ss[2][0]]+\
                "  评估值为: " +"%7s " % ss[2][1]+ "  概率第二大:"+states_index_label[ss[3][0]]+\
                "  评估值为: " +"%7s " % ss[3][1]+ "  概率第三大:"+states_index_label[ss[4][0]]+\
                "  评估值为: " +"%7s " % ss[4][1]+ "  概率第四大:"+states_index_label[ss[5][0]]+\
                "  评估值为: " +"%7s " % ss[5][1]+ "  概率第五大:"+states_index_label[ss[6][0]]+\
                "  评估值为: " +"%7s " % ss[6][1]

        sys.stdout = origin
        del observations_baum[:]
        f.flush()
        f.close()

    # print "\n Observe is " + str(i) + " The most possible Next hundred and probability are:"
    # print "百位数为 "+ str(i) + " 的概率为: " + str(math.exp(model1.score(np.array(observations_pre).reshape(-1, 1))))
    # print math.exp(model.score(np.array([1, 9, 9, 0]).reshape(-1, 1)))
    #
    # print model.predict(np.array([1, 9, 9, 0]).reshape(-1, 1))


    #
    # #判断按照隐马模型预测模型（取概率大值）的准确率

    #     seen = data[t-7:t, 0]
    #     seen1 = seen
    #
    #     nowV = math.exp(model.score(seen.reshape(-1, 1)))
    #     if seen1[-1:][0] == 0:
    #         seen1[-1:] = 1
    #     else:
    #         seen1[-1:] = 0
    #     compV = math.exp(model.score(seen1.reshape(-1, 1)))
    #     if seen1[-1:][0] == 0:
    #         seen1 [-1:] = 1
    #     else:
    #         seen1[-1:] = 0
    #     if nowV > compV:
    #         predict_suc = predict_suc + 1
    #     else:
    #         predict_fail = predict_fail + 1
    #
    #     print "model1 suc============" + str(predict_suc)
    #     print "model1 fail============" + str(predict_fail)
    # print "model1 成功率==============" + str(float(predict_suc)/float( data[:, 0].size-7))
    #
    #     # print model.decode(observe_seq.reshape(-1, 1),algorithm="viterbi")
    #     # print model.predict_proba(seen)
    # print "predict========================"
    # #
    # #
    # # hmm_baum = HMM(A, B, pi)
    # # # hmm_baum.baum_welch_train(observe_seq, 0.001)
    # # # A1 = hmm_baum.A
    # # # B1 = hmm_baum.B
    # # # pi1 = hmm_baum.pi
    # # # hmm_baum = HMM(A1, B1, pi1)
    # # # hmm_baum.printhmm()
    # # states_out = hmm2.state_path(observe_seq)[1]
    # # p = 0.0
    # # for s in states_data:
    # #     if next(states_out) == s: p += 1
    # #
    # # print "未监督学习正确率：" + str(p / len(states_data))
    # #
    # #
    # # model2 = hmm.MultinomialHMM(n_components=6, n_iter=300, tol=0.001)
    # # model2.fit(observe_seq.reshape(-1, 1))
    # # print model2.startprob_
    # # print model2.transmat_
    # # print model2.emissionprob_
    # # print model2.score(observe_seq.reshape(-1, 1))
    # # model2.fit(observe_seq.reshape(-1, 1))
    # # print model2.startprob_
    # # print model2.transmat_
    # # print model2.emissionprob_
    # # print model2.score(observe_seq.reshape(-1, 1))
    # # model2.fit(observe_seq.reshape(-1, 1))
    # # print model2.startprob_
    # # print model2.transmat_
    # # print model2.emissionprob_
    # # print model2.score(observe_seq.reshape(-1, 1))

