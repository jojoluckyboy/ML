# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from hmmlearn import hmm
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics.pairwise import pairwise_distances_argmin
import warnings

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
        path = list(self.build_viterbi_path(prev, last_state))

        return V[last_state,-1], reversed(path)

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

            xi = np.zeros((n_states,n_states,n_samples-1))
            for t in range(n_samples-1):
                denom = np.dot(np.dot(alpha[:,t].T, self.A) * self.B[:,observations[t+1]].T, beta[:,t+1])
                for i in range(n_states):
                    numer = alpha[i,t] * self.A[i,:] * self.B[:,observations[t+1]].T * beta[:,t+1].T
                    xi[i,:,t] = numer / denom

            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.sum(xi,axis=1)
            # Need final gamma element for new B
            prod =  (alpha[:,n_samples-1] * beta[:,n_samples-1]).reshape((-1,1))
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
            self.printhmm()


if __name__ == "__main__":

    states = ('7day_RR', '7day_RQ', '7day_RP', '7day_QP', '7day_QR', '7day_QQ')

    observations = ('pre_fail', 'pre_success')
    start_probability = {'pre_fail': 0.5, 'pre_success': 0.5}

    # 已知观测序列与对应的状态序列, 训练得到hmm模型
    data = np.loadtxt('hmmodletest.data', dtype=int, delimiter=',', skiprows=7, usecols=(0, 1, 2))
    predict_suc = 0
    predict_fail = 0

    for t in range(30, data[:, 0].size-7):
        up_down = data[:t, 1]
        states_data = data[:t, 2]
        observe_seq = data[:t, 0]

        states_label_index, states_index_label = generate_index_map(states)
        observations_label_index, observations_index_label = generate_index_map(observations)

        observations_index = convert_observations_to_index(observations, observations_label_index)
        pi = convert_map_to_vector(start_probability, observations_label_index)

        hmm2 = HMM()
        hmm2.train(states_data.reshape(-1, 1), observe_seq.reshape(-1, 1), 6, 2)  # 未知初始状态或观测值

        A = hmm2.A
        B = hmm2.B
        pi = hmm2.pi
        hmm2 = HMM(A, B, pi)



        # states_out = hmm2.state_path(observe_seq)[1]
        # p = 0.0
        # for s in states_data:
        #     if next(states_out) == s: p += 1
        #
        # print "监督学习正确率：" + str(p / len(states_data))
        model = hmm.MultinomialHMM(n_components=6)
        model.startprob_ = pi
        model.transmat_ = A
        model.emissionprob_ = B
        #判断按照隐马模型预测模型（取概率大值）的准确率

        seen = data[t-7:t, 0]
        seen1 = seen

        nowV = math.exp(model.score(seen.reshape(-1, 1)))
        if seen1[-1:][0] == 0:
            seen1[-1:] = 1
        else:
            seen1[-1:] = 0
        compV = math.exp(model.score(seen1.reshape(-1, 1)))
        if seen1[-1:][0] == 0:
            seen1 [-1:] = 1
        else:
            seen1[-1:] = 0
        if nowV > compV:
            predict_suc = predict_suc + 1
        else:
            predict_fail = predict_fail + 1

        print "model1 suc============" + str(predict_suc)
        print "model1 fail============" + str(predict_fail)
    print "model1 成功率==============" + str(float(predict_suc)/float( data[:, 0].size-7))

        # print model.decode(observe_seq.reshape(-1, 1),algorithm="viterbi")
        # print model.predict_proba(seen)
    print "predict========================"
    #
    #
    # hmm_baum = HMM(A, B, pi)
    # # hmm_baum.baum_welch_train(observe_seq, 0.001)
    # # A1 = hmm_baum.A
    # # B1 = hmm_baum.B
    # # pi1 = hmm_baum.pi
    # # hmm_baum = HMM(A1, B1, pi1)
    # # hmm_baum.printhmm()
    # states_out = hmm2.state_path(observe_seq)[1]
    # p = 0.0
    # for s in states_data:
    #     if next(states_out) == s: p += 1
    #
    # print "未监督学习正确率：" + str(p / len(states_data))
    #
    #
    # model2 = hmm.MultinomialHMM(n_components=6, n_iter=300, tol=0.001)
    # model2.fit(observe_seq.reshape(-1, 1))
    # print model2.startprob_
    # print model2.transmat_
    # print model2.emissionprob_
    # print model2.score(observe_seq.reshape(-1, 1))
    # model2.fit(observe_seq.reshape(-1, 1))
    # print model2.startprob_
    # print model2.transmat_
    # print model2.emissionprob_
    # print model2.score(observe_seq.reshape(-1, 1))
    # model2.fit(observe_seq.reshape(-1, 1))
    # print model2.startprob_
    # print model2.transmat_
    # print model2.emissionprob_
    # print model2.score(observe_seq.reshape(-1, 1))

