from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        for i in range(S):
            alpha[i,0] = self.pi[i] * self.B[i,self.obs_dict[Osequence[0]]]

        for t in range(1,L):
            for i in range(S):
                alpha[i,t] =  self.B[i,self.obs_dict[Osequence[t]]] * sum([self.A[k, i] * alpha[k, t - 1] for k in range(S)])
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        for i in range(S):
            beta[i, L-1] = 1

        for t in reversed(range(L-1)):
            for i in range(S):
                beta[i,t] = sum([beta[j, t + 1] * self.A[i, j] * self.B[j, self.obs_dict[Osequence[t + 1]]] for j in range(S)])
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        prob = sum(alpha[:, -1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta  = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        for i in range(S):
            for j in range(L):
                prob[i,j] = alpha[i,j] * beta[i,j] / seq_prob
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        for i in range(S):
            for j in range(S):
                for k in range(L-1):
                    prob[i,j,k] = alpha[i,k] * self.A[i,j] * self.B[j, self.obs_dict[Osequence[k+1]]] * beta[j,k+1] / seq_prob
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        Delta = np.zeros([S, L], dtype="int")
        delta[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]

        for t in range(1, L):
            for i in range(S):
                delta[i, t] = self.B[i, self.obs_dict[Osequence[t]]] * np.max(self.A[:, i] * delta[:, t - 1])
                Delta[i, t] = np.argmax(self.A[:, i] * delta[:, t - 1])
        z = np.argmax(delta[:, L - 1])
        path.append(z)

        for t in range(L - 1, 0, -1):
            z = Delta[z, t]
            path.append(z)
        path = path[::-1]

        new_dict = {v : k for k, v in self.state_dict.items()}
        for i in range(len(path)):
            path[i] = new_dict[path[i]]

        ###################################################
        return path
