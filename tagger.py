import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    # Edit here
    # create obs dict
    obs_dict = {}
    obs_index = 0
    for sentence in train_data:
        for word in sentence.words:
            if word not in obs_dict.keys():
                obs_dict[word] = obs_index
                obs_index += 1

    # create state dict
    state_dict = {}
    for index, name in enumerate(tags):
        state_dict[name] = index

    # create pi
    pi = np.zeros(len(tags))
    for sentence in train_data:
        state_index = state_dict[sentence.tags[0]]
        pi[state_index] += 1
    pi = pi / len(train_data)

    # create transition matrix A
    N = len(tags)
    A = np.zeros((N,N))
    for sentence in train_data:
        for index, state in enumerate(sentence.tags[0:len(sentence.tags)-1]):
            A[state_dict[state]][state_dict[sentence.tags[index + 1]]] += 1

    for i in A:
        sigma = np.sum(i)
        if sigma == 0:
            continue
        else:
            i /= sigma

    # create emission matrix B
    S = len(obs_dict)
    B = np.zeros((N,S))
    for sentence in train_data:
        for index, state in enumerate(sentence.tags):
            B[state_dict[state]][obs_dict[sentence.words[index]]] += 1

    for j in B:
        sigma = np.sum(j)
        if sigma == 0:
            continue
        else:
            j /= sigma

    model = HMM(pi,A,B,obs_dict,state_dict)


    ###################################################
    return model
def sentence_tagging(test_data, model, tags):
    """
# TODO:
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    # Edit here
    for sentence in test_data:

        for word in sentence.words:
            if word not in model.obs_dict:
                new_word = np.full((len(tags),1),1e-6)
                model.B = np.insert(model.B, -1, values = new_word, axis = 1)
                model.obs_dict[word] = len(model.obs_dict)
        tag = model.viterbi(sentence.words)
        tagging.append(tag)


    ###################################################
    return tagging
