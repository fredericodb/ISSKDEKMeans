from __future__ import division

import logging

import copy
import numpy as np
import networkx as nx
import scipy.spatial.distance as sp

# TODO should have a _get_activation_trajectories method too

class gwr():

    '''
    Growing When Required (GWR) Neural Gas, after [1]. Constitutes the base class
    for the Online Semi-supervised (OSS) GWR.

    [1] Parisi, G. I., Tani, J., Weber, C., & Wermter, S. (2017).
    Emergence of multimodal action representations from neural network
    self-organization. Cognitive Systems Research, 43, 208-221.
    '''

    def __init__(self, act_thr = 0.35, fir_thr = 0.1, eps_b = 0.1,
                 eps_n = 0.01, tau_b = 0.3, tau_n = 0.1, kappa = 1.05,
                 lab_thr = 0.5, max_age = 100, max_size = 100,
                 random_state = None):
        self.act_thr  = act_thr
        self.fir_thr  = fir_thr
        self.eps_b    = eps_b
        self.eps_n    = eps_n
        self.tau_b    = tau_b
        self.tau_n    = tau_n
        self.kappa    = kappa
        self.lab_thr  = lab_thr
        self.max_age  = max_age
        self.max_size = max_size
        if random_state is not None:
            np.random.seed(random_state)

    def _initialize(self, X):

        logging.info('Initializing the neural gas.')
        self.G = nx.Graph()
        # TODO: initialize empty labels?
        draw = np.random.choice(X.shape[0], size=2, replace=False)
        self.G.add_node(0, pos=X[draw[0], :])
        self.G.nodes[0]['fir'] = 1
        self.G.add_node(1, pos=X[draw[1], :])
        self.G.nodes[1]['fir'] = 1
        # self.G.add_node(0,attr_dict={'pos' : X[draw[0],:],
        #                              'fir' : 1})
        # self.G.add_node(1,attr_dict={'pos' : X[draw[1],:],
        #                              'fir' : 1})

    def get_positions(self):
        pos = np.array(list(nx.get_node_attributes(self.G, 'pos').values()))
        return pos

    def _get_best_matching(self, x):
        pos = self.get_positions()
        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)
        b = sorted_dist[0, 0]
        s = sorted_dist[0, 1]
        # b = self.G.nodes()[sorted_dist[0,0]]
        # s = self.G.nodes()[sorted_dist[0,1]]
        return b, s


    def _get_activation(self, x, b):
        p = self.G.nodes[b]['pos'][np.newaxis, :]
        dist = sp.cdist(x, p, metric='euclidean')[0, 0]
        act = np.exp(-dist)
        return act


    def _make_link(self, b, s):
        self.G.add_edge(b, s, age=0)
        # self.G.add_edge(b,s,attr_dict={'age':0})


    def _add_node(self, x, b, s):
        r = max(self.G.nodes()) + 1
        pos_r = 0.5 * (x + self.G.nodes[b]['pos'])[0, :]
        self.G.add_node(r, pos=pos_r)
        self.G.nodes[r]['fir'] = 1
        # self.G.add_node(r, attr_dict={'pos' : pos_r, 'fir' : 1})
        self.G.remove_edge(b, s)
        self.G.add_edge(r, b, age=0)
        self.G.add_edge(r, s, age=0)
        # self.G.add_edge(r, b, attr_dict={'age':0})
        # self.G.add_edge(r, s, attr_dict={'age':0})
        return r


    def _update_network(self, x, b):
        dpos_b = self.eps_b * self.G.nodes[b]['fir'] * (x - self.G.nodes[b]['pos'])
        self.G.nodes[b]['pos'] = self.G.nodes[b]['pos'] + dpos_b[0, :]

        neighbors = self.G.neighbors(b)
        for n in neighbors:
            # update the position of the neighbors
            dpos_n = self.eps_n * self.G.nodes[n]['fir'] * (x - self.G.nodes[n]['pos'])
            self.G.nodes[n]['pos'] = self.G.nodes[n]['pos'] + dpos_n[0, :]

            # increase the age of all edges connected to b
            self.G.edges[b, n]['age'] += 1


    def _update_firing(self, b):
        dfir_b = self.tau_b * self.kappa * (1 - self.G.nodes[b]['fir']) - self.tau_b
        self.G.nodes[b]['fir'] = self.G.nodes[b]['fir'] + dfir_b

        neighbors = self.G.neighbors(b)
        for n in neighbors:
            dfir_n = self.tau_n * self.kappa * (1 - self.G.nodes[b]['fir']) - self.tau_n
            self.G.nodes[n]['fir'] = self.G.nodes[n]['fir'] + dfir_n


    def _remove_old_edges(self):
        edges_r = []
        for e in self.G.edges():
            if self.G[e[0]][e[1]]['age'] > self.max_age:
                edges_r.append(e)
        self.G.remove_edges_from(edges_r)
        nodes_r = []
        for node in self.G.nodes():
            if len(self.G.edges(node)) == 0:
                logging.debug('Removing node %s', str(node))
                nodes_r.append(node)
        self.G.remove_nodes_from(nodes_r)

        # for e in self.G.edges():
        #     if self.G[e[0]][e[1]]['age'] > self.max_age:
        #         self.G.remove_edge(*e)
        #         for node in self.G.nodes():
        #             if len(self.G.edges(node)) == 0:
        #                 logging.debug('Removing node %s', str(node))
        #                 self.G.remove_node(node)

    def _check_stopping_criterion(self):
        # TODO: implement this
        pass

    def _training_step(self, x):
        # TODO: do not recompute all positions at every iteration
        b, s = self._get_best_matching(x)
        self._make_link(b, s)
        act = self._get_activation(x, b)
        fir = self.G.nodes[b]['fir']
        logging.debug('Training step - best matching: %s, %s \n'
                      'Network activation: %s \n'
                      'Firing: %s', str(b), str(s), str(np.round(act, 3)),
                      str(np.round(fir,3)))
        if act < self.act_thr and fir < self.fir_thr \
            and len(self.G.nodes()) < self.max_size:
            r = self._add_node(x, b, s)
            logging.debug('GENERATE NODE %s', self.G.nodes[r])
        else:
            self._update_network(x, b)
        self._update_firing(b)
        self._remove_old_edges()


    def train(self, X, n_epochs=20, warm_start = False):
        if not warm_start:
            self._initialize(X)
        for n in range(n_epochs):
            logging.info('>>> Training epoch %s', str(n))
            for i in range(X.shape[0]):
                x = X[i,np.newaxis]
                self._training_step(x)
                self._check_stopping_criterion()
        logging.info('Training ended - Network size: %s', len(self.G.nodes()))



class oss_gwr(gwr):

    '''
    Online Semi-supervised Growing When Required Neural Gas, after [1].
    Adds the label propagation methods to the `gwr` class.

    [1] Parisi, G. I., Tani, J., Weber, C., & Wermter, S. (2017).
    Emergence of multimodal action representations from neural network
    self-organization. Cognitive Systems Research, 43, 208-221.
    '''

    def _initialize(self, X):

        logging.info('Initializing the neural gas.')
        self.G = nx.Graph()
        # TODO: initialize empty labels?
        draw = np.random.choice(X.shape[0], size=2, replace=False)
        self.G.add_node(0, pos=X[draw[0], :])
        self.G.nodes[0]['fir'] = 1
        self.G.nodes[0]['lab'] = -1
        self.G.add_node(1, pos=X[draw[1], :])
        self.G.nodes[1]['fir'] = 1
        self.G.nodes[1]['lab'] = -1
        # self.G.add_node(0,attr_dict={'pos' : X[draw[0],:],
        #                              'fir' : 1,
        #                              'lab' : -1})
        # self.G.add_node(1,attr_dict={'pos' : X[draw[1],:],
        #                              'fir' : 1,
        #                              'lab' : -1})


    def _add_node(self, x, b, s):
        r = max(self.G.nodes()) + 1
        pos_r = 0.5 * (x + self.G.nodes[b]['pos'])[0, :]
        self.G.add_node(r, pos=pos_r)
        self.G.nodes[r]['fir'] = 1
        self.G.nodes[r]['lab'] = -1
        # self.G.add_node(r, attr_dict={'pos' : pos_r, 'fir' : 1, 'lab' : -1})
        self.G.remove_edge(b, s)
        self.G.add_edge(r, b, age=0)
        self.G.add_edge(r, s, age=0)
        # self.G.add_edge(r, b, attr_dict={'age':0})
        # self.G.add_edge(r, s, attr_dict={'age':0})
        return r


    def _assign_label(self, r, e, b):
        if e != -1:
            self.G.nodes[r]['lab'] = e
        else:
            self.G.nodes[r]['lab'] = self.G.nodes[b]['lab']


    def _update_label(self, b, e, s):
        pi = self._get_label_propagation_coeff(b, s)

        if e != -1:
            self.G.nodes[b]['lab'] = e
        elif e == -1 and pi >= self.lab_thr:
            self.G.nodes[b]['lab'] = self.G.nodes[s]['lab']
        # else it keeps its own label


    def _get_label_propagation_coeff(self, b, s):
        E = 1 if s in self.G.neighbors(b) else 0
        pi = E / (1 + self.G.nodes[b]['fir'] + self.G.nodes[s]['fir'])
        return pi


    def _training_step(self, x, e = -1):
        # TODO: do not recompute all positions at every iteration
        b, s = self._get_best_matching(x)
        self._make_link(b, s)
        act = self._get_activation(x, b)
        fir = self.G.nodes[b]['fir']
        logging.debug('Training step - best matching: %s, %s \n'
                      'Network activation: %s \n'
                      'Firing: %s', str(b), str(s), str(np.round(act, 3)),
                      str(np.round(fir,3)))
        if act < self.act_thr and fir < self.fir_thr:
            r = self._add_node(x, b, s)
            logging.debug('GENERATE NODE %s', self.G.nodes[r])
            self._assign_label(r, e, b)
        else:
            self._update_network(x, b)
            self._update_label(b, e, s)
        self._update_firing(b)
        self._remove_old_edges()


    def _predict_observation(self, x):
        b, s = self._get_best_matching(x)
        return self.G.nodes[b]['lab']


    def train(self, X, y, n_epochs=20, warm_start = False):
        if not warm_start:
            self._initialize(X)
        for n in range(n_epochs):
            logging.info('>>> Training epoch %s', str(n))
            for i in range(X.shape[0]):
                x = X[i, np.newaxis]
                e = y[i]
                self._training_step(x, e)
                self._check_stopping_criterion()
        logging.info('Training ended - Network size: %s', len(self.G.nodes()))


    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            y_pred[i] = self._predict_observation(x)
        return y_pred


    # TODO link up hierarchically
    # TODO implement a grid search

class oss_gwr2(oss_gwr):
    def _assign_label(self, r, e, b):
        if e != -1:
            self.G.nodes[r]['lab'] = e
        else:
            # TODO conditions to assign label to unlabeled example
            self.G.nodes[r]['lab'] = self.G.nodes[b]['lab']





