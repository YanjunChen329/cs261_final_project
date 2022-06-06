import random
import networkx as nx
from networkx.algorithms.tree.mst import SpanningTreeIterator
from link_cut_tree import Node, build_link_cut_tree
import copy
import numpy as np
import matplotlib.pyplot as plt


def sample_spanning_tree2(G, title, seed=0):
    random.seed(seed)

    # generate random spanning tree via DFS
    spanning_tree_iter = SpanningTreeIterator(G)
    Ts = []

    i = 0
    num_eval = 1
    T_ref = None
    for T in spanning_tree_iter:
        if i == 0:
            T_ref = T
            i += 1
        elif len(Ts) < num_eval:
            Ts.append(copy.deepcopy(T))
        else:
            break
    V = len(T_ref.edges)

    trees = random.sample(Ts, k=num_eval)
    distances = [[] for _ in range(num_eval)]
    for e in range(num_eval):
        T = trees[e]
        t = 10000
        counter = {}
        for i in range(t):
            print()
            T = Anari_algorithm(G, T, q=0)
            # convergence analysis
            # key = tree_difference(T_ref, T)
            key = tuple(sorted(list(T.edges)))
            counter[key] = counter.get(key, 0) + 1
            dist = calculate_pseudo_distance(counter, t, V)
            distances[e].append(dist)
            # print(dist)
    dist_array = np.array(distances)
    # average over 10 trees
    # dist_avg = dist_array.sum(axis=0) / float(dist_array.shape[0])
    dist_max = np.max(dist_array, axis=0)
    plot_distance(dist_max, title=title)

def sample_spanning_tree(G, seed=0):
    random.seed(seed)

    # generate random spanning tree via DFS
    spanning_tree_iter = SpanningTreeIterator(G)
    Ts = []

    i = 0
    num_eval = 10
    T_ref = None
    for T in spanning_tree_iter:
        if i == 0:
            T_ref = T
            i += 1
        elif len(Ts) < num_eval:
            Ts.append(copy.deepcopy(T))
        else:
            break
    V = len(T_ref.edges)

    trees = random.sample(Ts, k=num_eval)
    distances = [[] for _ in range(num_eval)]
    for e in range(num_eval):
        T = trees[e]
        iter = 100
        t = 10000
        for i in range(iter):
            T = Anari_algorithm(G, T, q=0)
            # convergence analysis
            if i > 1:
                counter = {}
                T_clone = copy.deepcopy(T)
                random_state = random.getstate()
                random.seed(random.randint(0, 100000))
                for _ in range(t):
                    T_clone = Anari_algorithm(G, T_clone, q=0)
                    key = tree_difference(T_ref, T_clone)
                    counter[key] = counter.get(key, 0) + 1
                dist = calculate_pseudo_variation_distance(counter, t, V)
                distances[e].append(dist)
                random.setstate(random_state)
                # print(dist)
    dist_array = np.array(distances)
    # average over 10 trees
    dist_avg = dist_array.sum(axis=0) / float(dist_array.shape[0])
    plot_distance(dist_avg)


def tree_difference(T_ref, T):
    diff = set(T_ref.edges).difference(set(T.edges))
    return len(diff)

def calculate_pseudo_variation_distance(counter, t, V):
    uniform_prob = 1. / V
    distance = 0
    for v in range(V):
        prob = counter.get(v, 0) / t
        distance += 0.5 * abs(prob - uniform_prob)
    return distance

def calculate_pseudo_distance(counter, t, V):
    uniform_prob = 1. / len(counter)
    distance = 0
    for k, v in counter.items():
        prob = v / t
        distance += 0.5 * abs(prob - uniform_prob)
    return distance


def Anari_algorithm(G, T, q=0):
    G_edges = set(G.edges)
    n = len(G_edges)
    # do nothing with prob |T| / n
    if random.random() < len(T.edges) / float(n):
        return T
    T_edges = set(T.edges)
    E = list(G_edges.difference(T_edges))
    # choose edge uniformly from E
    edge = random.choice(E)
    T.add_edge(*edge)
    # find cycle
    try:
        cycle = nx.find_cycle(T, source=edge[0])
        edge_to_remove = random.choice(cycle)
        T.remove_edge(*edge_to_remove)
    except nx.NetworkXNoCycle:
        if random.random() < q / (q+1.):
            if len(T.edges) > 0:
                edge_to_remove = random.choice(list(T.edges))
                T.remove_edge(*edge_to_remove)
    return T


def plot_distance(dists, title):
    plt.plot(np.arange(dists.shape[0]), dists)
    plt.xlabel("t")
    plt.ylabel("pseudo variation distance")
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # G = nx.complete_graph(n=5)
    # print(G)
    # sample_spanning_tree2(G, "complete graph with 5 nodes")

    G = nx.erdos_renyi_graph(n=20, p=0.3, seed=0)
    print(G)
    sample_spanning_tree2(G, f"Erdo Renyi (|V|={len(G.nodes)}, |E|={len(G.edges)})")


