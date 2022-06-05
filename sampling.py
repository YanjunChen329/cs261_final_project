import random
import networkx as nx
from networkx.algorithms.tree.mst import SpanningTreeIterator
from link_cut_tree import Node, build_link_cut_tree
import copy


def sample_spanning_tree(G, seed=0):
    random.seed(seed)

    # generate random spanning tree via DFS
    spanning_tree_iter = SpanningTreeIterator(G)
    all_spanning_trees = []

    for T in spanning_tree_iter:
        all_spanning_trees.append(copy.deepcopy(T))

    T = random.choice(all_spanning_trees)

    num_eval = 3
    t = 100
    for i in range(100002):
        T = Anari_algorithm(G, T, q=0)

        # convergence analysis
        if i > 100000:
            random_state = random.getstate()
            counter = {}
            for i in range(num_eval):
                random.seed(random.randint(0, 10000))
                T_clone = copy.deepcopy(T)
                for _ in range(t):
                    T_clone = Anari_algorithm(G, T_clone, q=0)
                    key = tuple(sorted(list(T_clone.edges)))
                    counter[key] = counter.get(key, 0) + 1
            print(counter.values())
            random.setstate(random_state)



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
        if random.random() < q * (q+1):
            if len(T.edges) > 0:
                edge_to_remove = random.choice(list(T.edges))
                T.remove_edge(*edge_to_remove)
    return T


def evaluate_markov_chain():
    pass


if __name__ == '__main__':
    G = nx.complete_graph(n=5)
    print(G.nodes)
    print(G.edges)

    sample_spanning_tree(G)
