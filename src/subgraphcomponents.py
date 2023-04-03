"""Subgraph components module."""

import time
import random
import pandas as pd
import networkx as nx
from diffusiontrees import EulerianDiffuser

class SubGraphComponents:
    """
    Methods separate the original graph and run diffusion on each node in the subgraphs.
    """
    def __init__(self, edge_list_path, seeding, vertex_set_cardinality):
        """
        Initializing the object with the main parameters.
        :param edge_list_path: Path to the csv with edges.
        :param seeding: Random seed.
        :param vertex_set_cardinality: Number of unique nodes per tree.
        """
        self.seed = seeding
        self.vertex_set_cardinality = vertex_set_cardinality
        self.read_start_time = time.time()
        self.graph = nx.from_edgelist(pd.read_csv(edge_list_path, index_col=None).values.tolist(), create_using=nx.DiGraph)
        self.og_graph = nx.from_edgelist(pd.read_csv(edge_list_path, index_col=None).values.tolist(), create_using=nx.DiGraph)
        self.counts = len(self.graph.nodes())+1
        self.separate_subcomponents()
        self.single_feature_generation_run()

    def separate_subcomponents(self):
        """
        Finding the connected components.
        """
        comps = [self.graph.subgraph(c) for c in nx.strongly_connected_components(self.graph)]
        self.graph = sorted(comps, key=len, reverse=True)
        self.read_time = time.time()-self.read_start_time


    def random_walk_directed(self, graph,  node, walk_lenght):
        walk = []
        current_node = node
        for i in range(walk_lenght):
            walk.append(str(current_node))
            neighbors = list(graph.successors(current_node))
            if len(neighbors) != 0:
                # select a neighbor at random to move to
                next_node = random.choice(neighbors)
                # set the next node as the current node for the next step of the walk
                current_node = next_node
        return walk
    
    def random_walk_directed_with_restart(self, graph,  node, walk_lenght, tries = 2):
        walk = []
        dead_end = 0
        current_node = node
        for i in range(walk_lenght):
            walk.append(str(current_node))
            neighbors = list(graph.successors(current_node))
            if len(neighbors) != 0:
                next_node = random.choice(neighbors)
                current_node = next_node
            else:
                dead_end += 1
                if dead_end == tries:
                    current_node = node
        return walk

    def single_feature_generation_run2(self):
        """
        Running a round of diffusions and measuring the sequence generation performance.
        """
        random.seed(self.seed)
        self.generation_start_time = time.time()
        self.paths = {}
        for sub_graph in self.graph:
            current_cardinality = len(sub_graph.nodes())
            if current_cardinality < self.vertex_set_cardinality:
                self.vertex_set_cardinality = current_cardinality
            diffuser = EulerianDiffuser(sub_graph, self.vertex_set_cardinality)
            self.paths.update(diffuser.diffusions)
        self.paths = [v for k, v in self.paths.items()]
        self.generation_time = time.time() - self.generation_start_time

    def single_feature_generation_run_basic_random_walk(self):
        """
        Running a round of diffusions and measuring the sequence generation performance.
        """
        random.seed(self.seed)
        self.generation_start_time = time.time()
        self.paths = {}
        for n in self.og_graph.nodes():
            walk = self.random_walk_directed(self.og_graph, n, 10)
            self.paths.update({n : walk})
        self.paths = [v for k, v in self.paths.items()]
        print(self.paths)
        self.generation_time = time.time() - self.generation_start_time

    def single_feature_generation_run(self):
        """
        Running a round of diffusions and measuring the sequence generation performance.
        """
        random.seed(self.seed)
        self.generation_start_time = time.time()
        self.paths = {}
        for n in self.og_graph.nodes():
            walk = self.random_walk_directed_with_restart(self.og_graph, n, 10)
            self.paths.update({n : walk})
        self.paths = [v for k, v in self.paths.items()]
        print(self.paths)
        self.generation_time = time.time() - self.generation_start_time
    