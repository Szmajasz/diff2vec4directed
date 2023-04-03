"""Helper functions."""

import argparse
import numpy as np
from tqdm import tqdm
from texttable import Texttable
from gensim.models.doc2vec import TaggedDocument

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Facebook Restaurants network.
    The default hyperparameters give a high quality representation already without grid search.
    :return : Object with hyperparameters.
    """

    parser = argparse.ArgumentParser(description="Run diffusion2vec.")

    parser.add_argument("--input",
                        nargs="?",
                        default="./data/restaurant_edges.csv",
	                help="Input graph path")

    parser.add_argument("--output",
                        nargs="?",
                        default="./output/restaurant.csv",
	                help="Embeddings path")

    parser.add_argument("--model",
                        nargs="?",
                        default="pooled",
	                help="Model type.")

    parser.add_argument("--directed",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Model type.")

    parser.add_argument("--dimensions",
                        type=int,
                        default=128,
	                help="Number of dimensions. Default is 128.")

    parser.add_argument("--vertex-set-cardinality",
                        type=int,
                        default=40,
	                help="Length of diffusion per source is 2*cardianlity-1. Default is 40.")

    parser.add_argument("--num-diffusions",
                        type=int,
                        default=10,
	                help="Number of diffusions per source. Default is 10.")

    parser.add_argument("--window-size",
                        type=int,
                        default=10,
                    	help="Context size for optimization. Default is 10.")

    parser.add_argument("--iter",
                        default=1,
                        type=int,
                        help="Number of epochs in ASGD. Default is 1.")

    parser.add_argument("--workers",
                        type=int,
                        default=4,
	                help="Number of cores. Default is 4.")

    parser.add_argument("--alpha",
                        type=float,
                        default=0.025,
	                help="Initial learning rate. Default is 0.025.")

    return parser.parse_args()


def argument_printer(args):
    """
    Function to print the arguments in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def generation_tab_printer(read_times, generation_times):
    """
    Function to print the time logs in a nice tabular format.
    :param read_times: List of reading times.
    :param generation_times: List of generation times.
    """
    t = Texttable()
    t.add_rows([["Metric", "Value"],
                ["Mean graph read time:", np.mean(read_times)],
                ["Standard deviation of read time.", np.std(read_times)]])
    print(t.draw())
    t = Texttable()
    t.add_rows([["Metric", "Value"],
                ["Mean sequence generation time:", np.mean(generation_times)],
                ["Standard deviation of generation time.", np.std(generation_times)]])
    print(t.draw())

def result_processing(results):
    """
    Function to separate the sequences from time measurements and process them.
    :param results: List of 3-length tuples including the sequences and results.
    :return walk_results: List of random walks.
    :return counts: Number of nodes.
    """
    walk_results = [res[0] for res in results]
    read_time_results = [res[1] for res in results]
    generation_time_results = [res[2] for res in results]
    counts = [res[3] for res in results]
    nodes = list(results[0][4])

    generation_tab_printer(read_time_results, generation_time_results)
    walk_results = [walk for walks in walk_results for walk in walks]
    return walk_results, counts, nodes


def process_non_pooled_model_data(walks, counts, args):
    """
    Function to extract proximity statistics.
    :param walks: Diffusion lists.
    :param counts: Number of nodes.
    :param args: Arguments objects.
    :return docs: Processed walks.
    """
    print("Run feature extraction across windows.")
    features = {str(node): [] for node in range(counts[0])}
    for walk in tqdm(walks):
        for i in range(len(walk)-args.window_size):
            for j in range(1, args.window_size+1):
                features[walk[i]].append(["+"+str(j)+"_"+walk[i+j]])
                features[walk[i+j]].append(["_"+str(j)+"_"+walk[i]])

    docs = [TaggedDocument(words=[x[0] for x in v], tags=[str(k)]) for k, v in features.items()]
    return docs

def process_raw_frequency_dict(raw_frequency_dict):
    """
    Function to extract proximity statistics.
    :param raw_frequency_dict: Dictionary with information about visits in local nodes for each node
    """
    print("Run feature extraction across windows.")
    nodes = raw_frequency_dict.keys()
    features = {str(node): [] for node in nodes}
    for node, frequencies in raw_frequency_dict.items():
        for subgraph_node, n_visits in raw_frequency_dict[node].items():
            for _ in range(n_visits):
                features[str(node)].append(["+1"+"_"+str(subgraph_node)])
                features[str(subgraph_node)].append(["_1"+"_"+str(node)])

    docs = [TaggedDocument(words=[x[0] for x in v], tags=[str(k)]) for k, v in features.items()]
    return docs


def successors_search(g, node, max_neighbours=5, max_depth=3):
    def step(g, node, visits, depth, max_neighbours, max_depth):
        if depth > max_depth:
            return

        neighbors = list(g.successors(node))[:max_neighbours]
        for n in neighbors:
            if n not in visits:
                visits[n] = 0
            visits[n] += 1

        for n in neighbors:
            step(g, n, visits, depth + 1, max_neighbours, max_depth)

    frequency_dict = dict()
    step(g, node, frequency_dict, 0, max_neighbours, max_depth)

    return frequency_dict


def predecessors_search(g, node, max_neighbours=5, max_depth=3):
    def step(g, node, visits, depth, max_neighbours, max_depth):
        if depth > max_depth:
            return

        neighbors = list(g.predecessors(node))[:max_neighbours]
        for n in neighbors:
            if n not in visits:
                visits[n] = 0
            visits[n] += 1

        for n in neighbors:
            step(g, n, visits, depth + 1, max_neighbours, max_depth)

    frequency_dict = dict()
    step(g, node, frequency_dict, 0, max_neighbours, max_depth)

    return frequency_dict

def get_raw_frequencies_dict(g):
    frequency_dicts = dict()
    for node in g.nodes():
        frequency_dicts[node] = successors_search(g, node)
    return frequency_dicts