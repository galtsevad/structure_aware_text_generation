from sentence_transformers import SentencesDataset
import torch
from typing import Dict, List
import seaborn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statistics
from scipy.stats import ks_2samp


class Evaluator():
    '''
    A class to represent an evaluator

    Attributes:
        vectorizer_model: A model for text vectorization. For example SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
            (The same one that was used during the generation)
        cos:
            Measures cosine similarity
    '''
    def __init__(self, vectorizer_model: SentencesDataset):
        self.vectorizer_model = vectorizer_model
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def get_cos_dist_between_parts(self, base_corpus: List[List[str]],
                                   generated_text_sets: Dict[str, List[List[str]]):
        '''
        Calculates cosine distances between parts of each text

        Parameters:
            base_corpus: List[List[str]]
                Corpus for comparison with generated texts
            generated_text_sets: Dict[str, List[List[List[str]]]])
                Dictionary of name: generated_text_set
        '''
        sets_names = ['base_corpus']
        sets_names.extend([key for key in generated_text_sets.keys()])
        all_text_sets = generated_text_sets.copy()
        all_text_sets['base_corpus'] = base_corpus
        all_results = {name: {} for name in sets_names}
        for name in sets_names:
            for text in all_text_sets[name]:
                vectors = self.vectorizer_model.encode(text, convert_to_tensor=True)
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        if str(i + 1) + '-' + str(j + 1) not in all_results[name].keys():
                            all_results[name][str(i + 1) + '-' + str(j + 1)] = []
                        all_results[name][str(i + 1) + '-' + str(j + 1)].append(
                            1 - self.cos(vectors[i], vectors[j]).item())
        return all_results

    def draw_boxplots(self, all_results: Dict):
        '''
        Draws boxplots of distances between different parts of texts

        Parameters:
            all_results: Dict
                Result of get_cos_dist_between_parts method
        '''
        subplots_num = len(all_results['base_corpus'])
        distance_types = [key for key in all_results['base_corpus'].keys()]
        names = [key for key in all_results.keys()]
        fig, axes = plt.subplots(1, subplots_num, figsize=(6 * subplots_num, 5), sharey=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        for i in range(subplots_num):
            axes[i].set_title(distance_types[i])
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha='right')
        for i in range(subplots_num):
            data = [all_results[method][distance_types[i]] for method in names]
            seaborn.boxplot(data=pd.DataFrame(
                data=pd.DataFrame(list(zip(*[all_results[method][distance_types[i]] for method in names])),
                                  columns=names)), ax=axes[i], palette="pastel")

    def measure_mean_std(self, all_results: Dict):
        '''
        Calculates mean and std for each set of distances

        Parameters:
            all_results: Dict
                Result of get_cos_dist_between_parts method
        '''
        calculated_results = []
        for set_name in all_results:
            calculated_results.append({'name': set_name})
            for dist in all_results[set_name]:
                calculated_results[-1][dist + '_mean'] = statistics.mean(all_results[set_name][dist])
                calculated_results[-1][dist + '_std'] = statistics.stdev(all_results[set_name][dist])
        return pd.DataFrame(calculated_results)

    def measure_ks(self, all_results: Dict):
        '''
        Performs ks test on each of the generated set of texts with the base corpus

        Parameters:
            all_results: Dict
                Result of get_cos_dist_between_parts method
        '''
        calculated_results = []
        for set_name in all_results:
            if set_name != 'base_corpus':
                calculated_results.append({'name': set_name})
                for dist in all_results[set_name]:
                    res = ks_2samp(all_results['base_corpus'][dist], all_results[set_name][dist])
                    calculated_results[-1][dist + '_ks_statistic'] = res.statistic
                    calculated_results[-1][dist + '_ks_p_value'] = res.pvalue
        return pd.DataFrame(calculated_results)
