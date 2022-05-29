from typing import List, Tuple, Dict, Union
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from torch import Tensor
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
from tqdm import tqdm


class StructureSet():
    '''
    A class for extracted structures

    Attributes:
        structures: Dict
            Dictionary of structures. Contains lists of structures for each of the topic starts
        centers: List[np.array]
            List of vectors corresponding to each topic
    '''
    def __init__(self, structures: Union[Dict, List[List[int]]], centers: List[np.ndarray]):
        self.centers = centers
        if isinstance(structures, dict):
            self.structures = structures
        else:
            self.structures = {str(i): [] for i in range(len(self.centers))}
            for struct in structures:
                start = str(struct[0])
                if struct not in [struct['struct'] for struct in self.structures[start]]:
                    self.structures[start].append({'struct': struct, 'count': 1})
                else:
                    for i in range(len(self.structures[start])):
                        if self.structures[start][i]['struct'] == struct:
                            self.structures[start][i]['count'] += 1
            if any([len(self.structures[topic]) == 0 for topic in self.structures.keys()]):
                warnings.warn('Some of the topics don`t start any structures. \
                This could couse troubles during generation process (if input belongs to one of this topics)')

    def get_structures(self):
        return self.structures

    def get_structures_list(self):
        '''
        Returns structures as a list
        '''
        structures_list = []
        for topic in self.structures.keys():
            if len(self.structures[topic]) != 0:
                structures_list.extend([struct['struct'] for struct in self.structures[topic]])
        return structures_list

    def get_structure_vectors(self, structure):
        '''
        For one structure (for example [1, 2, 1]) returns a list of corresponding vectors of topics
        '''
        return self.centers[structure]

    def filter(self, min_structure_occurence: int,
               max_per_theme_start: int, min_per_theme_start: int):
        '''
        Reduces the number of possible structures by choosing the most frequent
        '''
        structures_filtered = {str(i): [struct for struct in self.structures[str(i)]
                                        if struct['count'] >= min_structure_occurence]
                               for i in range(len(self.centers))}
        for topic in structures_filtered.keys():
            if len(structures_filtered[topic]) == 0 and self.structures[topic] != 0:
                structures_filtered[topic].extend(
                    sorted(self.structures[topic], key=lambda x: x['count'], reverse=True)[
                    :min(min_per_theme_start, len(self.structures[topic]))]
                )
            if len(structures_filtered[topic]) > max_per_theme_start:
                structures_filtered[topic] = sorted(self.structures[topic], key=lambda x: x['count'], reverse=True)[
                                             :max_per_theme_start]
        return StructureSet(structures_filtered, self.centers)



class StructureExtractor():
    '''
    Attributes:
        vectorizer_model: SentenceTransformer
            A model for text vectorization. (It is recommended to choose a fast model)
        k_means_model: MiniBatchKMeans
            Clustering model. If topics_number_range is set during __call__ the best cluster_num would be chosen
            and model would be updated.
        centers: List[np.array]
            List of vectors corresponding to each topic. Extracted from current k_means model.
    '''
    def __init__(self, vectorizer_model: SentenceTransformer = None,
                 k_means_model: MiniBatchKMeans = MiniBatchKMeans()):
        if vectorizer_model is None:
            vectorizer_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.vectorizer_model = vectorizer_model
        self.k_means_model = k_means_model
        self.centers = None

    def vectorize_data(self, texts: List[List[str]]) -> List[np.ndarray]:
        vectors = []
        for text in tqdm(texts):
            vectors.extend(self.vectorizer_model.encode(text))
        return vectors

    def choose_best_cluster_num(self, vectors: List[np.ndarray],
                                topics_number_range: Tuple[int, int, int]):
        best_score = -1
        best_cluster_num = self.k_means_model.get_params()['n_clusters']
        for cluster_num in tqdm(range(*topics_number_range)):
            cluster_model_kmeans = MiniBatchKMeans(n_clusters=cluster_num,
                                                   random_state=42)
            clusters = cluster_model_kmeans.fit_predict(vectors)
            score = silhouette_score(vectors, clusters)
            if score > best_score:
                best_score = score
                self.k_means_model = cluster_model_kmeans
                best_cluster_num = cluster_num
        return best_cluster_num

    def cluster_data(self, vectors: List[np.ndarray]):
        clusters = self.k_means_model.fit_predict(vectors)
        self.centers = self.k_means_model.cluster_centers_
        return clusters

    def extract_structures(self, texts, clusters, n):
        '''
        Extracts n-grams of the form [24, 3, 27] (here n = 3), wich can be used to get corresponding n-grams of
        vectors of the topics
        '''
        offset = 0
        structures = []
        for text in texts:
            if len(text) < n:
                offset += len(text)
                continue
            else:
                for i in range(offset, offset + len(text) - n + 1):
                    structures.append(list(clusters[i:i + n]))
                offset += len(text)
        return structures

    def __call__(self, texts: List[List[str]], n: int = 3,
                 topics_number_range: Tuple[int, int, int] = None,
                 min_structure_occurence: int = 3, max_per_theme_start: int = 20,
                 min_per_theme_start: int = 1):
        '''
        Extracts structures from the corpus of the texts and filters them.
        '''
        if n <= 0:
            raise ValueError('n must be a positive number')
        vectors = self.vectorize_data(texts)
        if topics_number_range is not None:
            self.choose_best_cluster_num(vectors, topics_number_range)
        if max_per_theme_start <= 0:
            raise ValueError('max_per_theme_start must be a positive number')
        clusters = self.cluster_data(vectors)
        all_structures = self.extract_structures(texts, clusters, n)
        all_structures = StructureSet(all_structures, self.centers)
        return all_structures.filter(min_structure_occurence, max_per_theme_start,
                                     min_per_theme_start)
