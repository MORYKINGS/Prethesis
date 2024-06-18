import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform
from tqdm import tqdm
from nltk import ngrams
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import f1_score as score
from sklearn.metrics import adjusted_rand_score
from scipy.stats import mode
import math
import time

class nTreeClus:
    def __init__(self, sequences, n=None, method="All",C=None, ntree=10, verbose=1): 
        self.labels_ = None
        self.n                                 = n   # Parameter n
        self.method                            = method
        self.ntree                             = ntree
        self.C_DT                              = C
        self.C_RF                              = C
        self.C_DT_p                            = C
        self.C_RF_p                            = C
        self.sequences                         = sequences
        self.seg_mat                           = None
        self.Dist_tree_terminal_cosine         = None # distance_DT
        self.assignment_tree_terminal_cosine   = None # labels_DT
        self.Dist_tree_terminal_cosine_p       = None # distance_DT + position
        self.assignment_tree_terminal_cosine_p = None # labels_DT   + position
        self.Dist_RF_terminal_cosine           = None # distance_RF
        self.assignment_RF_terminal_cosine     = None # labels_RF
        self.Dist_RF_terminal_cosine_p         = None # distance_RF + position
        self.assignment_RF_terminal_cosine_p   = None # labels_RF   + position
        self.verbose                           = verbose
        self.running_timeSegmentation          = None
        self.running_timeDT                    = None
        self.running_timeDT_p                  = None
        self.running_timeRF                    = None
        self.running_timeRF_p                  = None
    @staticmethod
    def _cosine_distance(matrix, batch_size=1000):
        """Calculate cosine distance matrix from sparse matrix in batches."""
        n_samples = matrix.shape[0]
        dist = np.zeros((n_samples, n_samples))
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            sim_batch = cosine_similarity(matrix[start:end], matrix)
            dist[start:end] = 1 - sim_batch
        return squareform(dist)

    def matrix_segmentation(self):
        seg_mat_list = []
        for i in tqdm(range(len(self.sequences)), desc="Matrix Segmentation (Splitting based on window size)", disable=1-self.verbose):
            sentence = self.sequences[i]
            ngrams_ = ngrams(list(sentence), self.n)
            for idx, gram in enumerate(ngrams_):
                seg_mat_list.append(list(gram + (idx,) + (i,)))
        self.seg_mat = pd.DataFrame(seg_mat_list)
        self.seg_mat.columns = np.append(np.arange(0, self.n-1), ('Class', 'Position', 'OriginalMAT_element'))

    def nTreeClus(self):
        if self.n is None:
            if self.verbose: print("Finding the parameter 'n'")
            total_avg = round(sum(map(len, self.sequences)) / len(self.sequences))  # average length of strings
            self.n = min(round(total_avg ** 0.5) + 1, min(map(len, self.sequences)) - 1)
            if self.verbose: print(f"Parameter 'n' is set to {self.n}")
        if self.n < 3:
            raise ValueError("Parameter n cannot be less than 3. Remove the sequences with length shorter than 3 and then re-run the function.")

        start_time = time.time()
        self.matrix_segmentation()
        self.running_timeSegmentation = round(time.time() - start_time)

        le = LabelEncoder()
        self.seg_mat.loc[:, 'Class'] = le.fit_transform(self.seg_mat.loc[:, 'Class'])
        self.seg_mat = pd.get_dummies(self.seg_mat).reset_index(drop=True)

        if self.method in ["All", "DT"]:
            self._process_DT()

        if self.method in ["All", "DT_position"]:
            self._process_DT_position()

        if self.method in ["All", "RF"]:
            self._process_RF()

        if self.method in ["All", "RF_position"]:
            self._process_RF_position()

    def _process_DT(self):
        start_time = time.time()
        xtrain = self.seg_mat.drop(labels=['OriginalMAT_element', 'Position', 'Class'], axis=1).copy()
        ytrain = self.seg_mat['Class'].copy()
        dtree = DecisionTreeClassifier()
        if self.verbose: print("Fit DT")
        fitted_tree = dtree.fit(X=xtrain, y=ytrain)
        terminal_tree = fitted_tree.tree_.apply(xtrain.values.astype('float32'))
        terminal_output_tree = pd.DataFrame(terminal_tree)
        terminal_output_tree['OriginalMAT_element'] = self.seg_mat['OriginalMAT_element'].values
        terminal_output_tree.columns = ['ter', 'OriginalMAT_element']
        i, r = pd.factorize(terminal_output_tree['OriginalMAT_element'])
        j, c = pd.factorize(terminal_output_tree['ter'])
        ij, tups = pd.factorize(list(zip(i, j)))
        terminal_output_tree_F = csr_matrix((np.bincount(ij), tuple(zip(*tups))))
        if self.verbose: print("Determining the cosine Distance")
        self.Dist_tree_terminal_cosine = self._cosine_distance(terminal_output_tree_F)
        if self.verbose: print("Applying Agglomerative Clustering")
        clustering = AgglomerativeClustering(n_clusters=self.ntree, affinity='precomputed', linkage='complete')
        self.labels_ = clustering.fit_predict(squareform(self.Dist_tree_terminal_cosine))
        self.running_timeDT = round(time.time() - start_time)

    def _process_DT_position(self):
        start_time = time.time()
        xtrain = self.seg_mat.drop(labels=['OriginalMAT_element', 'Class'], axis=1).copy()
        ytrain = self.seg_mat['Class'].copy()
        dtree = DecisionTreeClassifier()
        if self.verbose: print("Fit DT + POSITION")
        fitted_tree = dtree.fit(X=xtrain, y=ytrain)
        terminal_tree = fitted_tree.tree_.apply(xtrain.values.astype('float32'))
        terminal_output_tree = pd.DataFrame(terminal_tree)
        terminal_output_tree['OriginalMAT_element'] = self.seg_mat['OriginalMAT_element'].values
        terminal_output_tree.columns = ['ter', 'OriginalMAT_element']
        i, r = pd.factorize(terminal_output_tree['OriginalMAT_element'])
        j, c = pd.factorize(terminal_output_tree['ter'])
        ij, tups = pd.factorize(list(zip(i, j)))
        terminal_output_tree_F = csr_matrix((np.bincount(ij), tuple(zip(*tups))))
        if self.verbose: print("Determining the cosine Distance")
        self.Dist_tree_terminal_cosine_p = self._cosine_distance(terminal_output_tree_F)
        if self.verbose: print("Applying Agglomerative Clustering")
        clustering = AgglomerativeClustering(n_clusters=self.ntree, affinity='precomputed', linkage='complete')
        self.labels_ = clustering.fit_predict(squareform(self.Dist_tree_terminal_cosine_p))
        self.running_timeDT_p = round(time.time() - start_time)

    def _process_RF(self):
        start_time = time.time()
        xtrain = self.seg_mat.drop(labels=['OriginalMAT_element', 'Position', 'Class'], axis=1).copy()
        ytrain = self.seg_mat['Class'].copy()
        np.random.seed(123)
        forest = RandomForestClassifier(n_estimators=self.ntree, max_features=0.36)
        if self.verbose: print("Fit RF")
        fitted_forest = forest.fit(X=xtrain, y=ytrain)
        terminal_forest = fitted_forest.apply(xtrain)
        terminal_forest = pd.DataFrame(terminal_forest)
        terminal_forest = terminal_forest.astype('str')
        for col in terminal_forest:
            terminal_forest[col] = '{}_'.format(col) + terminal_forest[col]
        rbind_terminal_forest = pd.concat([self.seg_mat['OriginalMAT_element'], terminal_forest[0]], axis=1)
        for i in range(1, terminal_forest.shape[1]):
            temp = pd.concat([self.seg_mat['OriginalMAT_element'], terminal_forest[i]], axis=1)
            rbind_terminal_forest = pd.concat([rbind_terminal_forest, temp], ignore_index=True)
        rbind_terminal_forest.columns = ['OriginalMAT_element', 'ter']
        i, r = pd.factorize(rbind_terminal_forest['OriginalMAT_element'])
        j, c = pd.factorize(rbind_terminal_forest['ter'])
        ij, tups = pd.factorize(list(zip(i, j)))
        terminal_output_forest_F = csr_matrix((np.bincount(ij), tuple(zip(*tups))))
        if self.verbose: print("Determining the cosine Distance")
        self.Dist_RF_terminal_cosine = self._cosine_distance(terminal_output_forest_F)
        if self.verbose: print("Applying Agglomerative Clustering")
        clustering = AgglomerativeClustering(n_clusters=self.ntree, affinity='precomputed', linkage='complete')
        self.labels_ = clustering.fit_predict(squareform(self.Dist_RF_terminal_cosine))
        self.running_timeRF = round(time.time() - start_time)

    def _process_RF_position(self):
        start_time = time.time()
        xtrain = self.seg_mat.drop(labels=['OriginalMAT_element', 'Class'], axis=1).copy()
        ytrain = self.seg_mat['Class'].copy()
        np.random.seed(123)
        forest = RandomForestClassifier(n_estimators=self.ntree, max_features=0.36)
        if self.verbose: print("Fit RF + POSITION")
        fitted_forest = forest.fit(X=xtrain, y=ytrain)
        terminal_forest = fitted_forest.apply(xtrain)
        terminal_forest = pd.DataFrame(terminal_forest)
        terminal_forest = terminal_forest.astype('str')
        for col in terminal_forest:
            terminal_forest[col] = '{}_'.format(col) + terminal_forest[col]
        rbind_terminal_forest = pd.concat([self.seg_mat['OriginalMAT_element'], terminal_forest[0]], axis=1)
        for i in range(1, terminal_forest.shape[1]):
            temp = pd.concat([self.seg_mat['OriginalMAT_element'], terminal_forest[i]], axis=1)
            rbind_terminal_forest = pd.concat([rbind_terminal_forest, temp], ignore_index=True)
        rbind_terminal_forest.columns = ['OriginalMAT_element', 'ter']
        i, r = pd.factorize(rbind_terminal_forest['OriginalMAT_element'])
        j, c = pd.factorize(rbind_terminal_forest['ter'])
        ij, tups = pd.factorize(list(zip(i, j)))
        terminal_output_forest_F = csr_matrix((np.bincount(ij), tuple(zip(*tups))))
        if self.verbose: print("Determining the cosine Distance")
        self.Dist_RF_terminal_cosine_p = self._cosine_distance(terminal_output_forest_F)
        if self.verbose: print("Applying Agglomerative Clustering")
        clustering = AgglomerativeClustering(n_clusters=self.ntree, affinity='precomputed', linkage='complete')
        self.labels_ = clustering.fit_predict(squareform(self.Dist_RF_terminal_cosine_p))
        self.running_timeRF_p = round(time.time() - start_time)

    def output(self):
        return self.labels_

    def performance(self, Ground_Truth):
        """Report the performance using various metrics"""
        self.res = pd.DataFrame()
        if self.method in ["All", "DT"]:
            self.res.loc['DT', "F1S"] = self._calculate_f1_score(self.assignment_tree_terminal_cosine, Ground_Truth)
            self.res.loc['DT', "ARS"] = self._calculate_ars(self.assignment_tree_terminal_cosine, Ground_Truth)
            self.res.loc['DT', "RS"] = self._calculate_rs(self.assignment_tree_terminal_cosine, Ground_Truth)
            self.res.loc['DT', "Pur"] = self._calculate_purity(self.assignment_tree_terminal_cosine, Ground_Truth)
            self.res.loc['DT', "Sil"] = self._calculate_silhouette(self.Dist_tree_terminal_cosine, self.assignment_tree_terminal_cosine)
            self.res.loc['DT', "1NN"] = self._calculate_1nn(self.Dist_tree_terminal_cosine, Ground_Truth)

        if self.method in ["All", "RF"]:
            self.res.loc['RF', "F1S"] = self._calculate_f1_score(self.assignment_RF_terminal_cosine, Ground_Truth)
            self.res.loc['RF', "ARS"] = self._calculate_ars(self.assignment_RF_terminal_cosine, Ground_Truth)
            self.res.loc['RF', "RS"] = self._calculate_rs(self.assignment_RF_terminal_cosine, Ground_Truth)
            self.res.loc['RF', "Pur"] = self._calculate_purity(self.assignment_RF_terminal_cosine, Ground_Truth)
            self.res.loc['RF', "Sil"] = self._calculate_silhouette(self.Dist_RF_terminal_cosine, self.assignment_RF_terminal_cosine)
            self.res.loc['RF', "1NN"] = self._calculate_1nn(self.Dist_RF_terminal_cosine, Ground_Truth)

        if self.method in ["All", "DT_position"]:
            self.res.loc['DT_p', "F1S"] = self._calculate_f1_score(self.assignment_tree_terminal_cosine_p, Ground_Truth)
            self.res.loc['DT_p', "ARS"] = self._calculate_ars(self.assignment_tree_terminal_cosine_p, Ground_Truth)
            self.res.loc['DT_p', "RS"] = self._calculate_rs(self.assignment_tree_terminal_cosine_p, Ground_Truth)
            self.res.loc['DT_p', "Pur"] = self._calculate_purity(self.assignment_tree_terminal_cosine_p, Ground_Truth)
            self.res.loc['DT_p', "Sil"] = self._calculate_silhouette(self.Dist_tree_terminal_cosine_p, self.assignment_tree_terminal_cosine_p)
            self.res.loc['DT_p', "1NN"] = self._calculate_1nn(self.Dist_tree_terminal_cosine_p, Ground_Truth)

        if self.method in ["All", "RF_position"]:
            self.res.loc['RF_p', "F1S"] = self._calculate_f1_score(self.assignment_RF_terminal_cosine_p, Ground_Truth)
            self.res.loc['RF_p', "ARS"] = self._calculate_ars(self.assignment_RF_terminal_cosine_p, Ground_Truth)
            self.res.loc['RF_p', "RS"] = self._calculate_rs(self.assignment_RF_terminal_cosine_p, Ground_Truth)
            self.res.loc['RF_p', "Pur"] = self._calculate_purity(self.assignment_RF_terminal_cosine_p, Ground_Truth)
            self.res.loc['RF_p', "Sil"] = self._calculate_silhouette(self.Dist_RF_terminal_cosine_p, self.assignment_RF_terminal_cosine_p)
            self.res.loc['RF_p', "1NN"] = self._calculate_1nn(self.Dist_RF_terminal_cosine_p, Ground_Truth)

        return self.res

    def _calculate_f1_score(self, predictions, ground_truth):
        return max(
            score(ground_truth, predictions, average='macro', zero_division=0)[2],
            score(ground_truth, self._replace_labels_with_mode(predictions, ground_truth), average='macro', zero_division=0)[2]
        ).round(3)

    def _calculate_ars(self, predictions, ground_truth):
        return math.ceil((adjusted_rand_score(ground_truth, predictions)) * 1000) / 1000

    def _calculate_rs(self, predictions, ground_truth):
        return math.ceil((self.rand_index_score(ground_truth, predictions)) * 1000) / 1000

    def _calculate_purity(self, predictions, ground_truth):
        return math.ceil((self.purity_score(ground_truth, predictions)) * 1000) / 1000

    def _calculate_silhouette(self, distance_matrix, predictions):
        return math.ceil(silhouette_score(squareform(distance_matrix), predictions, metric='cosine').round(3) * 1000) / 1000

    def _calculate_1nn(self, distance_matrix, ground_truth):
        return math.ceil((self._1nn(ground_truth, distance_matrix)) * 1000) / 1000

    def _replace_labels_with_mode(self, predictions, ground_truth):
        replacement = {}
        for i in np.unique(predictions):
            replacement[i] = mode(ground_truth[predictions == i])[0][0]
        return np.vectorize(replacement.get)(predictions)

    def plot(self, which_model, labels, save=False, color_threshold=None, linkage_method='ward', annotate=False, xy=(0, 0), rotation=90):
        if which_model == 'RF':
            distance = self.Dist_RF_terminal_cosine
        elif which_model == 'RF_position':
            distance = self.Dist_RF_terminal_cosine_p
        elif which_model == 'DT':
            distance = self.Dist_tree_terminal_cosine
        elif which_model == 'DT_position':
            distance = self.Dist_tree_terminal_cosine_p
        else:
            raise Exception(f'Model {which_model} not supported.')

        HC_tree_terminal_cosine = linkage(distance, linkage_method)
        fig = plt.figure(figsize=(25, 10))
        ax = fig.add_subplot(1, 1, 1)
        if color_threshold is None:
            dendrogram(HC_tree_terminal_cosine, labels=labels, ax=ax)
        else:
            dendrogram(HC_tree_terminal_cosine, labels=labels, ax=ax, color_threshold=color_threshold)
        ax.tick_params(axis='x', which='major', labelsize=18, rotation=rotation)
        ax.tick_params(axis='y', which='major', labelsize=18)
        if annotate:
            ax.annotate(f"""
                        F1-score = {round(self.res.loc['DT_p', 'F1S'], 2)}
                        ARS        = {round(self.res.loc['DT_p', 'ARS'], 2)}
                        RS          = {round(self.res.loc['DT_p', 'RS'], 2)}
                        Purity     = {round(self.res.loc['DT_p', 'Pur'], 2)}
                        ASW       = {round(self.res.loc['DT_p', 'Sil'], 2)}
                        1NN       = {round(self.res.loc['DT_p', '1NN'], 2)}
                        """, xy=xy, xytext=(0, 0), fontsize=18,
                        textcoords='offset points', va='top', ha='left')
        if save:
            plt.savefig(f"dendrogram_{which_model}.png", dpi=300, bbox_inches='tight')
        return fig, ax

    def __version__(self):
        print('1.2.1')

    def updates(self):
        print("""
              - Adding Plotting option
              - Adding Executing time.
              - Adding positional version of nTreeClus 
              - Adding 1NN to the performance metrics
              - Fixing Some bugs in performance calculation
              """)

