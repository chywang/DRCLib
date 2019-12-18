import copy
import logging
import sys
import warnings

import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


class RelationLearner:
    # projection settings
    _project_dict = dict()
    # default feature settings
    __applied_defaults = set()

    # relation set
    __rel_set = set()
    __relation_map = dict()
    _reverse_relation_map = dict()

    # word embedding model
    __embed_model = None
    _n_embeddings = 100

    # data
    _training_set = list()
    _testing_set = list()

    # models
    _proj_models = dict()

    # constants, do not modify
    __project_methods = ['linear', 'orth_linear', 'piecewise', 'orth_piecewise']
    __default_methods = ['subject', 'object', 'offset', 'addition', 'product']

    # loading function
    def load_word_embeddings(self, model_path):
        warnings.filterwarnings("ignore")
        self.__embed_model = KeyedVectors.load_word2vec_format(model_path)
        self._n_embeddings = self.__embed_model.vector_size

    # loading function
    def load_training_set(self, training_set_path):
        file = open(training_set_path)
        while 1:
            line = file.readline()
            if not line:
                break
            line = line.replace('\n', '')
            str = line.split('\t')
            subject = str[0]
            if subject not in self.__embed_model:
                logging.error(subject + 'does not exist in the demo!')
                continue
            object = str[1]
            if object not in self.__embed_model:
                logging.error(object + 'does not exist in the demo!')
                continue
            rel_name = str[2]
            self.__rel_set.add(rel_name)
            self._training_set.append((subject, object, rel_name))
        file.close()
        self.generate_label_map()

    def generate_label_map(self):
        i = 0
        for rel_name in self.__rel_set:
            self.__relation_map[rel_name] = i
            self._reverse_relation_map[i] = rel_name
            i = i + 1

    # loading function
    def load_testing_set(self, testing_set_path):
        file = open(testing_set_path)
        while 1:
            line = file.readline()
            if not line:
                break
            line = line.replace('\n', '')
            str = line.split('\t')
            subject = str[0]
            if subject not in self.__embed_model:
                logging.error(subject + 'does not exist in the demo!')
                continue
            object = str[1]
            if object not in self.__embed_model:
                logging.error(object + 'does not exist in the demo!')
                continue
            rel_name = str[2]
            if rel_name not in self.__rel_set:
                logging.error(rel_name + 'does not exist in the training set!')
                continue
            self._testing_set.append((subject, object, rel_name))
        file.close()

    # setting function
    def impose_project(self, rel_name, project_method, k=1):
        if rel_name not in self.__rel_set:
            logging.error(rel_name + 'does not exist!')
            return
        if project_method not in self.__project_methods:
            logging.error(project_method + 'does not exist!')
            return
        if project_method == 'linear' or project_method == 'orth_linear':
            self._project_dict[rel_name] = (project_method, 1)
        else:
            k_int = int(k)
            if k_int <= 1:
                logging.error('projection parameter invalid!')
                return
            self._project_dict[rel_name] = (project_method, k)

    # setting function
    def set_default_feature_types(self, *feature_types):
        for feature_name in feature_types:
            if feature_name not in self.__default_methods:
                logging.error('default feature types invalid!')
            else:
                self.__applied_defaults.add(feature_name)

    def generate_sub_training_set(self, relation_name):
        sub_dataset = set()
        for first, second, relation in self._training_set:
            if relation == relation_name:
                sub_dataset.add((first, second))
        return sub_dataset

    def generate_matrix(self, sub_dataset):
        X_matrix = np.zeros(shape=(len(sub_dataset), self._n_embeddings))
        Z_matrix = np.zeros(shape=(len(sub_dataset), self._n_embeddings))
        i = 0
        for first, second in sub_dataset:
            X_matrix[i] = self.__embed_model[first]
            Z_matrix[i] = self.__embed_model[second]
            i = i + 1
        return X_matrix, Z_matrix

    def learn_linear_proj(self, input_matrix, output_matrix):
        X = input_matrix
        Z = output_matrix
        E = np.eye(self._n_embeddings, self._n_embeddings)
        lam = 0.001
        XXTr = np.linalg.inv(np.matmul(X.T, X) + lam * E)
        R = np.matmul(np.matmul(XXTr, X.T), Z)
        return R

    def learn_orth_linear_proj(self, input_matrix, output_matrix):
        B = np.zeros(shape=(self._n_embeddings, self._n_embeddings))
        for i in range(0, len(input_matrix)):
            a = input_matrix[i].reshape(self._n_embeddings, 1)
            b = output_matrix[i].reshape(1, self._n_embeddings)
            B = B + np.matmul(a, b)
        U, Sigma, Vt = np.linalg.svd(B)
        M = np.eye(self._n_embeddings, self._n_embeddings)
        M[self._n_embeddings - 1, self._n_embeddings - 1] = np.linalg.det(U) * np.linalg.det(Vt.T)
        R = np.matmul(np.matmul(U, M), Vt)
        return R

    def learn_piecewise_proj(self, input_matrix, output_matrix, k):
        Rs = list()
        X = input_matrix
        Z = output_matrix
        lam = 0.001
        kmeans = KMeans(n_clusters=k).fit(X - Z)
        for i in range(0, k):
            X_i = X[np.where(kmeans.labels_ == i)]
            Z_i = Z[np.where(kmeans.labels_ == i)]
            E = np.eye(self._n_embeddings, self._n_embeddings)
            XXTr = np.linalg.inv(np.matmul(X_i.T, X_i) + lam * E)
            R_i = np.matmul(np.matmul(XXTr, X_i.T), Z_i)
            Rs.append(R_i)
        return Rs

    def learn_multi_orth_proj(self, input_matrix, output_matrix, k):
        eta = 0.01
        diff = 0.01
        max_iter = 5
        # create weight matrix
        weight_matrix = np.random.rand(len(input_matrix), k)
        weight_matrix = weight_matrix / weight_matrix.sum(axis=0)

        matrices = list()
        for i in range(0, k):
            matrices.append(np.zeros(shape=(self._n_embeddings, self._n_embeddings)))
        for k in range(0, max_iter):
            # learn matrices
            old_matrices = copy.deepcopy(matrices)
            for i in range(0, k):
                R = self.learn_single_orth_proj(input_matrix, output_matrix, weight_matrix, i)
                matrices[i] = R
            # update weights
            for m in range(0, len(input_matrix)):
                for n in range(0, k):
                    weight_matrix[m][n] = weight_matrix[m][n] - eta * (np.linalg.norm(
                        np.matmul(matrices[n], output_matrix[m]) - input_matrix[m], ord=2)) ** 2
            # normalization
            weight_matrix = weight_matrix / weight_matrix.sum(axis=0)
            if not self.significantly_diff(old_matrices, matrices, diff):
                break
        return matrices

    def learn_single_orth_proj(self, input_matrix, output_matrix, weight_matrix, cluster_index):
        B = np.zeros(shape=(self._n_embeddings, self._n_embeddings))
        for i in range(0, len(input_matrix)):
            temp_weights = np.full((self._n_embeddings, self._n_embeddings), weight_matrix[i, cluster_index])
            a = input_matrix[i].reshape(self._n_embeddings, 1)
            b = output_matrix[i].reshape(1, self._n_embeddings)
            B = B + np.multiply(temp_weights, np.matmul(a, b))
        U, Sigma, Vt = np.linalg.svd(B)
        M = np.eye(self._n_embeddings, self._n_embeddings)
        M[self._n_embeddings - 1, self._n_embeddings - 1] = np.linalg.det(U) * np.linalg.det(Vt.T)
        R = np.matmul(np.matmul(U, M), Vt)
        return R

    def significantly_diff(self, old_matrices, new_matrices, thres):
        score = 0
        for i in range(0, len(old_matrices)):
            score = score + np.linalg.norm(old_matrices[i] - new_matrices[i], ord=2)
        return score > thres

    def generate_offset_vector_single(self, first, second, matrix):
        outcome = np.matmul(matrix, self.__embed_model[first]) - self.__embed_model[second]
        return outcome

    def generate_offset_vector_multi(self, first, second, Rs):
        min_value = sys.maxsize
        for R in Rs:
            temp_outcome = np.matmul(R, self.__embed_model[first]) - self.__embed_model[second]
            temp_value = np.linalg.norm(temp_outcome, ord=2)
            if min_value > temp_value:
                min_value = temp_value
                min_outcome = temp_outcome
        return min_outcome

    # main procedure function
    def train_all_project_models(self):
        for rel_name in self._project_dict:
            project_method, k = self._project_dict[rel_name]
            sub_train = self.generate_sub_training_set(rel_name)
            X, Z = self.generate_matrix(sub_train)
            if project_method == 'linear':
                R = self.learn_linear_proj(X, Z)
                self._proj_models[rel_name] = R
            elif project_method == 'orth_linear':
                R = self.learn_orth_linear_proj(X, Z)
                self._proj_models[rel_name] = R
            elif project_method == 'piecewise':
                Rs = self.learn_piecewise_proj(X, Z, k)
                self._proj_models[rel_name] = Rs
            else:
                Rs = self.learn_multi_orth_proj(X, Z, k)
                self._proj_models[rel_name] = Rs

    # main procedure function
    def generate_feature_vector(self, dataset):
        vec_num = len(self._project_dict) + len(self.__applied_defaults)
        F_matrix = np.zeros(shape=(len(dataset), self._n_embeddings * vec_num))
        Y_matrix = np.zeros(shape=(len(dataset), 1))

        data_count = 0
        for first, second, relation in dataset:
            feature_vector = np.zeros(shape=(1, self._n_embeddings * vec_num))
            current_index = 0
            # for projections
            for i in range(len(self.__rel_set)):
                current_rel_name = self._reverse_relation_map[i]
                if current_rel_name in self._project_dict:
                    project_method, _ = self._project_dict[current_rel_name]
                    temp_vec = np.zeros(shape=(1, self._n_embeddings))
                    if project_method == 'linear' or project_method == 'orth_linear':
                        temp_vec = self.generate_offset_vector_single(first, second,
                                                                      self._proj_models[current_rel_name])
                    else:
                        temp_vec = self.generate_offset_vector_multi(first, second,
                                                                     self._proj_models[current_rel_name])

                    feature_vector[0,
                    current_index * self._n_embeddings:(current_index + 1) * self._n_embeddings] = temp_vec
            # for others
            if 'subject' in self.__applied_defaults:
                feature_vector[0,
                current_index * self._n_embeddings:(current_index + 1) * self._n_embeddings] = self.__embed_model[
                    first]
                current_index = current_index + 1
            if 'object' in self.__applied_defaults:
                feature_vector[0,
                current_index * self._n_embeddings:(current_index + 1) * self._n_embeddings] = self.__embed_model[
                    second]
                current_index = current_index + 1
            if 'offset' in self.__applied_defaults:
                feature_vector[0,
                current_index * self._n_embeddings:(current_index + 1) * self._n_embeddings] = self.__embed_model[
                                                                                                     first] - \
                                                                                               self.__embed_model[
                                                                                                     second]
                current_index = current_index + 1
            if 'addition' in self.__applied_defaults:
                feature_vector[0,
                current_index * self._n_embeddings:(current_index + 1) * self._n_embeddings] = self.__embed_model[
                                                                                                     first] + \
                                                                                               self.__embed_model[
                                                                                                     second]
                current_index = current_index + 1
            if 'product' in self.__applied_defaults:
                feature_vector[0,
                current_index * self._n_embeddings:(current_index + 1) * self._n_embeddings] = np.multiply(
                    self.__embed_model[
                        first], self.__embed_model[second])
                current_index = current_index + 1
            F_matrix[data_count, :] = feature_vector
            relation_index = self.__relation_map[relation]
            Y_matrix[data_count, :] = relation_index
            data_count = data_count + 1
        return F_matrix, Y_matrix

    # general function
    def train_model(self):
        self.train_all_project_models()
        F_train, Y_train = self.generate_feature_vector(self._training_set)
        cls = MLPClassifier(solver='adam', hidden_layer_sizes=(self._n_embeddings,), early_stopping=True)
        cls.fit(F_train, Y_train)
        return cls

    def test_model(self, cls):
        F_test, Y_test = self.generate_feature_vector(self._testing_set)
        Y_predict = cls.predict(F_test)
        print(classification_report(Y_test, Y_predict, digits=3))
