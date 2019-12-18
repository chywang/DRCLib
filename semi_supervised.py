import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from supervised import RelationLearner


class SemiSupervisedRelationLearner(RelationLearner):
    # parameters
    __iteration_number = 5
    __conf_threshold = 0.8

    __expanded_training_set = set()
    __remaining_testing_set = set()

    def __init__(self, iteration_number, conf_threshold):
        super().__init__()
        self.__iteration_number = int(iteration_number)
        if int(iteration_number) <= 1:
            logging.error('iteration number invalid!')
            self.__iteration_number = 1
        self.__conf_threshold = float(conf_threshold)
        if float(conf_threshold) < 0 or float(conf_threshold) > 1:
            logging.error('confidence threshold invalid!')
            self.__iteration_number = 0.8

    def load_training_set(self, training_set_path):
        RelationLearner.load_training_set(self, training_set_path=training_set_path)
        self.__expanded_training_set = self._training_set.copy()

    def load_testing_set(self, testing_set_path):
        RelationLearner.load_testing_set(self, testing_set_path=testing_set_path)
        self.__remaining_testing_set = self._testing_set.copy()

    def generate_sub_training_set(self, relation_name):
        sub_dataset = set()
        for first, second, relation in self.__expanded_training_set:
            if relation == relation_name:
                sub_dataset.add((first, second))
        return sub_dataset

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

    def train_model_with_iteration(self):
        for iter in range(0, self.__iteration_number):
            self.train_all_project_models()
            F_train, Y_train = self.generate_feature_vector(self.__expanded_training_set)
            lr = LogisticRegression()
            lr.fit(F_train, Y_train)
            F_test, Y_test = self.generate_feature_vector(self.__remaining_testing_set)
            temp_testing_set = list()
            Y_predict = lr.predict_proba(F_test)
            for i in range(len(self.__remaining_testing_set)):
                (first, second, label) = self.__remaining_testing_set[i]
                proba = Y_predict[i]
                if max(proba) > self.__conf_threshold:
                    index = np.where(proba == max(proba))
                    index = int(index[0])
                    predicted_label = self._reverse_relation_map[index]
                    self.__expanded_training_set.append((first, second, predicted_label))
                else:
                    temp_testing_set.append((first, second, label))
            self.__remaining_testing_set = temp_testing_set
            if len(self.__remaining_testing_set) == 0:
                break
        F_train, Y_train = self.generate_feature_vector(self.__expanded_training_set)
        cls = MLPClassifier(solver='adam', hidden_layer_sizes=(self._n_embeddings,), early_stopping=True)
        cls.fit(F_train, Y_train)
        return cls
