#! /usr/bin/env python3
# coding: utf-8
import pandas as pd 
import numpy as np
import sys
# import phe.paillier as he
# import matplotlib.pyplot as plt
from math import isnan, floor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


class MLModel:

    def __init__(self, pubkey, precision_max_data):
        self.pubkey = pubkey
        self.precision_max_data = precision_max_data

    def get_training_data(self):
        # Chargement du dataset
        self.train_data = pd.read_csv('../data/training_data.csv', sep=',')
        self.train_data.set_index('id', inplace=True, drop=True)
        #print("Dimensions: {0}".format(train_data.shape))

    def set_test_data(self, test_data):
        self.test_data = test_data

    def prepare_training_data(self):
        X = self.train_data
        self.target = X.diagnosis
    
        # Suppression des features inutiles
        to_del = ['diagnosis', 'Unnamed: 32']
        for col in to_del : del X[col]

        # Scaling des données
        X = scaling(X)

        # Préparation de la matrice
        X = np.array(X, float)
        X = np.array(prepare_matrix(X, self.precision_max_data), int)
        self.weights = np.zeros(X.shape[1])
        self.train_data = X

    def encrypt_training_data(self):
    	pass

    def fit(self, n_iter, eta=0.01):
        # Régression linéaire pour n_iter
        for _ in range(n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient, eta)

    def gradient_step(self, gradient, eta=0.01):
        # Met à jour le modèle avec le gradient donné
        self.weights -= eta * gradient

    def compute_gradient(self):
        # Calcule le gradient du modèle à partir du dataset d'entraînement
        delta = self.predict(self.train_data) - self.target
        return delta.dot(self.train_data)

    def predict(self, X):
        return X.dot(self.weights)

    def encrypted_gradient(self, sum_to=None):
        # Calcule et chiffre le gradient
        gradient = self.compute_gradient()
        encrypted_gradient = encrypt_vector(self.pubkey, gradient)

        if sum_to is not None:
            return sum_encrypted_vectors(sum_to, encrypted_gradient)
        else:
            return encrypted_gradient

    def train_model(self):
        self.get_training_data()
        self.prepare_training_data()
        # self.encrypt_training_data()

        lr = LogisticRegression()
        lr.fit(self.train_data,self.target)
        print(compute_score(lr,self.train_data,self.target))
        
        self.clf = lr

    def test_predict(self):
        test_data = pd.read_csv('../data/test_data.csv', sep=',')
        test_data.set_index('id', inplace=True, drop=True)
        print(test_data)
        test_data = scaling(test_data)

        test_data = np.array(test_data, float)
        test_data = np.array(prepare_matrix(test_data, self.precision_max_data), int)

        y_pred = self.clf.predict(test_data)
        print(y_pred)

# Fonctions utiles au Machine Learning
def scaling(X):
    scaler = MinMaxScaler((0.1,0.9999))
    scaler.fit(X)
    return scaler.transform(X)

def compute_score(clf, X, y):
    xval = cross_val_score(clf, X, y, cv=5)
    return round(np.mean(xval),2)

# Fonctions utiles au chiffrement des données
def mean_square_error(y_pred, y):
    return np.mean((y - y_pred) ** 2)

def encrypt_vector(pubkey, U):
    return [pubkey.encrypt(u) for u in U]

def encrypt_matrix(pubkey, X):
    return [encrypt_vector(pubkey, U) for U in X]

def encrypted_vector_ciphertext(encrypted_U):
    return [u.ciphertext() for u in encrypted_U]

def encrypted_matrix_ciphertext(encrypted_X):
    return [encrypted_vector_ciphertext(encrypted_U) for encrypted_U in encrypted_X]

def sum_encrypted_vectors(x, y):
    if len(x) != len(y):
        raise ValueError('Encrypted vectors must have the same size')
    return [x[i] + y[i] for i in range(len(x))]
 
# Transforme un vecteur U de floats selon la configuration pour le préparer au chiffrement
def prepare_vector(U, precision_max_data):
    new_U = []
    for u in U:
        decimal_part_size = 0

        if((u-floor(u)) > 0):
            decimal_part_size = len(str(u)) - len(str(floor(u))) - 1
            new_u = str(round(u*(10**precision_max_data)) / 10**precision_max_data).replace('.','')
        else:
            new_u = str(u)

        zeros_to_add = 0

        if(decimal_part_size < precision_max_data):
            zeros_to_add = precision_max_data - decimal_part_size      

        for i in range(zeros_to_add):
            new_u += '0'
        new_U.append(int(new_u))
    return new_U

# Transforme une matrice X de floats selon la configuration pour le préparer au chiffrement
def prepare_matrix(X, precision_max_data):
    new_X = []
    for U in X:
        new_X.append(prepare_vector(U, precision_max_data))
    return new_X


if __name__ == '__main__':
    precision_max_data = 4
    ml = MLModel(2, precision_max_data)
    ml.train_model()
    ml.test_predict()

    # ATTENTION LA DATA ENVOYEE PAR LE CLIENT N'A PAS ENCORE ETE SCALEE !!!