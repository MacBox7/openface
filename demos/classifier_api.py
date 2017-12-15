#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import os
import pickle
import sys

from operator import itemgetter

import numpy as np

np.set_printoptions(precision=2)
import pandas as pd

import openface.openface as openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


class Parser:
    def __init__(self, imgs, classifierModel, workDir, classifier='LinearSvm'):
        self.dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
        self.networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
        self.classifierModel = classifierModel
        self.workDir = workDir
        self.imgDim = 96
        self.cuda = False
        self.verbose = False
        self.ldaDim = -1
        self.classifier = classifier
        self.imgs = imgs
        self.align = openface.AlignDlib(self.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(self.networkModel, imgDim=self.imgDim, cuda=self.cuda)

    def getRep(self, imgPath, multiple=False):
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))

        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if multiple:
            bbs = self.align.getAllFaceBoundingBoxes(rgbImg)
        else:
            bb1 = self.align.getLargestFaceBoundingBox(rgbImg)
            bbs = [bb1]
        if len(bbs) == 0 or (not multiple and bb1 is None):
            raise Exception("Unable to find a face: {}".format(imgPath))

        reps = []
        for bb in bbs:
            alignedFace = self.align.align(
                self.imgDim,
                rgbImg,
                bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                raise Exception("Unable to align image: {}".format(imgPath))

            rep = self.net.forward(alignedFace)
            reps.append((bb.center().x, rep))
        sreps = sorted(reps, key=lambda x: x[0])
        return sreps

    def train(self):
        print("Loading embeddings.")
        fname = "{}/labels.csv".format(self.workDir)
        labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
        labels = map(itemgetter(1),
                     map(os.path.split,
                         map(os.path.dirname, labels)))  # Get the directory.
        fname = "{}/reps.csv".format(self.workDir)
        embeddings = pd.read_csv(fname, header=None).as_matrix()
        le = LabelEncoder().fit(labels)
        labelsNum = le.transform(labels)
        nClasses = len(le.classes_)
        print("Training for {} classes.".format(nClasses))

        if self.classifier == 'LinearSvm':
            clf = SVC(C=1, kernel='linear', probability=True)
        elif self.classifier == 'GridSearchSvm':
            print("""
            Warning: In our experiences, using a grid search over SVM hyper-parameters only
            gives marginally better performance than a linear SVM with C=1 and
            is not worth the extra computations of performing a grid search.
            """)
            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
        elif self.classifier == 'GMM':  # Doesn't work best
            clf = GMM(n_components=nClasses)

        # ref:
        # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
        elif self.classifier == 'RadialSvm':  # Radial Basis Function kernel
            # works better with C = 1 and gamma = 2
            clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
        elif self.classifier == 'DecisionTree':  # Doesn't work best
            clf = DecisionTreeClassifier(max_depth=20)
        elif self.classifier == 'GaussianNB':
            clf = GaussianNB()

        # ref: https://jessesw.com/Deep-Learning/
        elif self.classifier == 'DBN':
            from nolearn.dbn import DBN
            clf = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                      learn_rates=0.3,
                      # Smaller steps mean a possibly more accurate result, but the
                      # training will take longer
                      learn_rate_decays=0.9,
                      # a factor the initial learning rate will be multiplied by
                      # after each iteration of the training
                      epochs=300,  # no of iternation
                      # dropouts = 0.25, # Express the percentage of nodes that
                      # will be randomly dropped as a decimal.
                      verbose=1)

        if self.ldaDim > 0:
            clf_final = clf
            clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),
                            ('clf', clf_final)])

        clf.fit(embeddings, labelsNum)

        fName = "{}/classifier.pkl".format(self.workDir)
        print("Saving classifier to '{}'".format(fName))
        with open(fName, 'w') as f:
            pickle.dump((le, clf), f)

    def infer(self, multiple=False):
        with open(self.classifierModel, 'rb') as f:
            if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
            else:
                (le, clf) = pickle.load(f, encoding='latin1')

        for img in self.imgs:
            print("\n=== {} ===".format(img))
            reps = self.getRep(img, multiple)
            if len(reps) > 1:
                print("List of faces in image from left to right")
            for r in reps:
                rep = r[1].reshape(1, -1)
                bbx = r[0]
                predictions = clf.predict_proba(rep).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]
                return person.decode('utf-8'), confidence
