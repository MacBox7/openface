#!/usr/bin/env python2
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

import argparse
import cv2
import numpy as np
import os
import random
import shutil

import openface.openface as openface
from openface.openface.data import iterImgs

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


class Parser:
    def __init__(self, inputDir, outputDir):
        self.dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.size = 96
        self.skipMulti = True
        self.verbose = False
        self.landmarks = 'outerEyesAndNose'
        self.fallbackLfw = None

    def write(self, vals, fName):
        if os.path.isfile(fName):
            print("{} exists. Backing up.".format(fName))
            os.rename(fName, "{}.bak".format(fName))
        with open(fName, 'w') as f:
            for p in vals:
                f.write(",".join(str(x) for x in p))
                f.write("\n")

    def alignMain(self):
        openface.helper.mkdirP(self.outputDir)

        imgs = list(iterImgs(self.inputDir))

        # Shuffle so multiple versions can be run at once.
        random.shuffle(imgs)

        landmarkMap = {
            'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
            'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
        if self.landmarks not in landmarkMap:
            raise Exception("Landmarks unrecognized: {}".format(self.landmarks))

        landmarkIndices = landmarkMap[self.landmarks]

        align = openface.AlignDlib(self.dlibFacePredictor)

        nFallbacks = 0
        for imgObject in imgs:
            print("=== {} ===".format(imgObject.path))
            outDir = os.path.join(self.outputDir, imgObject.cls)
            openface.helper.mkdirP(outDir)
            outputPrefix = os.path.join(outDir, imgObject.name)
            imgName = outputPrefix + ".png"

            if os.path.isfile(imgName):
                if self.verbose:
                    print("  + Already found, skipping.")
            else:
                rgb = imgObject.getRGB()
                if rgb is None:
                    if self.verbose:
                        print("  + Unable to load.")
                    outRgb = None
                else:
                    outRgb = align.align(self.size, rgb,
                                         landmarkIndices=landmarkIndices,
                                         skipMulti=self.skipMulti)
                    if outRgb is None and self.verbose:
                        print("  + Unable to align.")

                if self.fallbackLfw and outRgb is None:
                    nFallbacks += 1
                    deepFunneled = "{}/{}.jpg".format(os.path.join(self.fallbackLfw,
                                                                   imgObject.cls),
                                                      imgObject.name)
                    shutil.copy(deepFunneled, "{}/{}.jpg".format(os.path.join(self.outputDir,
                                                                              imgObject.cls),
                                                                 imgObject.name))

                if outRgb is not None:
                    if self.verbose:
                        print("  + Writing aligned file to disk.")
                    outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(imgName, outBgr)

        if self.fallbackLfw:
            print('nFallbacks:', nFallbacks)
