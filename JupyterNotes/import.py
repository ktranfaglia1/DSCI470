#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:11:07 2022

@author: kgteller

Imports file
"""

import pandas as pd
import scipy as sc
import numpy as np
#import mglearn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression 

from sklearn.neighbors import KNeighborsRegressor


from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, cross_val_score


from sklearn.metrics import r2_score

from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap

from sklearn.datasets import load_breast_cancer

import os
os.environ["PATH"] += os.pathsep + '/opt/homebrew/Cellar/graphviz/6.0.1/bin/'

#from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris