#!/usr/bin/python
#coding:utf-8

# packages 

import os
import sys
import copy
import datetime
import numpy as np
import pandas as pd
import time 
import csv
from tqdm import tqdm
from multiprocessing import Process, Lock
import itertools
import IPython
from IPython.display import display, HTML
import random
from os import path
from cycler import cycler
import pyodbc
import math
from random import choice

# import date-related
from calendar import month_abbr
from datetime import date,datetime,timedelta
import datetime as datetime
import dateutil

# matplotlib imports
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import matplotlib as mpl 
import matplotlib.dates
from matplotlib import colors as mcolors
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
register_matplotlib_converters()

# import pandas related
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from pandas.tseries.offsets import Day, MonthEnd

from mpl_toolkits.mplot3d import Axes3D
import random

# scipy imports
import scipy.stats as ss
from scipy.stats import beta,chi2,t,norm
import scipy.linalg
from scipy.linalg import toeplitz
import scipy.optimize
from scipy.optimize import minimize
import scipy.signal
import scipy as sp 
import scipy.special
import scipy.stats
from scipy import stats

from sys import version_info

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import centroid, fcluster
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors.nearest_centroid import NearestCentroid

def set_stuff():
	mpl.rcParams['lines.linewidth'] = 1.5
	mpl.rcParams['lines.color'] = 'blue'
	mpl.rcParams['axes.prop_cycle'] = cycler('color',['#30a2da',"#e5ae38","#fc4f30","#6d904f","#8b8b8b"])
	mpl.rcParams['legend.fancybox'] = True
	mpl.rcParams['legend.fontsize'] = 14

	np.set_printoptions(precision=3)
	pd.set_option('precision',3)
	pd.set_option('display.float_format',lambda x: '%.3f' % x)

	plt.rc('text',usetex=False)
	plt.style.use('ggplot')

	fsize = (10,7.5)
	tsize = 18
	lsize = 16
	csize = 14

	grid = True
	display(HTML("<style>.container {width: 100% ! important;} </style>"))

import warnings 
warnings.filterwarnings('ignore')

