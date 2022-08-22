#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
% Portrait toolbox - script 1
Here we perform the import and preprocessing of data previously
acquired in Virutal Reality (VR) environment using XREcho Unity plugin
(behavior, biosignals) and from other means (demographics, psychometrics)

-- History
 2022-05-01
 jonas.chatel.goldman (at) gmail.com

-- Prerequisites:
-
% Experiment information
 Data was acquired on 46 subjects.
 Experimental protocol is described in paper "Influence of Game Mechanics
  and Personality Traits on Flow and Engagement in Virtual Reality"

*** Description import dataset
  profiling/subjective data:
         demographics
         user experience in VR
  behavioral data:
        spatial movements
        interactions with objects
        eyetreacking
  physiological activity:
         heart pulse using PPG
         micro-sudation using GSR

"""

#%% import libs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob2
import os
import importlib
from data_io import *

print(os.getcwd())


#%% set various parameters -> THIS MUST BE CHANGED BY EACH USER
params = {}
params["dataPath"] = '.'
params["exportPath"] = '/output-analysis'
params["profilingFile"] = 'traits_dfs_df.csv'


#%%  0.Import data

# --- import profiling/subjective data from pre-arranged .csv file
profiling_file = os.path.join(params["dataPath"],params["profilingFile"])
profiling_df = importProfilingData(profiling_file)


# --- import behavioral data
# first remove a few bad subjects
params["subject_id_list"] =  profiling_df.index.to_frame()['subject_id'].unique()
remove_ids = [14,49,57] # these three subjects lack some datasets:
#   - S14 has no ObjectFormat file in scene/scenario:  Goal/HouseObjectives
#   - S49 has no ObjectFormat file in scene/scenario:  Aesthetic/HouseAestheticSpace
#   - S57 has no eyetracking file in scene/scenario:  Aesthetic/postApo
params["subject_id_list"] = list(filter(lambda x: x not in remove_ids, params["subject_id_list"])) # remove these sub_ids from list

# create metafile_df with all metadata structure (subject_id, scenario, scene and all file paths)
basePath = params["dataPath"] # TODO: remove this later
subject_id_list = params["subject_id_list"] # TODO: remove this later
metafile_df = createMetaFile(basePath, subject_id_list, profiling_df)

del remove_ids, basePath, subject_id_list, profiling_file # clean workspace

#%% 1. Analyse data
object_id = 0 # this corresponds to camera (i.e., user's moves)
sub_sel = 'all'
scenario_sel = 'Narrative'
scene_sel = 'HouseNarrative'
objectsData_merge_df = load_thisObjectData(metafile_df, object_id, sub_sel, scenario='all', scene='all')
objectsData_merge_df.info(memory_usage='deep')
