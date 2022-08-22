#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% XREcho Python Analysis Toolbox - tutorial script

This script shows the API functions that allow to :
- create a new experiment from scratch (design and config files)
- prepare data collection (participant passation)
- analyzing experiment data (import, verify and process, select and analyze)

% Experiment information  
Here we perform the import and preprocessing of data previously
acquired in Virtual Reality (VR) environment using XREcho Unity plugin
(behavior) and from other means (demographics, psychometrics).
"Sophie's Data" was acquired in 2021 on 46 subjects.
Experimental protocol is described in paper "Influence of Game Mechanics
  and Personality Traits on Flow and Engagement in Virtual Reality"

% History
2022-05-01
jonas.chatel.goldman (at) gmail.com
 
"""

#%% 0. import libs 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pathlib
import yaml
import importlib
from data_io import *
from data_proc import *
from questionnaire import *


#%% 1. prepare data collection for a new experiment

# --- setting local paths
params = {}
params["configDir"] = pathlib.Path('./config') # put a path to your config file here
params["configExperiment"] = params["configDir"].joinpath('design_Sophie.yaml')
params["configQuest"] = params["configDir"].joinpath('pyquest_FPT.yaml')
params["dataTest"] = pathlib.Path('Enregistrements') # root of your data (empty for testing)


# --- let's load some test configuration files defining experiment design 
config_design = loadConfigExperiment( params["configExperiment"] )       
config_questionnaire = loadConfigQuestionnaire(params["configQuest"] )  
# -> these are two internal utils function, no real use outside 
# (better to change config files manually using any available text editor)

# ---  generate everything required by Unity application to run the experiment
expe_metafile_df, subject_sequence_df_dict = newExperienceConfig(params["dataTest"], params["configExperiment"], writeFiles = True)
# now have a look to params["dataTest"] directory ! -> all participant directories and files were created
# expe_metafile_df: Pandas dataframe with all directory paths for all participants on this local machine
# subject_sequence_df_dict: dict of Pandas dataframes, each containing sequencing information for one participant


#%% 2. import data

# -- create "metafile_df", summing up all metadata structure (subject_id, condition, scene and all file paths)
params["root"] = pathlib.Path('.')
params["dataPath"] = params["root"].joinpath('Enregistrements')
metafile_df = createMetaFile(params["dataPath"], params["configExperiment"], isDataSophie=True)
# Here we rely on Sophie's data described above.
# Since it was not recorded with actual data format, some specific data wrangling must be done on this dataset (-> special parameter "isDataSophie").

# -- import profiling/subjective/questionnaire data from pre-arranged .csv file

params["profilingFile_Sophie"] = params["dataPath"].joinpath('traits_dfs_df.csv') # loading Sophie's prearranged dataset (v0)
profiling_df, metafile_df= importProfilingDataSophie(metafile_df, params["profilingFile_Sophie"]) # Sophie's pre-arrangement (v0)
params["profilingFile"] = params["dataPath"].joinpath('profiling.csv') # loading Jonas's prearranged dataset (v1)
#profiling_df = pd.read_csv(params["profilingFile"], header=[0, 1], encoding="iso8859_15")    # Jonas's pre-arrangement (v1)
# -- or create it de novo !
#profiling_df, metafile_df = importProfilingData(metafile_df, params["configExperiment"]) # generic import for ProfilingData (nothing specific to Sophie's data)
#profiling_df.to_csv(path_or_buf=params["profilingFile"], index= False, encoding="iso8859_15") # write it to disk for future use


#%% 3. select and analyse demographics data

# --- access to profiling data using multi-indexed columns
# access to profiling row by subject_id and condition (scenario)
sub_sel = 56
condition_sel = 'Aesthetic'    
profiling_mask = (profiling_df.loc[:,('experimental', 'subject_id')] == sub_sel) & (profiling_df.loc[:,('experimental', 'condition')] == condition_sel)  # select this subject / condition 
# /!\ using .loc[] function should be preffered over using chained indexing 'profiling_df.experimental.subject_id' (see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-view-versus-copy)
profiling_row = profiling_df.loc[profiling_mask].squeeze() # get single row in profiling_df and squeeze to a pd Series 
# access to single value (e.g., age of a single participant)
some_age = profiling_df.loc[profiling_mask,('demographics','age')].item()  # get single value using full mask...
index_sel = np.where(profiling_mask)[0][0] # this gets index of the first True value in mask / same as: profiling_mask[profiling_mask].index.values[0]
some_age = profiling_df.at[index_sel,('demographics','age')]  # ... or get single value using 'at' fct and single index

# access to column for all participants
sel_df = profiling_df.loc[:,('demographics', 'gender')] # returns a pd Series
sel_df = profiling_df.xs('gender', axis='columns', level='profiling_items') # another way using 'xs', returns a pd DataFrame
# -> this accesses entire column (demographics (1st level) / gender (2nd level))
# -> /!\ both methods should be prefered over using chained indexing 'profiling_df['demographics']['gender']' (see Pandas user guide: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-view-versus-copy)
sel_df = profiling_df.loc[:, [('demographics', 'gender'), ('experimental', 'passation_site')]] # access to x2 entire columns

# get all item names (column names level 1)
item_names = list(profiling_df.columns.levels[1]) # this access level 1 (items) in this 2-level multiindex (level 0 is category)

# --- filter data based on specific conditions
# filter indexes using a mask
data_sel = profiling_df.loc[:, ('demographics', 'gender')]  # mask on categorical data (here gender)
mask = (data_sel == "Femme") | (data_sel == "Autre")  # select 'Femme' AND 'Autre'
sel_df = profiling_df.loc[mask,:] 

# another way: filter indexes using powerful Panda's 'query' evaluation
col_save = profiling_df.columns # unfortunately 'query' needs single-level columns, so we save it before dropping one level
data_sel = profiling_df.droplevel(axis='columns', level='profiling_categories') # dropping first column level 
sel_df = data_sel.query('gender == ["Femme", "Autre"]') # calling query() on single column-lvel data    /!\ beware of ' and " !
sel_df.columns = col_save # re-integrate two-level column multi-indexing

# query() function allows for complex data selection with multiple conditions
sel_keys_1 = 'gender'
sel_values_1 =["Femme", "Autre"]
sel_df = data_sel.query(f'{sel_keys_1} in @sel_values_1') # using a list of values in a python variable
sel_keys_2 = 'passation_site'
sel_values_2 ="INSA Lyon"
sel_df = data_sel.query(f'{sel_keys_1} in @sel_values_1 and {sel_keys_2} in @sel_values_2 and age > 22') # x3 conditions here !


# --- show some quick data summary
# data summary
profiling_describe_df = profiling_df.describe(include="all") 

# number of participants
nb_subjects = profiling_df.loc[:,('experimental', 'subject_id')].nunique()

# histogram of age repartition
sel_df = profiling_df.loc[:, ('demographics', 'age')] 
plot = sns.histplot(sel_df, kde=True)
plt.show()

# pivot 1: difference entre les scores d'engagement des hommes et des femmes par classe d'age et pour chaque condition
data_sel = profiling_df.droplevel(axis='columns', level='profiling_categories') # dropping first column level 
sel_df = data_sel.query('gender in ["Femme", "Homme"]') # select data based on some conditions
age_cut = pd.cut(sel_df.loc[:,'age'], [0, 20, 21, 100]) # bin age values into 3 discrete non-uniform intervals 
pivot_profiling_df_1 = sel_df.pivot_table(values='engagement', index=['gender', age_cut], columns='condition')

# pivot 2: nombre de participants pour chaque site par genre + par classe d'age
data_sel = profiling_df.droplevel(axis='columns', level='profiling_categories') # dropping first column level 
sel_df = data_sel.query('gender in ["Femme", "Homme"]') # select data based on some conditions
age_cut = pd.cut(sel_df.loc[:,'age'], [0, 20, 100]) # bin age values into 3 discrete non-uniform intervals 
pivot_profiling_df_2 = sel_df.pivot_table(values='profiling_index',index=['gender', age_cut], columns='passation_site', aggfunc='count') # here using count as aggregate function -- here value hos no importance (simple element count)

# pivot 3: répartition (%) de niveaux de VR_mastery par genre
data_sel = profiling_df.droplevel(axis='columns', level='profiling_categories') # dropping first column level 
sel_df = data_sel.query('gender in ["Femme", "Homme"]') # select data based on some conditions
pivot_profiling_df_3 = sel_df.pivot_table(values='profiling_index',index='vr_mastery', columns='gender', aggfunc=lambda x: 100*len(x)/len(sel_df)) # custom aggregate function, here gives total percent (among men+women) -- here value hos no importance (simple element count)

# --- export profiling_df to a .csv file
csv_path = 'profiling_Sophie.csv' # give it a path
profiling_df.to_csv(path_or_buf=csv_path, index= False, encoding="iso8859_15")  # save it to disk
# read it: 
profiling_df_read = pd.read_csv(csv_path, header=[0,1], encoding="iso8859_15") 




#%% 4. select and analyse behavioral data and action / event data

# --- load all data for one specific subject / condition / scene
sub_sel = 7
condition_sel = 'Narrative'  
scene_sel = 'HouseNarrative'  
metafile_mask = (metafile_df.subject_id == sub_sel) & (metafile_df.condition == condition_sel) & (metafile_df.scene == scene_sel) # select this subject / condition / scene
metafile_sel = metafile_df.loc[metafile_mask].squeeze() # get single row in metafile_df and convert to a pd Series 
objectsData_df, objectsFormat_df = importDataFile(metafile_sel.file_objectsData, metafile_sel.file_objectsFormat, data_type='objects')
actionsData_df, actionsFormat_df = importDataFile(metafile_sel.file_actionsData, metafile_sel.file_actionsFormat, data_type='actions')
eyetracking_df = importDataFile(metafile_sel.file_HTCViveProEyeData, [], data_type='eyetracking')
# --- quick check memory usage for one dataframe
eyetracking_df.info(memory_usage='deep')
# --- quick bar plot with all tracked data (objects) representation in this dataset
print(objectsData_df.groupby(by="objectId")['timestamp'].count() * (100/len(objectsData_df))) # print a % of each object representation in objectsData_df
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="objectId", data=objectsData_df) # nice for a quick view on raw data
plt.show()

# --- show object repartition (with grouping of rare objects with a small representation)
labels_dict = {'subject': sub_sel, 'condition': 'condition_sel', 'scene': 'scene_sel'}
show_object_repartition(objectsData_df, objectsFormat_df)


# --- get data from one specific (object_id) for this (subject_id / condition / scene) 
sub_sel = 7
condition_sel = 'Narrative'  
scene_sel = 'HouseNarrative'  
metafile_mask = (metafile_df.subject_id == sub_sel) & (metafile_df.condition == condition_sel) & (metafile_df.scene == scene_sel) # select this subject / condition / scene
metafile_sel = metafile_df.loc[metafile_mask].squeeze() # get single row in metafile_df and convert to a pd Series 
objectsData_df, objectsFormat_df = importDataFile(metafile_sel.file_objectsData, metafile_sel.file_objectsFormat, data_type='objects')
eyetracking_df = importDataFile(metafile_sel.file_HTCViveProEyeData, [], data_type='eyetracking')

object_name = '/Player/SteamVRObjects/VRCamera' 
object_id = getObject_IdFromName(objectsFormat_df, object_name) # get object_id (int) from object name (str)
object_sel = objectsData_df.loc[objectsData_df.objectId == object_id] # /!\ shallow copy / view
object_sel = objectsData_df.loc[objectsData_df.objectId == object_id].copy() # deep copy
# ->> equivalent to: 
object_sel = objectsData_df.query('objectId == @object_id')


# -- calculation of some basic statistics on this specific object across all datasets (no aggregation yet !)
target_str = '/Player/SteamVRObjects/VRCamera' 
target_id = getObject_IdFromName(objectsFormat_df, target_str) # get object_id (int) from object name (str)
stats_info_df = calculate_basic_stats(metafile_df, 'objects', target_id) # here applies on all scenes and all participants
# stats_info_df = calculate_basic_stats(metafile_df, 'actions', 5) # (will not work... until we get some actions data)

# pivots on stats
pivot_stats_info_df = stats_info_df.pivot_table(values='total_pos_dist', index='scene', aggfunc='mean') # pivot: (distance) by (scene)
pivot_stats_info_df = stats_info_df.pivot_table(values='time_spend_on_scene_sec', index='scene', aggfunc='mean') # pivot: (time_spend_on_scene_sec) by (scene)
# here we miss some scenes... because they have ambiguous names ! (same scene names across condition, such as "tutorial")



# --- aggregate specific object_id data between datasets 

# switch between interactive plot in Qt5 and inline plot (works only in console)
interactive_plot = False
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell

shell = TerminalInteractiveShell.instance()
ipython = get_ipython()
if interactive_plot: ipython.magic("matplotlib Qt5") # use qt5 to plot -> MARCHO POOOO
if not interactive_plot: ipython.magic("matplotlib Inline")     # get back to inline plotting

# get this object's data for (this participant) and (every condition / scenes)
object_id = 0 # this corresponds to camera (i.e., user's moves)
sub_sel = 'all'
objectsData_merge_df = load_thisObjectData(metafile_df, object_id, subjectId = sub_sel, condition='all', scene='all')
objectsData_merge_df.info(memory_usage='deep') # print full memory info
mem_usage = get_MemoryUsage(objectsData_merge_df, verbose=True) # or just use getter fct to obtain the memory usage of the whole dataFrame (in MB)


# --- get quanti info on timestamps for all datasets for participant HMD and eyetracking data



#%% 5. Analyse data
# get quanti info on timestamps for all datasets for participant HMD and eyetracking data
# -- for one single object's data
object_id = 0 # this corresponds to camera (i.e., user's moves)
object_sel = objectsData_df.query('objectId == @object_id')
timestamp_serie = pd.Series(data=object_sel['pd_datetime']) 
timestamp_info = check_timestamps(timestamp_serie, verbose=True)
print("now let's resample all this...")
 # resample to evenly-spaced time serie and check again
object_sel_resampled = resample_timestamps(object_sel, timestamp_col='pd_datetime', target_fs='inferred', fill_gap_ms=100, interpolator='pchip')
timestamp_serie = pd.Series(data=object_sel_resampled['pd_datetime']) 
timestamp_info = check_timestamps(timestamp_serie)

# -- for a single eyetracking data
timestamp_serie = eyetracking_df['pd_datetime']
timestamp_info = check_timestamps(timestamp_serie)
# and now check timestamps after resampling
eyetracking_resampled_df = resample_timestamps(eyetracking_df, timestamp_col='pd_datetime', target_fs=12, fill_gap_ms=100, interpolator='pchip')
timestamp_serie = pd.Series(data=eyetracking_resampled_df['pd_datetime']) 
timestamp_info = check_timestamps(timestamp_serie)





