#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 22:18:26 2022

@author: jonas
"""

#%% 0. run main.py
# -> import libs + create profiling_df and metafile_df
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob2
import os
from data_io import *


#%% 1. select and analyse demographics data

# --- access to profiling data using multi-index axes
# access to profiling row by subject_id and scenario
sub_i = 57
scenario_i = 'Aesthetic'    
sel_df = profiling_df.loc[(sub_i, scenario_i)] # pd Series with full row of profiling_df
profiling_index = profiling_df.loc[(sub_i, scenario_i),('metadata','profiling_index')]  # get single value

# access to column for all participants
sel_df = profiling_df.loc[:,('demographics', 'gender')] # returns a pd Series
sel_df = profiling_df.xs('gender', axis='columns', level='profiling_items') # another way using 'xs', returns a pd DataFrame
# -> this accesses entire column (demographics (1st level) / gender (2nd level))
# -> both methods should be preffered over using chained indexing 'profiling_df['demographics']['gender']' (see Pandas user guide: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-view-versus-copy)
sel_df = profiling_df.loc[:, [('demographics', 'gender'), ('experimental', 'site_location')]] # access to x2 entire columns
index_scenario =  sel_df.index.to_frame(index=False)['scenario'] # access to scenario index

# --- filter data based on specific conditions
# filter indexes using a mask
data_sel = profiling_df.loc[:, ('demographics', 'gender')]  # mask on categorical data (here gender)
mask = (data_sel == "Femme") | (data_sel == "Autre")  # select 'Femme' AND 'Autre'
sel_df = profiling_df.loc[mask,:] 

# another way: filter indexes using powerful Panda's 'query' evaluation
col_save = profiling_df.columns # unfortunately this needs single-level columns, so we save it 
data_sel = profiling_df.droplevel(axis='columns', level='profiling_categories') # dropping first column level 
sel_df = data_sel.query('gender == ["Femme", "Autre"]') # calling query() on single column-lvel data    /!\ beware of ' and " !
sel_df.columns = col_save # re-integrate two-level column multi-indexing

# query() function allows for complex data selection with multiple conditions
sel_keys_1 = 'gender'
sel_values_1 =["Femme", "Autre"]
sel_df = data_sel.query(f'{sel_keys_1} in @sel_values_1') # using a list of values in a python variable
sel_keys_2 = 'site_location'
sel_values_2 ="INSA Lyon"
sel_df = data_sel.query(f'{sel_keys_1} in @sel_values_1 and {sel_keys_2} in @sel_values_2 and age > 22') # x3 conditions here !


# --- show some quick data summary
# data summary
profiling_describe_df = profiling_df.describe(include="all") 

# histogram of age repartition
sel_df = profiling_df.loc[:, ('demographics', 'age')] 
plot = sns.histplot(sel_df, kde=True)
plt.show()

# pivot 1: difference entre les scores d'engagement des hommes et des femmes par classe d'age et pour chaque scenario
data_sel = profiling_df.droplevel(axis='columns', level='profiling_categories') # dropping first column level 
sel_df = data_sel.query('gender in ["Femme", "Homme"]') # select data based on some conditions
age_cut = pd.cut(sel_df.loc[:,'age'], [0, 20, 21, 100]) # bin age values into 3 discrete non-uniform intervals 
pivot_profiling_df_1 = sel_df.pivot_table(values='engagement', index=['gender', age_cut], columns='scenario')

# pivot 2: nombre de participants pour chaque site par genre + par classe d'age
data_sel = profiling_df.droplevel(axis='columns', level='profiling_categories') # dropping first column level 
sel_df = data_sel.query('gender in ["Femme", "Homme"]') # select data based on some conditions
age_cut = pd.cut(sel_df.loc[:,'age'], [0, 20, 100]) # bin age values into 3 discrete non-uniform intervals 
pivot_profiling_df_2 = sel_df.pivot_table(values='profiling_index',index=['gender', age_cut], columns='site_location', aggfunc='count') # here using count as aggregate function -- here value hos no importance (simple element count)

# pivot 3: répartition (%) de niveaux de VR_mastery par genre
data_sel = profiling_df.droplevel(axis='columns', level='profiling_categories') # dropping first column level 
sel_df = data_sel.query('gender in ["Femme", "Homme"]') # select data based on some conditions
pivot_profiling_df_3 = sel_df.pivot_table(values='profiling_index',index='VR_mastery', columns='gender', aggfunc=lambda x: 100*len(x)/len(sel_df)) # custom aggregate function, here gives total percent (among men+women) -- here value hos no importance (simple element count)

del data_sel, sel_keys_1, sel_values_1, sel_keys_2, sel_values_2, age_cut, plot, col_save # clean workspace


#%% 2. select and analyse behavioral data

# --- load all data for one specific subject / scenario / scene
sub_i = 7
scenario_i = 'Narrative'  
scene_i = 'HouseNarrative'  
objectsData_df, objectsFormat_df = importObjectData(metafile_df.loc[(sub_i, scenario_i, scene_i)])
eventsData_df, eventsFormat_df, metadata_df = importEventsMeta(metafile_df.loc[(sub_i, scenario_i, scene_i)])   
eyetracking_df = importEyetrackingData(metafile_df.loc[(sub_i, scenario_i, scene_i)])  

# --- quick check memory usage for one dataframe
eyetracking_df.info(memory_usage='deep')

# --- quick bar plot with all tracked data (objects) representation in this dataset 
print(objectsData_df.groupby(by="actionId")['timestamp'].count() * (100/len(objectsData_df))) # print a % of each object representation in objectsData_df
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="actionId", data=objectsData_df) # nice for a quick view on raw data
plt.show()

# --- show object repartition (with grouping of rare objects with a small representation)
unknown_objectId_list, unknown_objectMask = maskUnknownObjects(objectsData_df, objectsFormat_df)
print('/!\ These objects present in objectsData are not listed in objectsFormat: ' + str(unknown_objectId_list))
df_sel = objectsData_df.loc[~unknown_objectMask] # remove unknown objects
groupBy_actionId = df_sel.groupby(by="actionId")['actionId'].count() * (100/len(objectsData_df)) # select by object_id and aggregate to get percentage / total count
groupBy_actionId = pd.DataFrame(data=groupBy_actionId)
groupBy_actionId = groupBy_actionId.rename(columns={'actionId': 'data_count'})
groupBy_actionId['objectNames'] = getObject_NameFromId(objectsFormat_df, groupBy_actionId.index.to_list())
rare_mask = (groupBy_actionId.data_count.sort_values().cumsum() < 10).sort_index() # this selects objects whose cumulative sum is <10% total representation in objectsData_df
rare_objects = rare_mask[rare_mask].index.to_list()
rare_summed = pd.DataFrame({'data_count': groupBy_actionId[rare_mask].data_count.sum(),'objectNames' : 'other objects whose cum sum <10%'},index = [0])
repartition_actionId = pd.concat([groupBy_actionId[~rare_mask], rare_summed], ignore_index=True, axis=0)
# pie chart
fig, ax = plt.subplots()
ax.pie(repartition_actionId['data_count'], labels=repartition_actionId['objectNames'], autopct='%.0f%%', shadow=False);
title_str = fig.suptitle(f'object repartition in data for subject "{sub_i}" / scenario "{scenario_i}" / scene "{scene_i}"')
plt.show()
# bar plot ()
# df_sel = objectsData_df.loc[~unknown_objectMask]
# df_sel.loc[df_sel.actionId.isin(rare_objects),'actionId']= -1 # here we replace all small represented objects to aggregate them with -1
# ax = sns.countplot(x="actionId", data=df_sel)
# plt.show()

# --- get data from one specific (object_id) for this (subject_id / scenario / scene) 
sub_sel = 7
scenario_sel = 'Narrative'  
scene_sel = 'HouseNarrative'  
objectsData_df, objectsFormat_df = importObjectData(metafile_df.loc[(sub_sel, scenario_sel, scene_sel)]) 

object_sel = '/Player/SteamVRObjects/VRCamera' 
object_id = getObject_IdFromName(objectsFormat_df, object_sel) # get object_id (int) from object name (str)
df_sel = objectsData_df.loc[objectsData_df.actionId == object_id] # /!\ shallow copy / view
df_sel = objectsData_df.loc[objectsData_df.actionId == object_id].copy() # deep copy 
# ->> equivalent to: 
df_sel = objectsData_df.query('actionId == @object_id')


del ax, fig, groupBy_actionId, index_scenario, mask, pivot_profiling_df_1, pivot_profiling_df_2, pivot_profiling_df_3, metadata_df
del profiling_describe_df, profiling_index, rare_mask, rare_objects, rare_summed, repartition_actionId, title_str, unknown_objectId_list, unknown_objectMask


#%% --- aggregate specific object_id data between datasets 
# get this object's data for (this participant) and (every scenario / scenes)
object_id = 0 # this corresponds to camera (i.e., user's moves)
sub_sel = 'all'
scenario_sel = 'Narrative'  
scene_sel = 'HouseNarrative'  
objectsData_merge_df = load_thisObjectData(metafile_df, object_id, sub_sel, scenario='all', scene='all')
objectsData_merge_df.info(memory_usage='deep') # print full memory info
mem_usage = get_MemoryUsage(objectsData_merge_df, verbose=True) # or just use getter fct to obtain the memory usage of the whole dataFrame (in MB)


# get this object's data for (all participant) and (this scenario / scene)
metafile_df.profiling_index




# TODO: select sub_id based on demographics -> aggregate object_id data for this scenario+scene between participants
# TODO: check timestamps in data object + eyetracking
