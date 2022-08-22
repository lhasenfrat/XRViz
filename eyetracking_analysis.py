#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:51:09 2022

@author: jonas
TODO: SUPPRIMER CE FICHIER

"""

#%% import libs
import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import yaml
from data_io import *
from data_proc import *
from questionnaire import *

from glob import glob
import numpy as np
import os, builtins



# --- init stuff
params = {}
params["root"] = pathlib.Path('/Users/jonas/Documents/PORTRAIT_local')
params["dataPath"] = params["root"].joinpath('data_Sophie')
params["profilingFile"] = params["dataPath"].joinpath('profiling.csv') # loading Jonas's prearranged dataset (v1)
params["configPath"] = pathlib.Path('/Users/jonas/Dropbox/Code/Python/XREcho_analysis/config') # TODO: change this (TEMPO)
params["configFile"] = params["configPath"].joinpath('design_Sophie.yaml')
params["exportPath"] = params["root"].joinpath('output-analysis')

metafile_df = createMetaFile(params["dataPath"], params["configFile"], isDataSophie=True)
profiling_df = pd.read_csv(params["profilingFile"], header=[0,1], encoding="iso8859_15")    # Jonas's pre-arrangement (v1)
 


# --- load all data for one specific subject / condition / scene
sub_sel = 7
condition_sel = 'Narrative'  
scene_sel = 'HouseNarrative'  
metafile_mask = (metafile_df.subject_id == sub_sel) & (metafile_df.condition == condition_sel) & (metafile_df.scene == scene_sel) # select this subject / condition / scene
metafile_sel = metafile_df.loc[metafile_mask].squeeze() # get single row in metafile_df and convert to a pd Series 

objectsData_df, objectsFormat_df = importDataFile(metafile_sel.file_objectsData, metafile_sel.file_objectsFormat, data_type='objects')
eventsData_df, eventsFormat_df = importDataFile(metafile_sel.file_eventsData, metafile_sel.file_eventsFormat, data_type='events')
eyetracking_df = importDataFile(metafile_sel.file_HTCViveProEyeData, [], data_type='eyetracking')

# save this to file for import using Salient360Toolbox
eyetrack_path = pathlib.Path('/Users/jonas/Documents/PORTRAIT_local/data_Sophie/S00').joinpath('eyetracking_df.csv')
eyetracking_df.to_csv(path_or_buf=eyetrack_path, index=False, encoding="iso8859_15") 


# --- init Salient360Toolbox
# Only print most important messages
builtins.verbose = 0
import Salient360Toolbox
from Salient360Toolbox.utils import misc
from Salient360Toolbox import helper
from Salient360Toolbox.generation import saliency as sal_generate
from Salient360Toolbox.generation import scanpath as scanp_generate

# Request to process head data
tracking = "H"
dim = [500, 1000]
path = eyetrack_path
savename = misc.getFileName(eyetrack_path)

# STOPPED HERE (issue installing opencv...)
# Load and process raw data
gaze_data, fix_list = helper.loadRawData(path,
                                   		# Gaze or Head tracking
                                   		tracking=tracking,
                                   		# If gaze tracking, which eye to extract
                                   		eye=tracking,
                                   		# Resampling at a different sample rate?
                                   		resample=None,
                                   		# Filtering algo and parameters if any is selected
                                   		filter=None,
                                   		# Fixation identifier algo and its parameters
                                   		parser=None)




""" 
INFOS ABOUT ACQUIRED EYETRACKING DATA:
CF https://github.com/Plateforme-VR-ENISE/XREcho/blob/main/Assets/ViveSR/Scripts/Eye/SRanipal_EyeData.cs
    
    
public enum SingleEyeDataValidity : int
            {
                /** The validity of the origin of gaze of the eye data */
                SINGLE_EYE_DATA_GAZE_ORIGIN_VALIDITY,
                /** The validity of the direction of gaze of the eye data */
                SINGLE_EYE_DATA_GAZE_DIRECTION_VALIDITY,
                /** The validity of the diameter of gaze of the eye data */
                SINGLE_EYE_DATA_PUPIL_DIAMETER_VALIDITY,
                /** The validity of the openness of the eye data */
                SINGLE_EYE_DATA_EYE_OPENNESS_VALIDITY,
                /** The validity of normalized position of pupil */
                SINGLE_EYE_DATA_PUPIL_POSITION_IN_SENSOR_AREA_VALIDITY"
                
/** The bits containing all validity for this frame.*/
                public System.UInt64 eye_data_validata_bit_mask;
                /** The point in the eye from which the gaze ray originates in millimeter.(right-handed coordinate system)*/
                public Vector3 gaze_origin_mm;
                /** The normalized gaze direction of the eye in [0,1].(right-handed coordinate system)*/
                public Vector3 gaze_direction_normalized;
                /** The diameter of the pupil in millimeter*/
                public float pupil_diameter_mm;
                /** A value representing how open the eye is.*/
                public float eye_openness;
                /** The normalized position of a pupil in [0,1]*/
                public Vector2 pupil_position_in_sensor_area;

 /** A instance of the struct as @ref EyeData related to the left eye*/
                public SingleEyeData left;
                /** A instance of the struct as @ref EyeData related to the right eye*/
                public SingleEyeData right;
                /** A instance of the struct as @ref EyeData related to the combined eye*/

From Charles:
"timestamp", "ticksSince1970", "gazeOrigin.x", "gazeOrigin.y", "gazeOrigin.z", 
"gazeDirection.x", "gazeDirection.y", "gazeDirection.z", "leftPupilDiameter", "rightPupilDiameter"

    Voilà les champs communs que j'ai conservé
Pour que ce soit compatible avec le HTC et le Varjo en tout cas
Dans le cas où l'eye tracking est perdu, je remplis les champs avec N/A
            
"""
