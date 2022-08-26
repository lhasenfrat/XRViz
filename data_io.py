#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:03:01 2022

@author: jonas
"""

import pathlib
import numpy as np
import pandas as pd
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from questionnaire import *

def loadConfigPaths(pathFile):
    '''
        Loads paths of all relevants files for the experiment

        Parameters
        ----------
        pathsFile : pathlib.Path
            path of the .yaml paths file.

        Returns
        -------
        params : dict
            contains paths of the experiement descrition, the criteria list and the data used for the experiment
             '''
    check_filePath_validity(pathFile)
    assert pathlib.Path(pathFile).suffix.lower() == '.yaml', "experiment paths file must be .yaml format"

    # --- read template config file
    with open(pathFile, 'r') as file:
        yamlread = yaml.full_load(file)

        # --- create related fields
    params = dict()
    folder=pathFile.parent
    params["configExperiment"] = folder.joinpath(yamlread['DESIGN'])
    params["configCriteria"] = folder.joinpath(yamlread['CRITERIAS'])
    params["dataPath"] = folder.joinpath(yamlread['DATA'])
    params["profilingFile_Sophie"] = folder.joinpath(yamlread['PROFILING'])
    return params
def loadConfigExperiment(configFile):
    '''
    Loads informations related to experiment structure from config file

    Parameters
    ----------
    configFile : pathlib.Path
        path of the .yaml config file.

    Returns
    -------
    config_META : dict 
        contains metadata information about the experiment.
    config_GLOBAL : dict
        contains global parameters used throughout the experiment (mostly within Unity game engine)
    config_DESIGN : dict 
        contains information about experiment structure (experimental design):
        ['questionnaire'] : list of string qith questionnaire names
        ['condition_scene_structure_df']: pandas dataFrame with definition of experimental conditions and their associated scenes
        ['experiment_sequence_df']: pandas dataFrame with definition of experiment sequencing
                                    because there could be any number of sequence subset at each step
    '''
    
    # --- assert inputs
    check_filePath_validity(configFile)
    assert pathlib.Path(configFile).suffix.lower() == '.yaml', "experiment config file must be .yaml format"

    # --- read template config file
    with open(configFile, 'r') as file:
        yamlread = yaml.full_load(file) 
       
    # --- create related fields   
    config_design = dict()
    config_design['META'] = yamlread['META']
    config_design['DESIGN'] = dict()
    config_design['DESIGN'].setdefault('questionnaires', yamlread['DESIGN']['questionnaires'])
    config_design['DESIGN'].setdefault('condition_scene_structure_df', pd.DataFrame(yamlread['DESIGN']['structure']))
    
    # --- create dataframe for experiment sequence
    def convert_step_to_dataframe(dico, has_substep):
        sequence_df = pd.DataFrame(dico)
        for step_ix, current_step in sequence_df.iterrows():
            has_substep = current_step['content'] is not None
            while has_substep:
                step_df, has_substep = convert_step_to_dataframe(current_step['content'], has_substep) # recursive call because they could be any number of sequence subset at each step
                sequence_df.at[step_ix,'content'] = step_df 
        return sequence_df, has_substep
    
    sequence_df, has_substep = convert_step_to_dataframe(yamlread['DESIGN']['experiment_sequence'], False)
    config_design['DESIGN'].setdefault('experiment_sequence_df', sequence_df)
    
    return config_design

def loadConfigCriterias(configFile):
    check_filePath_validity(configFile)
    assert pathlib.Path(configFile).suffix.lower() == '.yaml', "experiment config file must be .yaml format"

    # --- read template config file
    with open(configFile, 'r') as file:
        yamlread = yaml.full_load(file)

        # --- create related fields
    config_criterias = dict()
    config_criterias['PROFILING'] = yamlread['PROFILING']

    return config_criterias

def loadConfigQuestionnaire(configFile):
    '''
    Loads informations related to one questionnaire from its config file

    Parameters
    ----------
    design_configFile: optional, str or pathlib.Path
        path of the .yaml config file with experiment structure

    Returns
    -------
    questionnaire_config: dict, with fields as
        META : dict 
            metadata information about the questionnaire.
        BLOCS : dict
            infos about questionnaire constitutive blocs
        ITEMS : dict 
            infos about questionnaire items (questions)
        SEQUENCE : dict 
            infos regarding questionnaire sequencing (blocs orders)
        SCORES : dict 
            infos score calculation from participants answers
    '''
    
    # --- assert inputs
    check_filePath_validity(configFile)
    assert pathlib.Path(configFile).suffix.lower() == '.yaml', "experiment config file must be .yaml format"
    
    # --- read template config file
    with open(configFile, 'r') as file:
        yamlread = yaml.full_load(file) 
    
    # --- fill fields and convert what's needed to dataFrames
    questionnaire_config = dict()
    questionnaire_config['META'] = yamlread['META']
    questionnaire_config['BLOCS'] = pd.DataFrame(yamlread['BLOCS'])
    questionnaire_config['ITEMS'] = pd.DataFrame(yamlread['ITEMS'])
    questionnaire_config['SEQUENCE'] = yamlread['SEQUENCE']
    questionnaire_config['SCORES'] = pd.DataFrame(yamlread['SCORES'])
    
    return questionnaire_config


def createMetaFile(basePath, design_configFile = None, subjectId_list=None, isDataSophie=False):
    """creates a metafile_df dataFrame with all metadata structure (subject_id, condition, scene and all file paths) 
    This function is to be called a single time, after passations (data is already acquired) 
    and before starting doing some analysis on a new computer. 
    It uses .yaml 'design_configFile' to generate condition names + experiment structure 
    and crosscheck that everything is alright on disk (no missing data, etc.)
    
     --- Inputs
      basePath: pathlib.Path
          full path of the data directory 
      design_configFile: optional, pathlib.Path
          path of the .yaml config file with experiment structure (here we use the names of conditions and scenes)
      subject_id: optional, int
          list of subject id to import data from 
          default: None (build metafile with all subjects on disk regardless of missing data)
      isDataSophie: optional, bool
          selector to adapt to obsolet dataset structure specific to Sophie's data (2022)
        
    --- Outputs
    metafile_df: Pandas dataframe with all metadata structure (subject_id, condition, scene and all datafile paths) 
    
    --- Infos
    - base directory is organized  each following file for each VR scene in the session:
    	- CONFIG: one directory with all configuration files used to run and analyse this experiment
    		- exp-structure_title-exp.yaml: config file with experiment structure (names of questionnaires, names of conditions and scenes, sequence, etc.)
    		- pyquest_title-quest.yaml: one config file for each questionnaire that is used (contains all info required to present and analyse that questionnaire)
    	- DATA: one directory containing all participant data for this experiment (questionnaires + behavior + eyetracking + physio)
    		- subject dir: one directory per participant with name "S"+(index participant)
    			- profiling dir: inside each subject dir, one dir containing questionnaire answers /!\ -> these files are parsed in function importProfilingData()
    				- one .csv file containing participant answer for each questionnaire
    				- name of these files are: 
                        "S"+(index participant) + '_' + title_questionnaire (defined in pyquest_title-quest.yaml)
                        + '_' + condition (if quest is presented for each condition) + .csv
                        -> e.g., "S7_answers_DFS_Narrative.csv"
    			- condition dirs: inside each subject dir, one dir for each condition:
    				- folder name is the same as condition name defined in exp-structure_title-exp.yaml 
    				- this folder includes the following datasets (.csv files):
    				- "eventsData": data about contextual VR events (removing headset, loading scene, etc.)
    				- "eventsFormat": event metadata (event id + event name)
    				- "HTCViveProEyeData": HTC Vive Pro Eye data in original format
    				- "objectsData": VR objects data (one line per frame for each moving tracked object)
    				- "objectsFormat": XR-Echo info for VR replay (object type, trackedData, position, etc.)
    """ 
    
    # --- assert inputs
    assert isinstance(basePath,pathlib.Path), "'basePath' dir should be a valid pathlib"
    assert isinstance(design_configFile,pathlib.Path) or not design_configFile, "'design_configFile' should be a valid pathlib"
    assert subjectId_list is None or (type(subjectId_list) is list and type(subjectId_list[0]) is int), \
        "subject_id must be a list of int"
    assert type(isDataSophie) is bool, "isDataSophie must be True or False"
    assert design_configFile or isDataSophie, "when no config file is provided, 'isDataSophie' must be True to infer experiment structure"
    
    # --- look for participant directories 
    dirNames_onDisk = sorted(list(basePath.glob('S*')))
    try:
        subject_id_onDisk = list(map(lambda x: np.int32(str(x.name)[-2:]), dirNames_onDisk))
    except:
        print("something funny with name of participant directories within: " + str(basePath))
    if subjectId_list:  # a list of participants to parse was provided
        nonExistantIds = set(subjectId_list) - set(subject_id_onDisk)
        assert set(subjectId_list) <= set(
            subject_id_onDisk), f"these requested subjects ID are not on disk: '{nonExistantIds}'"
    else:
        subjectId_list = subject_id_onDisk
    
    # filenames on disk are hardcoded here (with same names from Unity)
    fileName_cols = ['file_objectsData', 'file_objectsFormat', 'file_actionsData', 'file_actionsFormat', \
                     'file_HTCViveProEyeData', 'file_metadata' 'file_eventsData', 'file_eventsFormat'] # this is name of columns in metafile_df
    fileName_scenes_onDisk = ['objectsData', 'objectsFormat','actionsData', 'actionsFormat', \
                              'HTCViveProEyeData', 'metadata','eventsData', 'eventsFormat'] # this is name of files on disk 
   
    # --- create a metafile_df dataFrame with all metadata structure (subject_id, condition, scene and all file paths)
    # populate condition / scene structure from information in config file 
    if design_configFile.exists():
        config_design = loadConfigExperiment(design_configFile)   
        exp_structure_df = config_design['DESIGN']['condition_scene_structure_df']    
    elif isDataSophie: # /!\ specific to Sophie's data (skip config file)
        exp_structure = [{'condition': 'Aesthetic', 'scene': ['Tutorial', 'HouseAestheticBase', 'HouseAestheticSpace', 'PostApo']}, 
                         {'condition': 'Narrative', 'scene': ['Tutorial', 'HouseNarrative']}, 
                         {'condition': 'Goals', 'scene': ['Tutorial', 'HouseObjectives']}]
        exp_structure_df = pd.DataFrame(exp_structure)
    if isDataSophie: # /!\ specific to Sophie's data (even when confif file is provided)
        condition_onDisk = ['Esthetique','Objectif','Narratif']  # special case where folder names are in French /!\ ORDER IS IMPORTANT HERE
        subjectId_list = [7,10,11,15,16,17,18,19,20,22,24,25,26,28,29,30,32,34,35,37,38,39,40,41,43,44,46,47,51,52,53,55,56]
        #   - S14 has no ObjectFormat file in scene/condition:  Goal/HouseObjectives
        #   - S49 has no ObjectFormat file in scene/condition:  Aesthetic/HouseAestheticSpace
        #   - S57 has no eyetracking file in scene/condition:  Aesthetic/postApo
        
    condition_list = sorted(exp_structure_df.condition.to_list()) # get list of condition names
    condition_scene_tuples = [] # create a list of tuples (condition, scene) for all combinations
    for row in exp_structure_df.itertuples(index=False):
        for scene in row.scene:
            condition_scene_tuples.append((row.condition, scene))  
    prod_sub_condition_scene = list(itertools.product(subjectId_list, condition_scene_tuples))
    sub_condition_scene_list = [(a, b[0], b[1]) for a, b in prod_sub_condition_scene]  
    sub_condition_scene_array = list(zip(*sub_condition_scene_list))
    
    # create Pandas dataframe 
    metafile_df = pd.DataFrame(columns=['subject_id', 'condition', 'scene', 'profiling_index', 'record_index','data_dir','profiling_dir']+fileName_cols)             
     
    # populate subject_id, condition, scene, and record_index
    metafile_df.subject_id = sub_condition_scene_array[0]
    metafile_df.condition = sub_condition_scene_array[1]
    metafile_df.scene = sub_condition_scene_array[2]
    metafile_df.record_index = list(range(len(metafile_df)))  # here we create unique integer record_index for each tuple(sub_id,condition,scene) respecting record order in time
    metafile_df.sort_index(inplace=True)  # sort data for efficient indexing (otherwise raises PerformanceWarning)
    # --- populate each participant's datafiles
    # parsing dirs: check that a folder on disk exist for all condition listed in config file
    for sub_dir_ix, sub_dir in enumerate(dirNames_onDisk):
        current_sub = np.int32(str(sub_dir.name)[-2:])
        if current_sub in subjectId_list:


            subdir_profiling = next(sub_dir.glob('**/profiling')) # profiling folder
            subdir_onDisk = list(sub_dir.glob("**/**"))[1:] # all subfolders
            subdir_names = list(map(lambda x: x.name, subdir_onDisk))
            metafile_mask = (metafile_df.subject_id == current_sub)# select this subject_id 
            if subdir_profiling:
                metafile_df.loc[metafile_mask,'profiling_dir'] = subdir_profiling
            
            # populate data directories (same for all scenes within a particular subject / condition)
            if not isDataSophie:

                assert set(condition_list) <= (set(subdir_names)-{'profiling'}), f"missing these condition {set(condition_list) - set(subdir_names)} in {sub_dir}"
                for condition_ix, current_condition in enumerate(condition_list):
                    metafile_mask = (metafile_df.subject_id == current_sub) & (metafile_df.condition == current_condition)  # select this subject_id AND current_condition
                    metafile_df.loc[metafile_mask,'data_dir'] = [i for i in subdir_onDisk if i.name == current_condition][0] 
            else: # in Sophie's data, condition/scenario folder on disk have different name (in french) than conditon/scenario in data (in english)  
                assert set(condition_onDisk) <= (set(subdir_names)-{'profiling'}), f"missing condition(s) {set(condition_onDisk) - set(subdir_names)} in {sub_dir}"
                for condition_ix, current_condition in enumerate(condition_list):
                    metafile_mask = (metafile_df.subject_id == current_sub) & (metafile_df.condition == current_condition)  # select this subject_id AND current_condition
                    metafile_df.loc[metafile_mask,'data_dir'] = [i for i in subdir_onDisk if i.name == condition_onDisk[condition_ix]][0] # same: a = list(filter(lambda x: x[-11:-1] =='Esthetique',subdir_onDisk))
    # parsing files: check and get all files paths for each participant and condition
    for meta_ix, meta_row in metafile_df.iterrows():
        for file_col, scene_onDisk in zip(fileName_cols, fileName_scenes_onDisk):
            curr_files = sorted(meta_row.data_dir.glob('*' + scene_onDisk + '*' + meta_row.scene + '.csv'))
            if len(curr_files) == 1:
                metafile_df.at[meta_ix, file_col] = curr_files[0]  # one file: path as pathlib.Path
            elif len(curr_files) > 1:
                metafile_df.at[meta_ix, file_col] = curr_files  # two or more files: path as a list of pathlib.Path
    return metafile_df


def newExperienceConfig(basePath, design_configFile, writeFiles = False):
    """From experiment configuration ('design_configFile' .yaml file),  
    generate all information required by Unity to run the experiment:
        - expe_metafile_df: a metafile with all directory paths for all participants on this local machine
        - expe_sequence_files: experimental sequence for each participant
    
    This function is to be called a single time, after writing experiment configuration
    and before running passations on a new computer. 
    It is here that we generate, for example, the latin square randomization 
    for condition sequencing.
        
     --- Inputs
      basePath: pathlib.Path
          root path of the data directory 
      design_configFile: pathlib.Path
          path of the .yaml config file with experiment structure 
      writeFiles: bool (optionnal, default is False)
          create all participant directories within root 'basePath'
          write a single .csv file on root with 'expe_metafile_df' (see below)
          write one .csv file for each participant 'subject_sequence_df' 
          in participant directory within root 'basePath' (see below)
          
    --- Outputs
    
        expe_metafile_df: Pandas dataframe 
            A metafile with all directory paths for all participants on this 
            local machine. Has one row per participant, with following columns:
                - subject_id: ID participant X
                - root dir (str): root directory for this participant's data
                - configuration file for this participant
                - condition_order: tuple x N conditions
                - questionnaire_order: tuple x N questionnaires
                 
        subject_sequence_df_dict: dict of Pandas dataframes
            Each dataframe contains sequencing information for one participant,
            with one row for each item to present. Columns are:
                - step index (str)
                - element (str): 'questionnaire', 'condition' or 'scene'
                - item (str): name of single questionnaire, condition or scene to present
                - dir_path (pathlib.Path): 
                    when element is condition: path of the condition directory in which to save participant data
                    when element is questionnaire: path of the .csv file in which to store participant answers and scores 
    """
    
    # -- implement some internal utils functions
    def items_fill(step, condition_substep=None):
        """fill in items when nothing is specified in experiment_sequence of .yaml config file 
        """
        if step['element'] == 'questionnaires':
            step['item'] = config_design['DESIGN']['questionnaires']  
        elif step['element'] == 'condition':
            step['item'] = config_design['DESIGN']['condition_scene_structure_df']['condition'].tolist()
        elif step['element'] == 'scene':
            if not condition_substep:
                xss = config_design['DESIGN']['condition_scene_structure_df']['scene'].tolist() # get all scenes
            else:
                mask_cond = config_design['DESIGN']['condition_scene_structure_df']['condition']== condition_substep
                xss = config_design['DESIGN']['condition_scene_structure_df'].loc[mask_cond,'scene'].tolist() # get all scenes in this condition
            scene_flat_list = [x for xs in xss for x in xs] # we need to flatten all that (lists in lists)
            step['item'] = scene_flat_list
        else:
            raise ValueError(f"Error in design config file, unknown 'element' in experiment_sequence: {step['element']}")
        return step
    
    
    def create_items_array(step, nb_subs):
        """Internal utils function that creates a ndarray of (nb_sub x nb_items)
        with order or randomizing defined in .yaml config file.
        """
        # create items list with ordering or with randomizing
        if not step['order']: 
            step['order'] = 'ordered' # when nothing is specified ordering by default 
        item_all_subs = np.reshape(nb_subs*step['item'],(nb_subjects,len(step['item']))) # duplicate for N participants keeping current order in item list    
        if step['order'] == 'ordered':
             pass # nothing to do here, we just keep the current item order
        elif step['order'] == 'random':
            rng = np.random.default_rng()
            rand_index = [list(rng.permutation(range(len(step['item'])))) for sub_ix in range(nb_subs)] # random index of shape (nb_sub x nb_items)
            item_all_subs = [list(items_this_sub[this_index]) for items_this_sub,this_index in zip(item_all_subs, rand_index)]
        elif step['order'] == 'latin_square':
            rand_index = balanced_latin_squares(len(step['item']))
            rand_index = (rand_index*int(np.ceil(nb_subs/len(rand_index))))[0:nb_subs] # create a latin square index of shape (nb_sub x nb_items)
            item_all_subs = [list(items_this_sub[this_index]) for items_this_sub,this_index in zip(item_all_subs, rand_index)]
        else: 
            raise ValueError(f"Error in design config file, unknown 'order' in experiment_sequence: {step['order']}")
        return item_all_subs
    
    
    def create_step_df(step, subject_id, items, profiling_paths, subjectDir_paths):
        """Internal utils function that creates a step_df pandas Dataframe
        filled from a new step infos.
        """
        nb_steps = len(items)
        step_dict = dict()
        step_dict['step_index'] = [str(step.step)]*nb_steps
        step_dict['element'] = [step.element]*nb_steps 
        step_dict['order'] = [step.order]*nb_steps 
        step_dict['item'] = items 
        if step.element == 'questionnaires':
            quest_filenames =  [(profiling_paths[subject_id].parent.name) + '_answers_' + curr_quest + '.csv' for curr_quest in step_dict['item']] # example: S07_answers_demographics.csv
            quest_filepaths = [profiling_paths[subject_id].joinpath(quest_filename) for quest_filename in quest_filenames]
            step_dict['dir_path'] = quest_filepaths
        elif step['element'] == 'condition' :
            cond_dirnames = step_dict['item']
            cond_dirpaths = [subjectDir_paths[subject_id].joinpath(dirname) for dirname in cond_dirnames]
            step_dict['dir_path'] = cond_dirpaths
        elif step['element'] == 'scene':
            step_dict['dir_path'] = None
        step_df = pd.DataFrame(step_dict) # create dataFrame from dict (one or more rows)
        return step_df
    
    
    # --- assert inputs
    assert isinstance(basePath,pathlib.Path), "'basePath' dir should be a valid pathlib"
    assert isinstance(design_configFile,pathlib.Path), "'design_configFile' should be a valid pathlib"
    assert design_configFile.exists(), f"invalid config file '{design_configFile.name}'"
    assert isinstance(writeFiles, bool), "parameter 'writeFiles' must be True or False"
    
    # --- load experiment info from .yaml config file 
    config_design = loadConfigExperiment(design_configFile)   
    exp_structure_df = config_design['DESIGN']['condition_scene_structure_df']    
    experiment_sequence_df = config_design['DESIGN']['experiment_sequence_df']   
    
    # -- generate metafile with all required paths for this local machine and for each participant
    condition_list = sorted(exp_structure_df.condition.to_list()) # get list of condition names 
    nb_subjects = config_design['META']['nb_participant_expected']
    subjectId_list = list(range(nb_subjects)) # ordered list with all subject IDs starting from 0 (int)
    subjectDir_names = [f"S{sub_i:02}" for sub_i in subjectId_list] # list of str with format: 'S' + subjectId (minimum 2-digits string)
    subjectDir_paths = [basePath.joinpath(subjectDir) for subjectDir in subjectDir_names]
    sequencing_files = [subjectDir + "_sequencing.csv" for subjectDir in subjectDir_names]
    sequencing_paths = [sub_path.joinpath(seq_file) for sub_path, seq_file in zip(subjectDir_paths,sequencing_files)]
    profiling_paths = [sub_path.joinpath('profiling') for sub_path in subjectDir_paths]
    expe_metafile_df = pd.DataFrame(columns=['subject_id', 'subject_dir', 'sequencing_file', 'condition_order','questionnaire_order'])    
    expe_metafile_df.loc[:, 'subject_id'] = subjectId_list
    expe_metafile_df.loc[:, 'subject_dir'] = subjectDir_paths
    expe_metafile_df.loc[:, 'sequencing_file'] = sequencing_paths
    expe_metafile_df['condition_order'] = expe_metafile_df.apply(lambda _: list(), axis=1) # initialize with empty lists 
    expe_metafile_df['questionnaire_order'] = expe_metafile_df.apply(lambda _: list(), axis=1) # initialize with empty lists 
     
    # -- generate items sequence (questionnaires, conditions, scenes) for all participants as defined in config_design .yaml file
    step_cols = ['step_' + str(step) for step in experiment_sequence_df.step.to_list()] # name of step columns ('step_1', 'step_2', etc.)
    subject_sequence_df_dict = dict() 
    for sub_i in subjectId_list:
        subject_sequence_df_dict[sub_i] = pd.DataFrame(columns=['step_index', 'element', 'order', 'item', 'dir_path']) # (one row per step)    
    
    # iterate over each experiment step defined in config file and fill expe_config_df
    for ix_i, step_i in experiment_sequence_df.iterrows():
        
        # create sequence of items for all participants
        if not step_i['item']:
            step_i = items_fill(step_i)
        
        # create items array 
        item_all_subs_i = create_items_array(step_i, nb_subjects)
        
        # assert that any substep exists under a step with element of type 'condition' 
        assert step_i['content'] is None or (step_i['content'] is not None and step_i['element'] == 'condition'), \
            f"Error in design config file, a substep can only be defined for 'condition' elements: {step_i['content']}"
        
        # case where there are substeps to consider in this step 
        if step_i['content'] is not None : 
            # for each participant, update subject_sequence_df with these new steps      
            for sub_i in subjectId_list:
                # iterate over conditions (necessarily, as it cannot be another type of element)   
                for cond_i in item_all_subs_i[sub_i]:   
                    step_df = create_step_df(step_i, sub_i, [cond_i], profiling_paths, subjectDir_paths)
                    subject_sequence_df_dict[sub_i] = pd.concat([subject_sequence_df_dict[sub_i], step_df], ignore_index=True) # udpdate subject_sequence_df with these steps (pd dataframe concatenation)
                    expe_metafile_df.at[sub_i,'condition_order'] += step_df.item.to_list() # update overall condition order
                    for ix_ii, step_ii in step_i['content'].iterrows():
                        if step_ii.element == 'questionnaires': # ignore all other types of elements 
                            # create questionnaire names with ordering or randomizing across all participants
                            item_all_subs_ii = create_items_array(step_ii, nb_subjects) 
                            item_all_subs_ii = np.reshape([[item_sub + '_' + cond_i] for item_sub in item_all_subs_ii.flatten()],item_all_subs_ii.shape) # append condition name to questionnaire name, example: 'DFS_Aesthetic' 'TPI_Aesthetic'
                            step_df = create_step_df(step_ii, sub_i, item_all_subs_ii[sub_i], profiling_paths, subjectDir_paths)
                            subject_sequence_df_dict[sub_i] = pd.concat([subject_sequence_df_dict[sub_i], step_df], ignore_index=True) # udpdate subject_sequence_df with these steps (pd dataframe concatenation)
                            expe_metafile_df.at[sub_i,'questionnaire_order'] += list(item_all_subs_ii[sub_i]) # add these questionnaires names to overall questionnaire_order    
        # case where there are no substeps to consider in this step
        else:
            # for each participant, update subject_sequence_df with these new steps   
            for sub_i in subjectId_list: 
                step_df = create_step_df(step_i, sub_i, item_all_subs_i[sub_i], profiling_paths, subjectDir_paths)
                subject_sequence_df_dict[sub_i] = pd.concat([subject_sequence_df_dict[sub_i], step_df], ignore_index=True) # udpdate subject_sequence_df with these steps (pd dataframe concatenation)
                if step_i.element == 'questionnaires': # update overall questionnaire order 
                    expe_metafile_df.at[sub_i,'questionnaire_order'] += step_df.item.to_list() # add these questionnaires names to overall questionnaire_order
                    
    # -- create directories and write files to disk if necessary
    if writeFiles:
         
        # write a single .csv file on root with 'expe_metafile_df' 
        metafile_path = basePath.joinpath('experiment_metafile.csv')
        expe_metafile_df.to_csv(path_or_buf=metafile_path, index=False, encoding="iso8859_15") 
        print('Experiment metafile saved on disk: ' + str(metafile_path))
        # expe_metafile_df_read = pd.read_csv(metafile_path, encoding="iso8859_15") # for debug: read it
        
        # iterate over participants in expe_metafile_df and subject_sequence_df_dict
        for sub_i in subjectId_list:
            
            # create participant directories within root 'basePath' 
            subject_dir = expe_metafile_df.at[sub_i,'subject_dir']
            if not subject_dir.exists():
                subject_dir.mkdir(parents=False, exist_ok=False) # raises error if not existing parent (root) and pre-existing participant dir (i.e., no override)
            
            # write one .csv file for each participant with 'subject_sequence_df' in participant directory within root 'basePath' 
            subject_sequence_df = subject_sequence_df_dict[sub_i]
            subject_sequence_path = expe_metafile_df.at[sub_i,'sequencing_file']
            subject_sequence_df.to_csv(path_or_buf=subject_sequence_path, index=False, encoding="iso8859_15") 
            #subject_sequence_df_read = pd.read_csv(subject_sequence_path, encoding="iso8859_15")  # for debug: read it
            print('Participant sequencing saved on disk: ' + str(subject_sequence_path))
            
            # create condition directories within participant directory
            for cond_i in condition_list:
                cond_dir = subject_dir.joinpath(cond_i)
                if not cond_dir.exists():
                    cond_dir.mkdir(parents=False, exist_ok=False) # raises error if not existing parent (root) and pre-existing participant dir (i.e., no override)
            
            
    return expe_metafile_df, subject_sequence_df_dict
         
    


def importProfilingData(metafile_df, design_configFile):
    """ Creates profiling data from all participants listed in metafile_df 
        importing answers to all questionnaires listed in design_configFile.
        For each questionnaire compute all scores. 
        This function relies on config files defining each questionnaire used 
        in the experiment.
        Also returns metafile_df updated with indexes matched to profiling_df data
        
    -- Inputs
    metafile_df: Pandas DataFrame 
           used here to get list of participants to import data from with their
           profiling dir paths on disk
    design_configFile: str
        path of the .yaml config file with experiment structure (here we use the names of questionnaires)
        
    -- Outputs
    profiling_df : Pandas dataframe
        dataframe where questionnaire data is aggregated across all participants
    metafile_df: Pandas DataFrame 
           updated with a 'record_ix' column matching corresponding indexes in profiling_df.
    """

    # -- assert inputs
    assert (type(metafile_df) is pd.core.frame.DataFrame) or not metafile_df, \
        "input 'metafile_df' must be a valid Pandas dataframe"
    assert isinstance(design_configFile,pathlib.Path) or not design_configFile, "'design_configFile' should be a valid pathlib"
    
        
    # -- load config file, get questionnaire name and load questionnaire configs
    configPath = pathlib.Path(design_configFile).parents[0]
    config_design = loadConfigExperiment(design_configFile)   
    
    # -- check that a config file exists for each questionnaire listed in config_design
    quest_configFiles_onDisk = sorted(configPath.glob('pyquest_*.yaml')) # list of questionnaire config files on disk (PATHS)
    quest_list_onDisk = list(map(lambda x: x.stem.split("_",1)[1], quest_configFiles_onDisk)) # list of questionnaire config files on disk (NAMES)
    quest_list_design = config_design['DESIGN']['questionnaires'] # list of questionnaires utilized for this experiment (as described in design_configFile)
    assert set(quest_list_design) <= set(quest_list_onDisk), f"config files for these requested questionnaires are missing: {set(quest_list_design)-set(quest_list_onDisk)}"
    
    # -- create 2-level column names for profiling_df dataFrame
    profiling_columns_lev1 = []     # contains names of categories
    profiling_columns_lev2 = []     # contains names of items
    
    # /!\ hardcoded fill of column names for metadata and experimental items (assuming one completion timestamp for each questionnaire)
    # future prog: this could be changed to something more configurable sa as to anticipate for potential change in what to record as experimental data
    # Experimental datafiles should typically contain the following fields:
        #'subject_id' (int), 'site_location' (str), 'condition' (str), 'condition_order' (int) 
        # + x1 completion timestamp for each filled questionnaire    
    # -> see: questionnaire.save_experimental_data(questionnaire_list, expe_info_dict)
    completion_strings = [quest_i + '_completion' for quest_i in quest_list_design]
    experimental_items = ['condition','condition_order','passation_site'] + completion_strings 
    profiling_columns_lev1 += ['experimental']*2 # this is for 'profiling_ix' and 'subject_id'
    profiling_columns_lev1 += ['experimental']*len(experimental_items)
    profiling_columns_lev2 += ['profiling_index', 'subject_id'] # items corresponding to metadata category
    profiling_columns_lev2 += experimental_items # experimental items
    
    # check in experiment config file (using section 'experiment_sequence') which questionnaires are presented a single time or once per condition
    list_quest_per_condition = [] # questionnaires in this list are presented again at each condition
    list_quest_global = [] # questionnaires in this list are presented only once
    exp_sequence_df = config_design['DESIGN']['experiment_sequence_df']
    for step_lev1_ix, step_lev1 in exp_sequence_df.iterrows():
        # add questionnaires listed in level1-steps to global list 
        if step_lev1.element == 'questionnaires':
            list_quest_global += step_lev1['item']
        # add questionnaires listed in level1-steps to per condition list 
        elif step_lev1.element == 'condition' and isinstance(step_lev1.content, pd.DataFrame):
            for step_lev2_ix, step_lev2 in step_lev1.content.iterrows():
                if step_lev2.element == 'questionnaires':
                    list_quest_per_condition += step_lev2['item']
    assert set(list_quest_global+list_quest_per_condition) == set(quest_list_design), "something wrong in 'experiment_sequence'... did you list here all questionnaires? \n-> fix experience config file: " + design_configFile

    # fill column names from items listed in questionnaires config files
    question_cols_design = dict() # this dict contains all column names (tuple cat./item) for questions only (no scores)
    config_quest_dict = dict() # this dict contains all questionnaire config that we load from disk
    quest_configFile_list = [] # list with path of all config files for all questionnaires listed in design config file
    for quest_i in quest_list_design:
        quest_configFile = pathlib.Path(configPath,'pyquest_'+quest_i+'.yaml')
        quest_configFile_list.append(quest_configFile)
        config_quest = loadConfigQuestionnaire(quest_configFile)    
        config_quest_dict[quest_i] = config_quest
    
        # get names of items and categories
        question_items_names = config_quest['ITEMS']['item_id'].to_list()
        if not config_quest['SCORES'].empty:
            question_category_names = [quest_i+'_items']*len(question_items_names)
            score_items_names = config_quest['SCORES']['score_id'].to_list()
            score_category_names = [quest_i+'_scores']*len(score_items_names)
        else:
            question_category_names = [quest_i]*len(question_items_names)
            score_items_names = []
            score_category_names = []
        category_names = question_category_names + score_category_names # column name level-1 (categories) to write on file
        item_names = question_items_names + score_items_names # column name level-2 (items) to write on file
        question_cols_design[quest_i] = list(zip(question_category_names, question_items_names))  # this are items of questionnaire data expected when reading the design config file
        
        # insert category and item names from this questionaire into new profiling columns
        profiling_columns_lev1 += category_names
        profiling_columns_lev2 += item_names
         

    # -- create Pandas dataframe from 2-levels column index and from unique sub/condition lists  
    profiling_multicol = list(zip(profiling_columns_lev1, profiling_columns_lev2))
    multicol_index = pd.MultiIndex.from_tuples(profiling_multicol, names=['profiling_categories', 'profiling_items'])
    profiling_df = pd.DataFrame(columns=multicol_index) #  create Pandas dataframe using 2-levels columns
    meta_unique_subs = metafile_df.subject_id.unique()
    meta_unique_conditions = metafile_df.condition.unique()
    # TODO: check if list_quest_per_condition is empty (then it changes profiling_df because only global questionnaires and no condition field are required anymore)
    sub_cond_list = list(itertools.product(meta_unique_subs,meta_unique_conditions)) # product between sub and condition
    profiling_df.loc[:,[('experimental','subject_id'),('experimental','condition')]] = sub_cond_list # fill profiling_df with this info

    
    # -- import all profiling data for each participant 
    for profiling_ix, profiling_row in profiling_df.iterrows():   
        
        # get subject_id, condition and path_profiling_dir
        subject_id = profiling_row.at[('experimental','subject_id')]
        condition =  profiling_row.at[('experimental','condition')]
        meta_mask_sub = metafile_df.subject_id == subject_id
        meta_ix = meta_mask_sub[meta_mask_sub].index.values[0]
        path_profiling_dir = pathlib.Path(metafile_df.at[meta_ix, 'profiling_dir'])
        
        # UI print
        print(f"Importing questionnaire data for participant #'{subject_id}' / condition '{condition}'...")
        
        # import experimental info (considered by default even if not in design config file)
        file_name = 'S' + str(subject_id) + '_answers_experimental_' + condition +'.csv'
        csv_path = path_profiling_dir.joinpath(file_name) 
        check_filePath_validity(csv_path) # check if file exists
        data_quest_df = pd.read_csv(csv_path, header=[0,1], encoding="iso8859_15") # load questionnaire data (participant answers)
        profiling_df.loc[profiling_ix,data_quest_df.columns] = data_quest_df.loc[0,data_quest_df.columns] # load all experimental info
         
        # -- import data from each questionnaire into profiling_df 
        for quest_ix, quest_name_i in enumerate(quest_list_design):
            config_path = quest_configFile_list[quest_ix]
            if quest_name_i in list_quest_per_condition:  # questionnaires in this list are presented again at each condition
                file_name = 'S' + str(subject_id) + '_answers_' + quest_name_i + '_' + condition +'.csv'
            elif quest_name_i in list_quest_global:    # questionnaires in this list are presented only once   
                file_name = 'S' + str(subject_id) + '_answers_' + quest_name_i + '.csv'    
            csv_path = path_profiling_dir.joinpath(file_name) 
            quest_i = Questionnaire(config_path, csv_path) # create Questionnaire object
            quest_answers_df, quest_answers_duration_df = quest_i.load_results() # load data from disk
            profiling_df.loc[profiling_ix,quest_answers_df.columns] = quest_answers_df.loc[0,:] # store it in profiling_df
            
    # -- sort by record time according to subject_id -> condition_order
    if ('experimental','condition_order') in profiling_df.columns: 
        profiling_df.sort_values([('experimental','subject_id'), ('experimental','condition_order')], ascending=[True, True], inplace=True)
    
    # -- create profiling_ix in profiling_df and also add it to metafile_df for future matching
    for profiling_ix, profiling_row in profiling_df.iterrows():
        profiling_df.at[profiling_ix,('experimental','profiling_index')] = profiling_ix
        metafile_mask = (metafile_df.subject_id == profiling_row.experimental.subject_id) & (metafile_df.condition == profiling_row.experimental.condition)  # select this subject_id AND this condition in metafile_df
        metafile_df.loc[metafile_mask, 'profiling_index'] = profiling_ix
        
    return profiling_df, metafile_df


def importProfilingDataSophie(metafile_df, filePath):
    """ import function for profiling data (adapted for Sophie's dataset)
    Here profiling data is already aggregated in one big .csv file
    Also returns metafile_df updated with indexes matched to profiling_df data
    

    Parameters
    ----------
    metafile_df: Pandas DataFrame 
           used here to fill a 'record_ix' column matching corresponding 
           indexes in profiling_df
   filePath : str
       string with full path of .csv file containing preprocessed profiling data.

    Returns
    -------
    profiling_df : Pandas dataframe
        Pandas dataframe with all data from specified .csv file.
    metafile_df: Pandas DataFrame 
           updated with a 'record_ix' column matching corresponding indexes 
           in profiling_df.
    
    *** Description of profiling data (gamer traits, flow and engagement scores)
    (player-) trait questionnaire ?Five Traits Model?
      * ref: Tondello et al. (2019) + paper submitted by Sophie / ?lise / Guillaume / ? Influence of
      Game Mechanics and Personality Traits on Flow and Engagement in Virtual Reality 
      * content: 25 items ?
      * output: 5 dimensions of gamer trait on a 0-100 scale
          1. aesthetic  
          2. challenge
          3. narrative
          4. objectives
          5. social
    
    Dispositional Flow Scale (DFS) 
      * ref: Jackson and Eklund (2002) 
      * content: 36-items on a likert scale from 1 to 5
      * output: 9 dimensions of the flow experience 
          1. challenge_skill_balance
          2. action_awareness
          3. clear_goals
          4. unambiguous_feedback
          5. concentration
          6. sense_of_control
          7. loss_self_consciousness
          8. time_transformation
          9. autotelic_experience
    
    Temple Presence Inventory (TPI) scale
      * ref: Lombard, Bolmarcich and Weinstein (2009)
      * content: 6 items on a likert scale from 1 to 7
      % output: 1 dimension of engagement
      
    *** Description import dataset
     Pre-imported .csv file 'traits_dfs_df.csv' 
    
    Here we assume demographics data already formated and preselected (bad
    recordings are already exluded) + psychometrics scores are already
    calculated. 
    
    
    """

    # assert inputs
    check_filePath_validity(filePath)

    # open file and import data
    profiling_df = pd.read_csv(filePath, header=0, encoding="iso8859_15")
    # reorder and rename some columns + multi-indexing on participants
    profiling_df = organizeProfilingDataSophie(profiling_df)

    # add one column in metafile_df with correct record_ix from corresponding profiling_df indexes 
    for profiling_ix, profiling_row in profiling_df.iterrows():
        metafile_mask = (metafile_df.subject_id == profiling_row.experimental.subject_id) & (metafile_df.condition == profiling_row.experimental.condition)  # select this subject_id AND this condition
        metafile_df.loc[metafile_mask, 'profiling_index'] = profiling_row.experimental.profiling_index
    
    # remove bad participants
    subjectId_list = [7,10,11,15,16,17,18,19,20,22,24,25,26,28,29,30,32,34,35,37,38,39,40,41,43,44,46,47,51,52,53,55,56]
    #   - S14 has no ObjectFormat file in scene/condition:  Goal/HouseObjectives
    #   - S49 has no ObjectFormat file in scene/condition:  Aesthetic/HouseAestheticSpace
    #   - S57 has no eyetracking file in scene/condition:  Aesthetic/postApo
    # subjectId_list = [7,10,11,14,15,16,17,18,19,20,22,24,25,26,28,29,30,32,34,35,37,38,39,40,41,43,44,46,47,49,51,52,53,55,56,57]
    data_sel = profiling_df.loc[:, ('experimental', 'subject_id')]  # select subject_id as a Series
    mask = data_sel.apply(lambda x: x not in subjectId_list)    # mask subject_ids not in subjectId_list
    dropTheseIndexes = list(mask[mask].index.values)            # get index values from mask
    profiling_df.drop(dropTheseIndexes, axis=0, inplace=True)   # drop these rows 
    
    
    return profiling_df, metafile_df


def organizeProfilingDataSophie(df_in):
    """ multi-indexing on participants + reorder and hierarchize columns 
    
    Initial file content (column index and data type in Sophie's dataset'):
    1. expe/date and time after submission of trait questionnaire (str)
    2. demographics/age (int)
    3. traits/maitrise VR (str - 4 vals)
    4. traits/freq jeux vidéo (str - 4 vals)
    5. demographics/gender (str - 3 vals)
    6. expe/site-location (str - 2 vals)
    7-31. traits/items -> 25 items (int - values: 1-9 ?...)
    32. expe/id participant (int)
    33-37. traits/scores -> 5 items (int) 
    38. expe/date and time after submission of state questionnaire
    39-80. states/items_DFS-TPI (questions DFS + TPI) -> 36+6=42 items (int - values: 1-9 ?...)
    81. expe/session-order (string - 3 values)
    82. expe/session-scenario (string - 3 values)
    83-91. states/scores-DFS -> 9 items (int - values: ?...)
    92. states/scores-engagement (float)
    
    Dataframe content after reorganizing: 
    - 'experimental' (7)
        - profiling_index (1)
    	- subject_id (1)
    	- condition (1)
        - condition_order (1)
        - passation_site (1)
        - FPT_completion' (1)
    	- DFS_completion' (1)
    - 'demographics' (2)
        - gender (1)
    	- age (1)
    - 'gaming' (2)
        - vr_mastery (1)
        - videogame_practice (1)
    - 'FPT' (Five Player Traits) (30)
        - items_FPT (25)
        - score_FPT (5)
    - 'DFS' (Dispositional Flow Scale) (45)
       - items_DFS (36)
       - score_DFS (9)
    - 'TPI' (Temple Presence Inventory) (7)
       - items_DFS (6)
       - score_DFS (1)
       """

    # --- insert profiling_index as a new column for later reuse
    df_out = df_in.reset_index(drop=False) # this adds an 'index' column to the dataframe

    # --- rename columns
    
    # rename metadata + experimental + demographics (column names)
    df_out = df_out.rename(columns={ 
        'index': 'profiling_index',
        'identifiant.participant': 'subject_id',
        'sur.quel.site.allez.vous.participer.a.l.experimentation..': 'passation_site',
        'qu.avez.vous.vu.lors.de.l.experience..': 'condition',
        'est.ce.votre.premiere..deuxieme.ou.troisieme.experience..': 'condition_order',
        'horodateur.x': 'FPT_completion',
        'horodateur.y': 'DFS_completion'})

    # rename demographics + gaming (column names)
    df_out = df_out.rename(columns={ 
        'quel.est.votre.genre..': 'gender',
        'quel.est.votre.age....exemple...21.': 'age',
        'quel.est.votre.niveau.de.maitrise.de.la.realite.virtuelle..': 'vr_mastery',
        'a.quelle.frequence.jouez.vous.aux.jeux.videos..': 'videogame_practice'})
    
    # rename FPT items and scores (column names)
    df_out = df_out.rename(columns={
        'j.aime.interagir.avec.d.autres.personnes.dans.un.jeu.': 'S1',
        'je.prefere.souvent.jouer.tout.seul...r.': 'S2',
        'je.n.aime.pas.jouer.avec.d.autres.personnes.joueurs...r.': 'S3',
        'j.aime.jouer.en.ligne.avec.d.autres.joueurs.': 'S4',
        'j.aime.les.jeux.ou.je.peux.jouer.en.equipe.ou.au.sein.d.une.guilde.': 'S5',
        'j.aime.les.jeux.qui.me.donnent.l.impression.d.etre.dans.un.lieu.different.': 'A1',
        'j.aime.les.jeux.avec.des.mondes.ou.univers.detailles.a.explorer.': 'A2',
        'je.me.sens.souvent.en.admiration.devant.les.paysages.ou.d.autres.elements.visuels.du.jeu.': 'A3',
        'j.aime.personnaliser.l.apparence.de.mon.personnage.': 'A4',
        'j.aime.passer.du.temps.a.explorer.le.monde.du.jeu.': 'A5',
        'j.aime.les.histoires.complexes.dans.un.jeu.': 'N1',
        'j.aime.les.jeux.ou.je.peux.me.plonger.dans.l.histoire': 'N2',
        'j.ai.l.habitude.de.sauter.l.histoire.ou.les.cinematiques.lorsque.je.joue...r.': 'N3',
        'j.ai.l.impression.que.l.histoire.m.empeche.souvent.de.vraiment.jouer...r.': 'N4',
        'l.histoire.n.est.pas.importante.quand.je.joue...r.': 'N5',
        'j.apprecie.les.defis.tres.difficiles.dans.les.jeux.': 'C1',
        'je.joue.souvent.aux.modes.de.jeux.les.plus.difficiles.': 'C2',
        'j.aime.quand.les.jeux.me.mettent.au.defi.': 'C3',
        'j.aime.quand.la.progression.dans.un.jeu.requiert.des.competences.': 'C4',
        'j.aime.quand.les.objectifs.sont.difficiles.a.atteindre.dans.les.jeux.': 'C5',
        'd.habitude.je.me.fiche.de.finir.tous.les.objectifs.optionnels.d.un.jeu...r.': 'G1',
        'je.me.sens.stresse.e..si.je.ne.termine.pas.tous.les.objectifs.d.un.jeu.': 'G2',
        'j.aime.finir.toutes.les.taches.et.objectifs.d.un.jeu.': 'G3',
        'j.aime.finir.les.jeux.a.100..': 'G4',
        'j.aime.finir.les.quetes.': 'G5',
        'social':'social',
        'aesthetic': 'aesthetic',
        'challenge': 'challenge',
        'narrative': 'narrative',
        'objectives': 'goal'})
    
    # rename DFS items and scores (column names)
    df_out = df_out.rename(columns={ 
        'c.etait.exigeant..mais.je.crois.que.mes.competences.m.ont.permis.de.relever.le.defi.': 'S01',
        'j.ai.fait.les.bonnes.actions.sans.essayer.d.y.penser.': 'A02',
        'je.savais.clairement.ce.que.je.voulais.faire.': 'G03',
        'pour.moi.c.etait.clair...je.savais.comment.les.actions.allaient.s.enchainer.': 'U04',
        'j.etais.entierement.concentre.e..sur.ce.que.je.faisais.': 'C05',
        'j.avais.l.impression.de.controler.ce.que.je.faisais.': 'O06',
        'je.n.etais.pas.preoccupe.e..par.ce.que.les.autres.auraient.pu.penser.de.moi.': 'L07',
        'ma.notion.du.temps.etait.differente..plus.rapide.ou.plus.lente.que.d.habitude..': 'T08',
        'j.ai.vraiment.aime.cette.experience.': 'X09',
        'mes.capacites.etaient.a.la.hauteur.du.defi.eleve.de.la.situation.': 'S10',
        'je.faisais.les.bonnes.actions.de.facon.automatique.': 'A11',
        'j.avais.une.vision.tres.claire.de.ce.que.je.voulais.faire.': 'G12',
        'j.etais.conscient.e..de.la.qualite.de.mes.actions.': 'U13',
        'j.arrivais.bien.a.rester.concentre.e..sur.ce.qui.se.passait.': 'C14',
        'je.sentais.que.je.pouvais.controler.ce.que.je.faisais.': 'O15',
        'je.n.etais.pas.preoccupe.e..par.le.jugement.des.autres.': 'L16',
        'le.temps.semblait.s.ecouler.de.facon.differente.que.d.habitude.': 'T17',
        'j.ai.aime.les.sensations.liees.a.la.maitrise.de.mes.competences.et.je.veux.encore.les.ressentir.': 'X18',
        'je.me.sentais.capable.de.faire.face.aux.exigences.elevees.de.la.situation.': 'S19',
        'j.agissais.de.maniere.fluide.sans.trop.y.penser.': 'A20',
        'je.voulais.atteindre.un.but.precis.': 'G21',
        'je.savais.ce.que.je.faisais.pendant.l.activite.': 'U22',
        'j.etais.completement.concentre.e..': 'C23',
        'j.avais.une.impression.de.controle.total.': 'O24',
        'je.n.etais.pas.preoccupe.e..par.mon.apparence.': 'L25',
        'j.avais.l.impression.que.le.temps.passait.rapidement.': 'T26',
        'cette.experience.m.a.donne.beaucoup.de.plaisir.': 'X27',
        'mes.competences.etaient.a.la.hauteur.de.ce.defi.eleve.': 'S28',
        'je.faisais.les.choses.spontanement.et.automatiquement..sans.avoir.a.reflechir.': 'A29',
        'mes.objectifs.etaient.clairement.definis.': 'G30',
        'j.etais.capable.d.evaluer.la.qualite.de.ma.performance.': 'U31',
        'j.etais.completement.concentre.e..sur.ce.que.je.faisais.': 'C32',
        'je.sentais.que.je.controlais.parfaitement.mes.actions.': 'O33',
        'je.n.etais.pas.preoccupe.e..par.ce.que.les.autres.pouvaient.penser.de.moi.': 'L34',
        'j.ai.perdu.ma.notion.habituelle.du.temps.': 'T35',
        'j.ai.trouve.cette.experience.extremement.valorisante.': 'X36',
        'challenge_skill_balance': 'challenge_skill_balance',
        'action_awareness': 'action_awareness',
        'clear_goals': 'clear_goals',
        'unambiguous_feedback': 'unambiguous_feedback',
        'concentration':'concentration',
        'sense_of_control':'sense_of_control',
        'loss_self_consciousness':'loss_self_consciousness',
        'time_transformation':'time_transformation',
        'autotelic_experience':'autotelic_experience'})
    
    # rename TPI items and scores (column names)
    df_out = df_out.rename(columns={ 
        'dans.quelle.mesure.vous.etes.vous.senti.mentalement.immerge.dans.l.experience..': 'E1',
        'dans.quelle.mesure.vous.etes.vous.senti.implique.dans.cette.experience..': 'E2',
        'dans.quelle.mesure.vos.sens.ont.ils.ete.sollicites..': 'E3',
        'dans.quelle.mesure.avez.vous.ressenti.une.sensation.de.realisme..': 'E4',
        'dans.quelle.mesure.cette.experience.a.t.elle.ete.relaxante.ou.excitante..': 'E5',
        'l.histoire.etait.elle.interessante..': 'E6',
        'engagement': 'engagement' })
    
    # --- reorder columns
    profiling_columns_lev2 = ['profiling_index','subject_id','condition','condition_order','passation_site','FPT_completion','DFS_completion',
                              'gender','age','vr_mastery','videogame_practice',
                              'S1', 'S2', 'S3', 'S4', 'S5', 'A1', 'A2', 'A3', 'A4', 'A5', 'N1', 'N2', 'N3', 'N4', 'N5', 'C1', 'C2', 'C3', 'C4', 'C5', 'G1', 'G2', 'G3', 'G4', 'G5', 'social', 'aesthetic', 'challenge', 'narrative', 'goal',
                              'S01', 'A02', 'G03', 'U04', 'C05', 'O06', 'L07', 'T08', 'X09', 'S10', 'A11', 'G12', 'U13', 'C14', 'O15', 'L16', 'T17', 'X18', 'S19', 'A20', 'G21', 'U22', 'C23', 'O24', 'L25', 'T26', 'X27', 'S28', 'A29', 'G30', 'U31', 'C32', 'O33', 'L34', 'T35', 'X36', 'challenge_skill_balance', 'action_awareness', 'clear_goals', 'unambiguous_feedback', 'concentration', 'sense_of_control', 'loss_self_consciousness', 'time_transformation', 'autotelic_experience',
                              'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'engagement']
    # list(set(profiling_df.columns[:]) - set(profiling_multicolumns)) # this checks that all columns are well considered 
    df_out = df_out.reindex(columns=profiling_columns_lev2)
    
    
    # create a multi-level index on columns (categories in column names)
    # create list of column names for categories (first level)
    profiling_columns_lev1 = ['experimental']*7 + ['demographics']*2 + ['gaming']*2 \
        +['FPT_items']*25 + ['FPT_scores']*5 + ['DFS_items']*36 + ['DFS_scores']*9+['TPI_items']*6 + ['TPI_scores']*1

    # make a first-level row multi-indexing for participants
    # /!\ -> this feature is finally abandonned because it adds unnecessary complexity:
    # - query on column values is as straightforward as direct indexing with subject_id / condition / scene
    # - multi-index on rows cannot be easily saved/loaded in conventional .csv files
    #multirow_index = pd.MultiIndex.from_tuples(list(zip(df_out['subject_id'], df_out['condition'])), names=['subject_id', 'condition'])
    #df_out.index = multirow_index
    #df_out.pop('subject_id')
    #df_out.pop('condition')
    #profiling_columns_lev2 = df_out.columns

    # make a multi-index column indexing for item categories 
    profiling_multicol = list(zip(profiling_columns_lev1, profiling_columns_lev2))
    multicol_index = pd.MultiIndex.from_tuples(profiling_multicol, names=['profiling_categories', 'profiling_items'])
    df_out.columns = multicol_index  # apply this column multi-indexing
    
    # convert values in condition_order from string to int
    target_str_list = ['Première','Deuxième','Troisième']
    target_int_list = [1, 2, 3]
    for target_str, target_int in zip(target_str_list,target_int_list):
        target_mask = df_out.loc[:,('experimental','condition_order')] == target_str
        df_out.loc[target_mask,('experimental','condition_order')] = target_int

    return df_out 


def importDataFile(data_filepath, format_filepath=None, data_type='objects'):
    """
    Import data and format (single files) into Pandas DataFrames
    
    Parameters
    ----------
    data_filepath : pathlib.Path or string
        single filepath of actionsData file
        may also be a list of filepaths, in which case only one dataset is imported (see resolve_multiple_files())
    format_filepath : pathlib.Path or string
        single filepath of actionsData file
        may also be a list of filepaths, in which case only one dataset is imported (see resolve_multiple_files())
        may also be empty when no file format is available
    data_type: string, default: 'objects'
        specifies data type among 'objects', 'actions', 'eyetracking'

    Returns
    -------
    data_df: Pandas DataFrame with imported data info
    format_df: Pandas DataFrame with imported format info
        (only if format_filepath is specified)
    
    Usage example
    -------
    # select a file from subject_id / condition / scene tp import actionsData and actionsFormat
    metafile_mask = (metafile_df.subject_id == 7) & (metafile_df.condition == 'Narrative') & (metafile_df.scene == 'HouseNarrative') # select this subject / condition / scene
    metafile_sel = metafile_df.loc[metafile_mask].squeeze() # get single row in metafile_df and convert to a pd Series 
    actionsData_df, actionsFormat_df = importDataFile(metafile_sel.file_actionsData, metafile_sel.file_actionsFormat, data_type='actions')
    """
    print(data_filepath)
    # -- assert inputs
    assert ((format_filepath is not None) or type(format_filepath) == list or isinstance(format_filepath,(str, pathlib.Path, pathlib.PurePath)), f"please check filepath: filename '{format_filepath}' must be empty, or it must be a string or pathlib.Path or pathlib.PurePath")
    assert (type(data_filepath) == list or isinstance(data_filepath,(str, pathlib.Path, pathlib.PurePath)), f"please check filepath: filename '{data_filepath}' must be a string or pathlib.Path or pathlib.PurePath")
    all_types = {'objects', 'actions', 'eyetracking'}
    assert (data_type in all_types, f"data_type must be specified among {all_types}")
    
    
    # -- check filepaths are valid
        # + handle specific case when multiple files are listed... 
        # (this happens e.g., when a participant visits a VR scene multiple times, 
        # in which case multiple datasets are stored on disk for the same condition / scene / datatype) 
        # -> arbitrary solving choice: select only the oldest file
    data_filepath = resolve_multiple_files(data_filepath)  # handle specific case when multiple files are listed...
    check_filePath_validity(data_filepath) # check filepaths are valid (files exist on disk)      
    if format_filepath: # in case this filepath was provided
        format_filepath = resolve_multiple_files(format_filepath) 
        check_filePath_validity(format_filepath) # check filepaths are valid (files exist on disk)        

    # -- load object data and format from disk depending on data_type
    data_df = pd.read_csv(data_filepath, header=0)
    format_df = None # None by default, except if this was provided
    if format_filepath:
        format_df = pd.read_csv(format_filepath, header=0)
    
    # -- make specific arrangements depending on data_type
    if data_type == 'objects': 
        # convert timestamp in unix (100ns) format to proper Pandas datetime format (super useful for later manipulation)
        unix_timestamps = 100*pd.to_numeric(data_df['ticksSince1970 (100ns)']) # convert string to int64 with 'ns' unit
        data_df.insert(1,'pd_datetime', pd.to_datetime(unix_timestamps, unit='ns', infer_datetime_format=True, origin='unix'))
        data_df.drop('ticksSince1970 (100ns)', axis='columns', inplace=True)
        # for convenience, duplicate the only very useful column of objectsFormat_df 'trackedData' with name of tracked object
        # -> finally NOP: this more than DOUBLES memory usage! ...
        # objectsData_df['trackedData'] = None
        # for object_ix, object_row in objectsFormat_df.iterrows():
        #    mask = (objectsData_df.actionId == object_ix)
        #    objectsData_df.loc[mask,'trackedData'] = object_row.trackedData    
    elif data_type == 'actions': 
        # convert timestamp in unix (100ns) format to proper Pandas datetime format (super useful for later manipulation)
        unix_timestamps = 100*pd.to_numeric(data_df['ticksSince1970 (100ns)']) # convert string to int64 with 'ns' unit
        data_df.insert(1,'pd_datetime', pd.to_datetime(unix_timestamps, unit='ns', infer_datetime_format=True, origin='unix'))
        data_df.drop('ticksSince1970 (100ns)', axis='columns', inplace=True)
    elif data_type == 'events': 
        print("Warning: event file format is deprecated and will be removed in a future release")
    elif data_type == 'eyetracking':    
        # only keep these columns
        data_df = data_df[["time(100ns)", "time_stamp(ms)", 
        'gaze_origin_L.x(mm)', 'gaze_origin_L.y(mm)', 'gaze_origin_L.z(mm)', 
        'gaze_origin_R.x(mm)', 'gaze_origin_R.y(mm)', 'gaze_origin_R.z(mm)', 
        'gaze_origin_C.x(mm)', 'gaze_origin_C.y(mm)', 'gaze_origin_C.z(mm)', 
        'gaze_direct_L.x', 'gaze_direct_L.y', 'gaze_direct_L.z', 
        'gaze_direct_R.x', 'gaze_direct_R.y', 'gaze_direct_R.z', 
        'gaze_direct_C.x', 'gaze_direct_C.y', 'gaze_direct_C.z', 
        'pupil_diameter_L(mm)', 'pupil_diameter_R(mm)', 'pupil_diameter_C(mm)']]
        data_df = data_df.rename(columns={'time(100ns)': 'ticksSince1970'}) # CHANGE THIS IF FIELD NAME EVOLVES FOR ticksSince1970
        
        # convert timestamp in unix (100ns) format to proper Pandas datetime format (super useful for later manipulation)
        unix_timestamps = 100*pd.to_numeric(data_df['ticksSince1970']) # convert string to int64 with 'ns' unit
        data_df.insert(1,'pd_datetime', pd.to_datetime(unix_timestamps, unit='ns', infer_datetime_format=True, origin='unix'))
        data_df.drop('ticksSince1970', axis='columns', inplace=True)
    
    if format_filepath:
        return data_df, format_df
    else:
        return data_df



def get_indexes_from_sub_list(df, sub_list):
    """
    small utils function returning indexes of rows in 'df' having 'subject_id' 
    values in sub_list.

    Parameters
    ----------
    df : Pandas DataFrame
        must contain a "subject_id" column.
    sub_list : list(int)
        list of subject_id to get indexes from in df.

    Returns
    -------
    index_select : list(int)
        indexes of rows in df having subject_id values in sub_list.
    """
    
    df_mask = df.apply(lambda x : x['subject_id'] in sub_list, axis=1)
    index_select = df[df_mask].index.values
    
    return index_select


def resolve_multiple_files(file_list):
    """
    simple solver in case there are multiple file listed for the same condition / scene
    allows to change the rule once for all here
    
    *** CURRENT RULE : take the oldest file (first in alphabetical order) ***

    Parameters
    ----------
    file_list : single str or pathlib.Path (no list) or list(str or pathlib.Path)
        in case single path -> passthrough (nothing to solve)

    Returns
    -------
    file : pathlib.Path
        file from solver
    """
    # input must be a valid pathlib.Path or could be str
    assert isinstance(file_list,str) or isinstance(file_list, (pathlib.Path, pathlib.PurePath)) or \
           (isinstance(file_list[0],(str, pathlib.Path, pathlib.PurePath))), \
        "filename must be a string or pathlib.Path or pathlib.PurePath"

    if type(file_list) == list:
        file = pathlib.Path(sorted(file_list)[0])  # solve it and hard cast to pathlib.Path just in case a string was provided
    else:
        file = file_list  # passthrough
    return file


def load_thisObjectData(metafile_df, objectId=0, subjectId='all', condition='all', scene='all', verbose=True):
    """
    this loads and merge one specific object's data from several datasets / conditions

    Parameters
    ----------
    metafile_df: Pandas dataframe 
        contains all metadata structure (subject_id, condition, scene and all file paths) 
    objectId: int, optional
        selector for objectId (same as actionId) to get data from. The default is 0 (VR camera). 
    subjectId : int or list(int), or str 'all', optional
        selector for subjectId to get data from. The default is 'all'.
    condition : str or list(str), optional
        selector for condition to get data from. The default is 'all'.
    scene : str or list(str), optional
        selector for scene to get data from. The default is 'all'.
        /!\ some scenes have the same name across conditions/scenarios (e.g., 'tutorial')
        therefore if multiple condition are selected, all corresponding scenes 
        will be loaded (e.g., aestethic/tutorial AND narrative/tutorial)
    verbose: bool, optional
        True prints loading info in console

    Returns
    -------
    objectsData_merge_df : Pandas dataframe 
        same cols as a single objectsData_df + record_index (integer that allows to get back to subject_id + condition + scene)
    """
    
    # -- assert inputs and and make sure input selections for subjectId, condition and scenes all exist in metadata    
    assert type(objectId) is int, "'objectId' type must be integer"
    subjectId, condition, scene = resolve_index_metadata(metafile_df, subjectId, condition, scene)
    
    # -- loop on all (subject_id, condition, scene)
    super_dataList = []
    mem_usage = 0
    file_count = 0

    for count_sub, sub_sel in enumerate(subjectId):
        for count_condition, condition_sel in enumerate(condition):

            # special processing for scenes, whose structure is specific to each condition
            metafile_mask = (metafile_df.subject_id == sub_sel) & (metafile_df.condition == condition_sel) # select this subject AND condition
            possible_scenes = metafile_df.loc[metafile_mask].scene.to_list()

            if scene == 'all':
                curr_scenes = possible_scenes
            else:
                curr_scenes = scene
                assert set(curr_scenes) <= set(
                    possible_scenes), f"requested scenes is absent from metafile_df: {set(curr_scenes) - set(possible_scenes)}"

            # import all object data for this (subject_id, condition, scene)
            for scene_sel in curr_scenes:

                # load objectsData from file
                metafile_mask = (metafile_df.subject_id == sub_sel) & (metafile_df.condition == condition_sel) & (metafile_df.scene == scene_sel) # select this subject / condition / scene
                metafile_sel = metafile_df.loc[metafile_mask].squeeze() # get single row in metafile_df and convert to a pd Series 
                objectData_sel_df, format_df = importDataFile(metafile_sel.file_objectsData, metafile_sel.file_objectsFormat, data_type='objects')

                # assert this objectId is contained in data and select it
                assert objectId in list(objectData_sel_df.objectId.unique()), \
                    "target object '" + str(objectId) + \
                    "' is not listed in current dataFormat for subjectId: " + str(sub_sel) + \
                    " / condition: " + condition_sel + " / scene: " + scene_sel
                thisObject_sel = objectData_sel_df.loc[objectData_sel_df.objectId == objectId].copy()

                # add field 'record_index' to later identify to which (subject_id, condition, scene) belongs this raw data
                thisObject_sel['record_index'] = metafile_df.loc[metafile_mask].squeeze().record_index
                        
                # and add to list !
                # ... in comparison, this is NOT efficient: direct pd.concat inside a for-loop (leads to quadratic copying)
                super_dataList.append(thisObject_sel)  # append in a list without copy (optimal)

                # UI stuff
                if verbose:
                    mem_usage += get_MemoryUsage(thisObject_sel)
                    file_count += 1
                    print('.', end='')

            # UI print
            if verbose:
                percent_completion = round(
                    100 * (count_sub * len(condition) + count_condition) / (len(subjectId) * len(condition)), 0)
                if percent_completion % 10 == 0:
                    print(f'\nloading and aggregating data... {int(percent_completion)}% (RAM: {round(mem_usage, 1)} MB)', end='')

    # -- concatenate in big merger dataframe
    objectsData_merge_df = pd.concat(super_dataList, axis=0)

    # UI print
    if verbose:
        mem_usage = get_MemoryUsage(objectsData_merge_df)
        print(f'\nloading and aggregating data... DONE (RAM: {mem_usage} MB)')
        print(f'\ndata aggregated across x{len(subjectId)} participants and x{len(condition)} conditions (x{file_count} files)')

    return objectsData_merge_df


def resolve_index_metadata(metafile_df, subjectId, condition, scene):
    """
    utils function (internal) used to resolve function parameters when indexing metafile_df
    -> converts 'all' to actual indexes and make sure input selections for 
    subjectId, condition and scenes all exist in metadata    
    
    Parameters
    ----------
    metafile_df: Pandas dataframe 
        contains all metadata structure (subject_id, condition, scene and all file paths) 
    subjectId : int or list(int), or str 'all', optional
        selector for subjectId to get data from. The default is 'all'.
    condition : str or list(str), optional
        selector for condition to get data from. The default is 'all'.
    scene : str or list(str), optional
        selector for scene to get data from. The default is 'all'.
        /!\ some scenes have the same name across condition (e.g., 'tutorial')
        therefore if multiple condition are selected, all corresponding scenes 
        will be loaded (e.g., aestethic/tutorial AND narrative/tutorial)
    
    Returns
    -------
    subjectId, condition, scene: -> all valid compared to data in metafile_df
    
    """    
        
    # make sure inputs have the right format
    assert type(metafile_df) is pd.core.frame.DataFrame, "metafile_df must be a valid Pandas dataframe"
    assert subjectId == 'all' or (type(subjectId) is int) or (type(subjectId) is list and type(
        subjectId[0]) is int), "'subjectId' selector must be a int or list(int), or 'all'"
    assert (type(condition) is str) or (
                type(condition) is list and type(condition[0]) is str), "'condition' selector must be a str or list(str)"
    assert (type(scene) is str) or (
                type(scene) is list and type(scene[0]) is str), "'scene' selector must be a str or list(str)"

    # get unique index values from metafile
    meta_unique_subs = metafile_df.subject_id.unique()
    meta_unique_condition = metafile_df.condition.unique()
    meta_unique_scene = metafile_df.scene.unique()

    # fill inputs when 'all' is specified
    if subjectId == 'all': subjectId = meta_unique_subs
    if condition == 'all': condition = meta_unique_condition
    # (special case for scenes, because they are not necessarily the same across conditions)

    # makes lists when entries are single valued 
    if type(subjectId) is int: subjectId = [subjectId]
    if type(condition) is str: condition = [condition]
    if scene != 'all' and type(scene) is str: scene = [scene]

    # assert input selections for subjectId, condition and scenes all exist in metadata    
    assert set(subjectId) <= set(meta_unique_subs), \
        f"info from these requested participants is absent from metafile_df: {set(subjectId) - set(meta_unique_subs)}"
    assert set(condition) <= set(meta_unique_condition), \
        f"info from these requested condition is absent from metafile_df: {set(condition) - set(meta_unique_condition)}"
    assert scene == 'all' or set(scene) <= set(meta_unique_scene), \
        f"info from these requested scenes is absent from metafile_df: {set(scene) - set(meta_unique_scene)}"
    
    return subjectId, condition, scene




def selectProfilingData(profiling_df, filter_key=None, filter_to_apply=None):
    """ Dataset selection based on condition to satisfy in profiling data
    THIS FUNCTION WILL BE NEEDED WHEN AUTOMATIZING REQUESTS WITH GUI (???)
    
    Parameters
    ----------
    profiling_df: Pandas dataframe with formated profiling data
    filter_key: (str) must correspond to an actual column name of dataFrame profiling_df
    filter_to_apply: format depends on type of data to filter in 'filtered_col'
    if 'filtered_col' data is categorical (e.g., gender, recording-site, etc.) 
        -> 'filter_to_apply' must be a string or a sequence of strings whose values 
        are represented in the column 'filter_key'
        e.g.: 'Femme', 'INSA Lyon'
    if 'filtered_col' data is numerical (e.g., states_scores.engagement) 
        -> then condition 'filter_to_apply' should be given as a tuple or list of tupples 
        with format ['boolean logical operator', value OR 'summary function'] 
        summary function can be: 'mean', 'median', 'quantile(0.25)'
        e.g., ('>', 10)  # -> filter values above 10
        e.g., ('<=', 'quantile(0.25)')  # -> filter values below or equal to 25th quantile

    Returns
    -------
    returns a dataFrame and indexes of datasets that satisfy 'filter_to_apply' in column 'filter_key'
    filtered_indexes: list of unique dataset indexes 
    profiling_df_groupedBy: Pandas groupby object that contains information about the groups.
    profiling_pivot_df: Pandas DataFrame spreadsheet-style pivot table (if two columns are asked in selection array)
    
    Example:
    ------    
    filtered_indexes = selectProfilingData(profiling_df, index_filter = ('demographics.gender','Homme' ), data_selection = None):
        
        'gender' -> [male, female, other]
        'age_bin' -> create 2years bins for age (18-20 / 20-22 / 22- 24 / >24)
    """


def maskUnknownObjects(objectsData_df, objectsFormat_df):
    """
    return a list and mask of objects in objectsData_df that are not listed in objectsFormat_df

    Parameters
    ----------
    objectsData_df : pd DataFrame
        contains objects data 
    objectsFormat_df : (pd DataFrame, optional)
        contains objects description

    Returns
    -------
    unknown_objectId_list : list of integers
        list with objectId of unknown objects
    mask_objectsData_df : TYPE
        mask on objectsData_df rows (False for rows with actionID that corresponds to unknown objects)
    """
    objectId_fromData = sorted(objectsData_df['objectId'].unique())
    objectId_fromFormat = objectsFormat_df.index.to_list()
    unknown_objectId_list = list(set(objectId_fromData) - set(objectId_fromFormat))
    if len(unknown_objectId_list) == 0:
        mask_objectsData_df = None
    else:
        mask_objectsData_df = objectsData_df.objectId.isin(unknown_objectId_list)
    return unknown_objectId_list, mask_objectsData_df


def getObject_indexes(objectsData_df, object_id=0, objectsFormat_df=None):
    """
    return all indexes of that specific object in objectsData_df

    Parameters
    ----------
    objectsData_df : pd DataFrame
        contains object data to be indexed
    object_id : int or str (default: 0 is VRcamera)
        if int: object_id is directly actionID in objectsData_df (object id)
        if str: object_id is trackedData in objectsFormat_df (object name)
    objectsFormat_df : (pd DataFrame, optional)
        must be provided if object_id is given as a string

    Returns
    -------
    mask_out: bool pd Series
        True for objectsData_df rows corresponding to object_id
    """
    assert type(object_id) is str or type(object_id) is int or len(
        object_id) > 0, "type of object_id must be integer or string"
    if type(object_id) is int:
        assert type(
            objectsFormat_df) is pd.core.frame.DataFrame, "when object_id is provided as integer, objectsData_df must be provided too"

        # get integer value (index) of object_id given as a string
    if type(object_id) is str:
        object_name = object_id
        object_id = getObject_IdFromName(objectsFormat_df, object_name)

        # create mask
    mask_out = objectsData_df.objectId == object_id
    return mask_out


def getObject_NameFromId(objectsFormat_df, object_id=0):
    """
    get object object name (str) from object id (int) 

    Parameters
    ----------
    objectsFormat_df : pd DataFrame 
        dataframe with metadata about object (especially here index and trackData columns) 
    object_id : int (or list of int), default: 0 is VRcamera)
        object index(es) in objectsFormat_df, SAME as actionID in objectsData_df

    Returns
    -------
    object_name : str (or list of str), default is player's camera': '/Player/SteamVRObjects/VRCamera'
        object_name corresponds to str in column 'trackedData' in objectsFormat_df 
    """

    def getCurrName(objectsFormat_df, object_id):
        assert type(object_id) is int, "object_name must be integer"
        assert object_id in objectsFormat_df.index.to_list(), "target object '" + str(
            object_id) + "' is not listed in current dataFormat"
        return objectsFormat_df.loc[object_id, 'trackedData']

    if type(object_id) is list:
        object_name = []
        for object_id_i in object_id:
            object_name.append(getCurrName(objectsFormat_df, object_id_i))
    else:
        object_name = getCurrName(objectsFormat_df, object_id)

    return object_name


def getObject_IdFromName(objectsFormat_df, object_name='/Player/SteamVRObjects/VRCamera'):
    """
    get object id (int) from object name (str)

    Parameters
    ----------
    objectsFormat_df : pd DataFrame 
        dataframe with metadata abouy object (especially here index and trackData columns) 
    object_name : str (default is player's camera': '/Player/SteamVRObjects/VRCamera')
        object_name corresponds to str in column 'trackedData' in objectsFormat_df 

    Returns
    -------
    object_id: int
        object_id (object index in objectsFormat_df, SAME as actionID in objectsData_df
    """
    assert isinstance(object_name, str) and len(object_name) > 0, "object_name must be a valid string"
    assert object_name in objectsFormat_df.trackedData.to_string(), "target object '" + object_name + "' is not listed in current dataFormat"
    maskFormat = objectsFormat_df.trackedData == object_name
    object_id = objectsFormat_df.loc[maskFormat].index.tolist()[0]  # get integer value of object_id
    return object_id




def get_MemoryUsage(df_in, verbose=False):
    """
    get memory usage (RAM) of a Pandas DataFrames

    Parameters
    ----------
    df_in : pd DataFrame 
    verbose: bool (default: False), if True print memory usage

    Returns
    -------
    mem_usage: float with memory allocation in MB
    """
    mem_usage = round(sum(df_in.memory_usage(deep=True)) / (2 ** 20),
                      1)  # or just get the memory usage  or whole dataFrame (in MB)
    if verbose:
        print(f"Memory usage: {mem_usage} MB")

    return mem_usage


def check_filePath_validity(filePath):
    """
    Small internal utils function to check if filepath is valid and a file exists

    Parameters
    ----------
    filePath : str or pathlib.Path  
        directory path or filepath to check.

    Returns
    -------
    None.

    """
    assert isinstance(filePath,(str, pathlib.Path, pathlib.PurePath)), \
        "filename must be a string or pathlib.Path or pathlib.PurePath"
    if not isinstance(filePath,pathlib.Path) :
        filePath = pathlib.Path(filePath)

    dirName = filePath.parent
    fileName = filePath.name
    
    assert dirName.is_dir(), f"specified directory '{dirName}' does not exist"
    assert filePath.is_file(), f"specified file '{fileName}' does not exist in directory '{dirName}'"



def back_to_raw_data_questionnaires_Sophie(metafile_df, profiling_df, design_configFile):
    """
    Internal utils function, used to back-generate raw questionnaire data 
    with .csv files from aggregated and preprocessed data in profiling_df.

    Parameters
    ----------
    metafile_df: Pandas DataFrame 
           contains participants dir paths on disk
    profiling_df : Pandas dataframe
        dataframe where questionnaire data is aggregated across all participants
    design_configFile: str
        path of the .yaml config file with experiment structure 
        
    Returns
    -------
    None.
    
    
    Notes on file structure
    -------
   	- DATA: one directory containing all participant data for this experiment (questionnaires + behavior + eyetracking + physio)
   		- subject dir: one directory per participant with name "S"+(index participant)
   			- profiling dir: inside each subject dir, one dir containing questionnaire answers /!\ -> these files are parsed in function importProfilingData()
   				- one .csv file containing participant answer for each questionnaire
   				- name of these files are: "S"+(index participant) + '_' + condition (if quest is presented for each condition) + '_' + title_questionnaire (defined in pyquest_title-quest.yaml)

    """
    
    # /!\ IF ENABLED: OVERWRITE ALL EXISTING FILES IN 'profiling' FOLDERS
    overwrite_it_baby = True    # are you sure ?.. 
    
    # -- assert inputs
    assert (type(metafile_df) is pd.core.frame.DataFrame), "input 'metafile_df' must be a valid Pandas dataframe"
    assert (type(profiling_df) is pd.core.frame.DataFrame), "input 'profiling_df' must be a valid Pandas dataframe"
    assert isinstance(design_configFile,(str, pathlib.Path, pathlib.PurePath)), "design_configFile must be a string or pathlib.Path or pathlib.PurePath"
    
    # -- fill unknown data with fake (well, approximate) data (these are columns that should be recorded in future experiments)
    profiling_df.loc[:,('experimental','demographics_completion')] = profiling_df.loc[:,('experimental','FPT_completion')]
    profiling_df.loc[:,('experimental','gaming_completion')] = profiling_df.loc[:,('experimental','FPT_completion')]
    profiling_df.loc[:,('experimental','TPI_completion')] = profiling_df.loc[:,('experimental','DFS_completion')]
    
    # -- load config file, get questionnaire name and load questionnaire configs
    configPath = pathlib.Path(design_configFile).parents[0]
    config_design = loadConfigExperiment(design_configFile)   
    
    # -- check that a config file exists for each questionnaire listed in config_design
    quest_configFiles_onDisk = sorted(configPath.glob('pyquest_*.yaml')) # list of questionnaire config files on disk (PATHS)
    quest_list_onDisk = list(map(lambda x: x.stem.split("_",1)[1], quest_configFiles_onDisk)) # list of questionnaire config files on disk (NAMES)
    quest_list_design = config_design['DESIGN']['questionnaires'] # list of questionnaires utilized for this experiment (as described in design_configFile)
    assert set(quest_list_design) <= set(quest_list_onDisk), f"config files for these requested questionnaires are missing: {set(quest_list_design)-set(quest_list_onDisk)}"
    
    # -- create (OR RESET) profiling folder for each participant in metafile_df
    meta_unique_subs = metafile_df.subject_id.unique()
    for sub_i in meta_unique_subs:
        metafile_mask = (metafile_df.subject_id == sub_i)
        first_meta_ix = metafile_mask[metafile_mask].index.values[0]    # index of the first occurence of this subject_id in metafile_df
        meta_row = metafile_df.iloc[first_meta_ix]
        breakpoint()
        path_profiling = meta_row.data_dir.parent.joinpath('profiling')      
        dir_exists = path_profiling.exists()
        if not dir_exists :
            path_profiling.mkdir()
        if dir_exists and overwrite_it_baby:
            files = sorted(path_profiling.glob('*.csv'))
            [x.unlink() for x in files]     # erase all .csv files...
        metafile_df.loc[metafile_mask,'profiling_dir'] = path_profiling # update metafile_df 
        
    # --- create all files containing participant profiling data (answers to questionnaire)
    for profiling_ix, profiling_row in profiling_df.iterrows():
        subject_id = profiling_row.at[('experimental','subject_id')]
        condition = profiling_row.at['experimental','condition']
        print('generate back raw questionnaire data for participant #' + str(subject_id) + ' / ' + condition )

        # get profiling dir using profiling_index match between profiling_df and metafile_df
        prof_ix = profiling_row.loc[('experimental','profiling_index')]
        
        metafile_mask = (metafile_df.profiling_index == prof_ix)
        first_meta_ix = metafile_mask[metafile_mask].index.values[0]    # index of the first occurence of this profiling_index in metafile_df
        path_profiling = pathlib.PurePath(metafile_df.at[first_meta_ix,'profiling_dir'])
    
        # -- create file containing metadata and experimental items
        # experimental datafiles should typically contain the following fields:
        # 'subject_id', 'passation_site', 'condition', 'condition_order' + one completion timestamp per questionnaire    
        completion_strings = [s + '_completion' for s in quest_list_design]
        item_names = ['subject_id','condition','condition_order','passation_site'] + completion_strings
        category_names = ['experimental']*1 + ['experimental']*(3+len(completion_strings))
        profiling_multicol = list(zip(category_names, item_names))
        multicol_index = pd.MultiIndex.from_tuples(profiling_multicol, names=['profiling_categories', 'profiling_items'])
        data_df = pd.DataFrame(columns=multicol_index)
        data_df.loc[0] = profiling_row.loc[profiling_multicol]
        file_name = 'S' + str(subject_id) + '_answers_experimental_' + condition + '.csv'
        csv_path = path_profiling.joinpath(file_name)
        data_df.to_csv(path_or_buf=csv_path, index= False, encoding="iso8859_15") 
        # data_df_read = pd.read_csv(csv_path, header=[0,1], encoding="iso8859_15") # (check)
    
        # -- create files containing participant answers and scores to each questionnaire
        for quest_i in quest_list_design:
            
            # load questionnaire config 
            quest_configFile = str(pathlib.Path(configPath,'pyquest_'+quest_i+'.yaml'))
            config_quest = loadConfigQuestionnaire(quest_configFile)          
            
            # get names of items
            question_items_names = config_quest['ITEMS']['item_id'].to_list()
            if not config_quest['SCORES'].empty:
                question_category_names = [quest_i+'_items']*len(question_items_names)
                score_items_names = config_quest['SCORES']['score_id'].to_list()
                score_category_names = [quest_i+'_scores']*len(score_items_names)
            else:
                question_category_names = [quest_i]*len(question_items_names)
                score_items_names = []
                score_category_names = []
            category_names = question_category_names + score_category_names # column name level-1 (names categories) to write on file
            item_names = question_items_names + score_items_names # column name level-2 (names items) to write on file
            
            # make a dataFrame with questionnaire answers 
            profiling_multicol = list(zip(category_names, item_names))
            multicol_index = pd.MultiIndex.from_tuples(profiling_multicol, names=['profiling_categories', 'profiling_items'])
            data_df = pd.DataFrame(columns=multicol_index)
            data_df.loc[0] = profiling_row.loc[profiling_multicol]
            if quest_i in ['DFS', 'TPI']: # these two questionnaires are presented again at each condition
                file_name = 'S' + str(subject_id) + '_answers_' + quest_i + '_' + condition +'.csv'
            else:   # the rest is presented only once (here brutal overwrite °-°)
                file_name = 'S' + str(subject_id) + '_answers_' + quest_i + '.csv' 
            csv_path = path_profiling.joinpath(file_name)
            data_df.to_csv(path_or_buf=csv_path, index= False, encoding="iso8859_15") 
            # data_df_read = pd.read_csv(csv_path, header=[0,1], encoding="iso8859_15") # (check)
    
    
def balanced_latin_squares(n):
    """
    Small utils function that produces Balanced Latin Squares randomization
    implementation adapted from Paul Grau:
    https://medium.com/@graycoding/balanced-latin-squares-in-python-2c3aa6ec95b9

    Parameters
    ----------
    n : integer 
    size of latin square

    Returns
    -------
    l : list
        contains all permutations for this Balanced Latin Squares.

    """
    l = [[((j//2+1 if j%2 else n-j//2) + i) % n for j in range(n)] for i in range(n)]
    if n % 2:  # Repeat reversed for odd n
        l += [seq[::-1] for seq in l]
    return l


