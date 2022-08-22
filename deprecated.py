#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 20:18:22 2022

@author: jonas
"""


# TODO: export profiling_df to a .h5 (hdf5) file
#hdf_path = '/Users/jonas/Documents/PORTRAIT_local/output-analysis/profiling.h5'
#profiling_df.to_hdf(hdf_path,'table', append=True)
# read it: 
#profiling_df_read = pd.read_hdf(hdf_path)



# --- remove a few bad subjects
params["subject_id_list"] =  profiling_df.loc[:,('metadata', 'subject_id')].unique()
remove_ids = [14,49,57] # these three subjects lack some datasets:
#   - S14 has no ObjectFormat file in scene/condition:  Goal/HouseObjectives
#   - S49 has no ObjectFormat file in scene/condition:  Aesthetic/HouseAestheticSpace
#   - S57 has no eyetracking file in scene/condition:  Aesthetic/postApo
params["subject_id_list"] = list(filter(lambda x: x not in remove_ids, params["subject_id_list"])) # remove these sub_ids from list
metafile_df.drop(dropTheseIndexes, axis=0, inplace=True)


 
# --- old ways of accessing index data in multilevel row indexes
params["subject_id_list"] =  profiling_df.index.to_frame()['subject_id'].unique()


# --- multirow indexing in metafile_df
def createMetaFile(basePath, subject_id_list, profiling_df=None):
    """create a metafile_df dataFrame with all metadata structure (subject_id, scenario, scene and all file paths) 
    (adapted for initial Sophie's dataset)
    -- Inputs
      basePath (str): string with full dir path containing folders with datasets
      subject_id (int): list of subject id to import data from 
      profiling_df (optional: pd DataFrame): allows to fill column with record_ix from corresponding profiling_df indexes
    
    -- Outputs
    metafile_df: Pandas dataframe with all metadata structure (subject_id, scenario, scene and all file paths) 
    
    base directory is organized each following file for each VR scene in the session:
        - subject dir: one directory per participant with name "S"+(index participant)
        - session dir: inside each subject dir, one dir for each session with name "Esthetique", "Narratif", "Objectif"    
        - following datasets (csv files):
        -   "eventsData": data about contextual VR events (removing headset, loading scene, etc.)
        -   "eventsFormat": event metadata (event id + event name)
        -   "HTCViveProEyeData": HTC Vive Pro Eye data in original format
        -   "objectsData": VR objects data (one line per frame for each moving tracked object)
        -   "objectsFormat": XR-Echo info for VR replay (object type, trackedData, position, etc.)
    """

    # assert inputs
    assert type(subject_id_list) is list and type(
        subject_id_list[0]) is np.int64, "subject_id must be a list of int (np.int64)"
    assert len(subject_id_list) > 0, "subject_id must have one or more subject id to import data from"
    assert type(basePath) is str and len(basePath) > 0, "base dir path should be a valid string"

    # look for participant directories 
    # dirNames_from_id = [i + j for i, j in zip(['S']*len(subject_id_list), list(map(str,subject_id_list)))]
    dirNames_onDisk = sorted(glob2.glob(basePath + os.sep + 'S*' + os.sep))
    try:
        subject_id_onDisk = list(map(lambda x: np.int32(x[-3:-1]), dirNames_onDisk))
    except:
        print("something funny with name of participant directories within: " + basePath)
    nonExistantIds = set(subject_id_list) - set(subject_id_onDisk)
    assert set(subject_id_list) <= set(
        subject_id_onDisk), f"these requested subjects ID are not on disk: '{nonExistantIds}'"

    # create a metafile_df dataFrame with all metadata structure (subject_id, scenario, scene and all file paths)
    # -- scene organization (ordered) for each scenario: 
    #  Aesthetic: Tutorial, HouseAestheticBase (/!\ x2), HouseAestheticSpace, PostApo
    #  Narrative: Tutorial, HouseNarrative
    #  Goals: Tutorial, HouseObjectives
    index_arrays_lev0 = subject_id_list
    index_arrays_lev1 = ['Aesthetic', 'Aesthetic', 'Aesthetic', 'Aesthetic', 'Narrative', 'Narrative', 'Goals', 'Goals']
    index_arrays_lev2 = ['Tutorial', 'HouseAestheticBase', 'HouseAestheticSpace', 'PostApo', 'Tutorial',
                         'HouseNarrative', 'Tutorial', 'HouseObjectives']
    index_arrays_lev01 = list(itertools.product(index_arrays_lev0, index_arrays_lev1))
    index_arrays_lev02 = list(itertools.product(index_arrays_lev0, index_arrays_lev2))
    index_arrays_final = [(a[0], a[1], b[1]) for a, b in zip(index_arrays_lev01, index_arrays_lev02)]
    fileName_cols = ['file_eventsData', 'file_eventsFormat', 'file_HTCViveProEyeData', 'file_metadata',
                     'file_objectsData', 'file_objectsFormat']
    fileName_scenes_onDisk = ['eventsData', 'eventsFormat', 'HTCViveProEyeData', 'metadata', 'objectsData',
                              'objectsFormat']
    metafile_df = pd.DataFrame(columns=['profiling_index', 'record_index', 'data_dir'] + fileName_cols,
                               index=pd.MultiIndex.from_tuples(index_arrays_final,
                                                               names=('subject_id', 'scenario', 'scene')))
    metafile_df.record_index = list(range(
        len(metafile_df)))  # here we create unique integer record_index for each tuple(sub_id,scenario,scene) respecting record order in time
    metafile_df.sort_index(inplace=True)  # sort data for efficient indexing (otherwise raises PerformanceWarning)

    # check each participant data 
    expConditions_profiling = ['Aesthetic', 'Narrative', 'Goals']  # names in engligh within Profiling data
    expConditions_onDisk = ['Esthetique', 'Narratif', 'Objectif']  # names in French in files from Sophie...
    for ix, sub_dir in enumerate(dirNames_onDisk):
        current_sub = int(sub_dir[-3:-1])
        if current_sub in subject_id_list:
            subdir_onDisk = sorted(glob2.glob(sub_dir + '*' + os.sep))
            subdir_names = list(map(lambda x: os.path.basename(os.path.normpath((x))), subdir_onDisk))
            assert expConditions_onDisk <= subdir_names, f"missing these conditions {set(expConditions_onDisk) - set(subdir_names)} in {sub_dir}"
            metafile_df.loc[(current_sub, 'Aesthetic'), 'data_dir'] = \
            [i for i in subdir_onDisk if os.path.basename(os.path.normpath((i))) == 'Esthetique'][
                0]  # same: a = list(filter(lambda x: x[-11:-1] =='Esthetique',subdir_onDisk))
            metafile_df.loc[(current_sub, 'Narrative'), 'data_dir'] = \
            [i for i in subdir_onDisk if os.path.basename(os.path.normpath((i))) == 'Narratif'][0]
            metafile_df.loc[(current_sub, 'Goals'), 'data_dir'] = \
            [i for i in subdir_onDisk if os.path.basename(os.path.normpath((i))) == 'Objectif'][0]
    del ix, sub_dir, current_sub, subdir_onDisk, subdir_names

    # parser: check and get all files paths for each participant and scenario
    for meta_ix, meta_row in metafile_df.iterrows():
        for file_col, scene_onDisk in zip(fileName_cols, fileName_scenes_onDisk):
            curr_files = glob2.glob(meta_row.data_dir + '*' + scene_onDisk + '*' + meta_ix[2] + '.csv')
            if len(curr_files) == 1:
                metafile_df.at[meta_ix, file_col] = curr_files[0]  # one file: path as a string
            elif len(curr_files) > 1:
                metafile_df.at[meta_ix, file_col] = curr_files  # two or more files: path as a list of string

    # add correct record_ix from corresponding profiling_df indexes
    if type(profiling_df) is pd.core.frame.DataFrame:
        for meta_ix, meta_row in metafile_df.iterrows():
            metafile_df.at[meta_ix, 'profiling_index'] = profiling_df.loc[
                (meta_ix[0], meta_ix[1]), ('metadata', 'profiling_index')]

    return metafile_df



