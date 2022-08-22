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
from data_io import *




def calculate_basic_stats(metafile_df, data_type='objects', target_id=0):
   """
   calculation of some basic statistics on multiple files (objectsData and actionsData) 
   listed in metafile_df, and then parsed and loaded here file by file 
   (nothing is kept in working memory) 
   any new stats on objectsData or actionsData may be implemented here
   
    -- Inputs
    metafile_df: Pandas DataFrame 
        used here to get list of participants to import data from with their
        profiling dir paths on disk
    data_type: str, default is 'objects' 
        specifies data type among 'objects' or 'actions' 
    target_id: int, default is VR camera (object_id = 0)
        specifies name of target on which to calculate basic stats 
            - target corresponds to 'object_id' when data_type is 'objects' 
            - target corresponds to 'action_id' when data_type is 'actions' 
        
    -- Outputs
    stats_info_df : Pandas dataframe
        dataframe with as many rows as metafile_df + specific columns with stat infos
   """     
        
   # -- assert inputs 
   assert isinstance(metafile_df, pd.core.frame.DataFrame), "input 'metafile_df' must be valid pd DataFrame"
   all_data_types = {'objects','actions'}
   assert isinstance(data_type, str) and data_type in all_data_types, f"data_type must be specified among {all_data_types}"
   assert isinstance(target_id, int), "target_id must be an integer"

   # loop on all scenes and participants
   stats_info_df = metafile_df[['subject_id', 'condition', 'scene', 'profiling_index', 'record_index']].copy() # create a dataFrame with stats info
   for meta_ix, meta_row in metafile_df.iterrows():
       
       # -- UI print
       percent_completion = 100*(meta_ix/metafile_df.shape[0])
       if round(percent_completion % 10 , 0) == 0.0:
           print(f'\ncalculating basic stats on all objectsData...{int(percent_completion)}%', end='')
       print('.', end='')     
        
       # -- load datafiles
       if data_type == 'objects':
           objectsData_df, objectsFormat_df = importDataFile(meta_row.file_objectsData, meta_row.file_objectsFormat, data_type='objects') # import single file (usually single scene for one participant)
           actionId_in_data = sorted(objectsData_df.objectId.unique())
           if target_id in actionId_in_data: # assert target is in data
               sel_df = objectsData_df.query('objectId == @target_id') # select this object_sel from target_id
           else:
               print(f"object #{target_id} was not found in dataset '{meta_row.file_objectsData}'")
       if data_type == 'actions':
           actionsData_df, actionsFormat_df = importDataFile(meta_row.file_actionsData, meta_row.file_actionsFormat, data_type='actions') # import single file (usually single scene for one participant)
           actionId_in_data = sorted(actionsData_df.ActionId.unique()) # /!\ 2 majuscules ici à "ActionId" (pas cohérent)
           if target_id in actionId_in_data: # assert target is in data
               sel_df = actionsData_df.query('ActionId == @target_id') # select this object_sel from target_id
           else:
               print(f"action #{target_id} was not found in dataset '{meta_row.file_actionsData}'")
     
       # -- calculate basic statistics
       if data_type == 'objects':
           # measure time spend on each scene for each participant
           stats_info_df.loc[meta_ix,'time_spend_on_scene_sec'] = (sel_df.pd_datetime.iat[-1] - sel_df.pd_datetime.iat[0]).round('S') # Timedelta in seconds
           # calculate some stats on distances
           sel_df = calculate_distances(sel_df, norm=2) # calculate all distances (positions and rotations)
           stats_info_df.at[meta_ix,'total_pos_dist'] = sel_df.loc[:,'pos_distance'].sum()
           stats_info_df.at[meta_ix,'total_rot_dist'] = sel_df.loc[:,'rot_distance'].sum()
           # ***
           # TODO: here add whatever stats you need on objects data
           # stats_info_df.at[meta_ix,'xxx'] = ...
           # ***
       elif data_type == 'actions':
           # calculate number of actions
           stats_info_df.at[meta_ix,'total_nb_actions'] = len(sel_df)
           # ***
           # TODO: here add whatever stats you need on actions data
           # stats_info_df.at[meta_ix,'xxx'] = ...
           # ***
           
   return stats_info_df



def calculate_distances(objectsData, norm=2):
    """
    calculate sample-to-sample distances in some objectsData using a given distance norm

    Parameters
    ----------
    objectsData : Pandas DataFrame 
        objectsData with fields position.x/y/z and rotation.x/y/z
    norm: {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, default is '2'
        norm used for distance calculation
        see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    Returns
    -------
    objectsData : Pandas DataFrame 
        + added columns 'pos_distance' and 'rot_distance'
    """
    
    # -- assess input
    assert isinstance(objectsData, pd.core.frame.DataFrame), "input 'objectsData' must be valid pd DataFrame"
    cols_pos = ['position.x', 'position.y', 'position.z']
    cols_rot = ['rotation.x', 'rotation.y', 'rotation.z']
    target_cols = cols_pos + cols_rot
    assert set(target_cols) <= set(objectsData.columns), f"missing these columns in input objectsData: {set(target_cols)-set(objectsData.columns)}"
    
    # -- calculate distances
    pd.options.mode.chained_assignment = None # handles false positive of "SettingWithCopyWarning"
    objectsData.loc[:,'pos_distance'] = np.linalg.norm(objectsData.loc[:,cols_pos].diff(), axis=1, ord=norm)
    objectsData.loc[:,'rot_distance'] = np.linalg.norm(objectsData.loc[:,cols_rot].diff(), axis=1, ord=norm)
    pd.options.mode.chained_assignment = 'warn' # get back to normal "SettingWithCopyWarning" settings

    return objectsData


def synchronize_datasets(dataset_list, sync_timestamps, reset_timestamps):
    # absolute sync: look for ref_timestamp in all dataset_list (in a range such as max 100ms around)
    # relative sync: directly take a list of ref_timestamp as input (one for each dataset)
    #   usage will be to first attempt identifying an action / event and get this timestamp in each dataset 
    # shift timestamps so as to 
    #   1. reference this time_sync instant as time zero (reset_timestamps = True) 
    #       or
    #   2. rereference all timestamps to first sync_timestamp (reset_timestamps = False)
    #  
    # TODO
    # example
    #   synchronize_datasets([objectsData_df, eyetracking_df], [(),(first sample)])
    print('todo')


def resample_timestamps(df_in, timestamp_col, target_fs='inferred', fill_gap_ms=100, interpolator='pchip'):
    """
    This function transform unevenly-spaced times series data to evenly-spaced 
    representations using interpolation.

    Parameters
    ----------
    df_in : Pandas DataFrame 
        typically contains timestamp data in "col_name"
    timestamp_col : str 
        name of column in which we find timestamps.
    target_fs: float or int, or 'inferred'; default is 'inferred'
        target sampling_rate, expressed in Hz
        if set to 'inferred', keeping median frequency rate in input timeserie and round it to upper value
    fill_gap_ms: float, default is 100
        time threshold, expressed in milliseconds. All gaps larger than this value
        are entirely filled with Nans instead of being interpolated.
    interpolator: str, default is pchip (1-d monotonic cubic interpolation)
        interpolation technique to use, methods of pandas.DataFrame.interpolate
        

    Returns
    -------
    df_out : Pandas DataFrame 
        "df_in" evenly-spaced with gaps filled

    Notes
    ------
    Here we transform unevenly-spaced times series data to evenly-spaced 
    representations using interpolation.
    To deal with outliers in timestamping (i.e., too large sample 
    intervals) a time threshold allows to fill gap samples with Nans.
    The goal here is to avoid generating lots of intermediate values that can affect 
    later analysis (e.g., spurious apparence of smooth pursuit behavior in 
    eyetracking data).
    We use pchip algorithm for interpolation, as it preserves monotonicity in 
    the interpolation data and does not overshoot if the data is not smooth
    (see scipy.interpolate.PchipInterpolator)
    """
    
    # -- assess input
    assert isinstance(df_in, pd.core.frame.DataFrame), "input 'df_in' must be valid pd DataFrame"
    assert type(timestamp_col) == str, "input 'timestamp_col' must be a string"
    assert target_fs == 'inferred' or isinstance(target_fs, (int, float)), "'sampling_rate' is not valid"
    assert timestamp_col in df_in.columns, f"no such timestamp column in df_in: {timestamp_col}"
    assert type(interpolator) == str, "input interpolator must be in a string with one method of pandas.DataFrame.interpolate"
    
    # -- infer target sampling rate from input timestamps  
    df = df_in.copy() 
    timestamps_in = df.loc[:,timestamp_col]
    if target_fs == "inferred": 
        sample_intervals = timestamps_in.sort_values().diff() # diff of unique values (after sorting indexes) / type : timedelta64[ns]
        # TODO: adapt to np.timedelta64 unit in case it is not right
        median_fs = np.timedelta64(1, "s") / sample_intervals.median() # inferred value is chosen as the median / expressed in Hz
        target_fs = int(np.ceil(median_fs)) # round it to upper integer value
        target_interval = int(np.floor( sample_intervals.median()/np.timedelta64(1, "ms") )) # median period rounded to lower integer value, in ms
    else:
        target_interval = int(np.floor( (1000/target_fs) )) # inverse of target frequency rate, rounded to lower integer value, in ms
        target_fs = int(np.ceil(target_fs)) # round it to upper integer value
    
    # -- resample and fill values
    pd_offset_str = str(target_interval)+'ms' # dateoffset string used to define resample interval, e.g., '12ms'
    fill_gap_samples = int(np.ceil(fill_gap_ms / target_interval)) # number of samples corresponding to fill_gap time interval
    df.set_index(timestamp_col,inplace=True, drop=True, verify_integrity=True) # set dataFrame index as timesample to use it for next upsampling 
    df_resampled = df.resample(pd_offset_str, closed='right').fillna(method='ffill',limit=1) #  use previous valid observation to fill gap with nans (forward fill).
    df_interpolated = df_resampled.interpolate(method=interpolator, limit=fill_gap_samples) # this method interpolates each gap until fill_gap_samples and then fill the overhead with NaNs
    nan_gap_mask = mask_gaps(df_resampled.iloc[:,0], fill_gap_samples) # get mask with all long gaps in data (i.e., exceeding 'fill_gap_samples' samples)
    df_interpolated.loc[nan_gap_mask] = np.nan # hard set these to NaN
    
    # -- return resampled data with new index
    df_out = df_interpolated.reset_index()
    
    return df_out
    


def mask_gaps(serie_in, fill_gap_samples):
    """
    small utils function that returns a mask with all long gaps in data 
    (Nan values), i.e., gaps of length exceeding 'fill_gap_samples' samples
    
    Parameters
    ----------
    serie : Pandas Serie 
    fill_gap_samples : int
        number of consecutive NaN samples beyond which masking is applied.
        gaps of length exactly 'fill_gap_samples' are kept (mask to False)

    Returns
    -------
    nan_gap_mask : Pandas Serie 
        mask (True for long gaps, False for non-Null values and small gaps).
    """
    
    # -- assert inputs
    assert type(serie_in) == pd.core.series.Series, "input 'serie_in' must be valid Pandas Series"
    assert type(fill_gap_samples) == int, "input 'fill_gap_samples' must be an integer value"
    
    # -- create masks
    index_in = serie_in.index
    nan_mask_serie = serie_in.isnull().reset_index(drop=True) # convert to bool (True if Nan) and make sure index is using integer values
    nan_gap_mask = nan_mask_serie
    
    # -- fill mask -quite inefficient way, we could find a more optimal approach using a rolling window (see below)
    nan_count = 0
    is_gap = False
    gap_start_ix = None
    gap_end_ix = None
    for ix, val in nan_mask_serie.iteritems():
        if not val: # case this timestamp is NOT a NaN value
            if is_gap: # this data ends a gap with NaNs
                gap_end_ix = ix
                if nan_count > fill_gap_samples:
                    nan_gap_mask.iloc[gap_start_ix:gap_end_ix] = np.nan
            is_gap = False
            nan_count = 0
        else: # case this timestamp is a NaN value
           nan_count += 1
           if not is_gap: # this is a new gap with NaNs
               is_gap = True
               gap_start_ix = ix 
        nan_gap_mask.iloc[ix] = nan_count
    nan_gap_mask = nan_gap_mask.isnull()
    
    # TODO (optimization): alternative way (draft using rolling window)
    #roll_sum = nan_mask_serie.rolling(window=fill_gap_samples+1, min_periods=1).sum()
    #roll_bool = roll_sum.rolling(window=fill_gap_samples+1, min_periods=1).apply(lambda x: x[-1]>fill_gap_samples, raw=False)
    
    nan_gap_mask.index = index_in # putting input index back
    return nan_gap_mask



def remove_outliers(df_in, col_name=None, method='IQR'):
    """
    This function removes outliers in "df_in[col_name]" 
    heuristic for outlier detection can be chosen using "method" argument

    Parameters
    ----------
    df_in : pd DataFrame or pd Series
        typically contains timestamp data in "col_name"
    col_name : str (optional)
        name of column in which we find timestamps info
        if empty, df_in must have only one column
    method: str (optional, default: IQR)
        method used to detect outliers:
        'IQR' : outliers are all samples below or above 1.5 * interquartile range (IQR)
        (no other method implemented yet)

    Returns
    -------
    df_out : Pandas DataFrame 
        "df_in" with outliers removed

    """
    
    # -- assert input
    assert isinstance(df_in, pd.core.frame.DataFrame) or isinstance(df_in, pd.core.series.Series), \
        "input 'timestamp_serie' must be valid pd DataFrame or pd Series"
    if isinstance(df_in, pd.core.series.Series):
        input_is_serie = True
        df_in = pd.DataFrame(df_in) 
    else:
        input_is_serie = False
    if not col_name:
        col_name = df_in.columns[0]
    if not method: 
        method = 'IQR'
    method_list = ['IQR']
    assert method in method_list, f"method should be one of: {method_list}"
        
    # -- implement IQR method 
    if method == 'IQR':
        # df_min = df_in[col_name].min()    # tempo: used for check 
        # df_max = df_in[col_name].max()    # tempo: used for check 
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3-q1 # interquartile range
        fence_low  = q1-1.5*iqr # this value '1.5' is equivalent to taking 2.7σ when data follows Gaussian Distribution
        fence_high = q3+1.5*iqr
        outlier_mask = (df_in[col_name] > fence_low) & (df_in[col_name] < fence_high) # this mask selects all but outliers
    
    # -- exclude outliers
    df_out = df_in.loc[outlier_mask]
    if input_is_serie:
        df_out = df_out.squeeze() 
    return df_out




def check_timestamps(timestamp_serie, sort_timestamps=True, remove_outliers = False, verbose = True):
    """
       check timestamps in a timestamp_serie (only for unique datasets)
          /!\ use instead function 'check_timestamps_multifiles()' when data is aggregated from multiple datasets

       Parameters
       ----------
       timestamp_serie : (pd Serie) with timestamps to check 
           timestamps can be float values (assumed unit: ms)
           or typed as Pandas Timestamp (better)
       sort_timestamps : (bool, optional, default: True) 
           when True, compute time intervals after sorting timestamps (in case timestamp is not monotonically increasing)
       remove_outliers : (bool, optional, default: False) 
           when True, apply outlier removal before checking timestamps
       verbose: (bool, optional, default: True) 
           UI display in console + display figures

       Returns
       -------
       timestamp_info: (dict) with 
           monotonic_increasing: bool, is True if timestamp is monotonically increasing (within dataset indexes if provided)
           constant_intervals: bool, is True if all intervals are constant
       """
       
    # -- assert input
    assert type(timestamp_serie) == pd.core.series.Series, "input 'timestamp_serie' must be valid pd Series"
    assert isinstance(timestamp_serie.iloc[0],(float,pd.Timestamp)), "input 'timestamp_serie' must have float or Pandas Timestamp values"
    assert type(sort_timestamps) == bool, "input 'sort_index' must be a bool value"
    assert type(remove_outliers) == bool, "input 'remove_outliers' must be True or False"
    assert type(verbose) == bool, "input 'verbose' must be a bool value"

    
    # -- check timestamp is monotonically increasing in the entire 'timestamp_serie'
    if timestamp_serie.is_monotonic_increasing: 
        monotonic_increasing = True 
        if verbose: print('OK: timestamps are monotonically increasing')
    if not timestamp_serie.is_monotonic_increasing: 
        monotonic_increasing = False
        if verbose: print('NOT OK: timestamps are NOT monotonically increasing')
  

    # -- check consistency of time intervals (diff of timestamps) in the entire 'timestamp_serie'   
    if not sort_timestamps :
        diff_timestamps = timestamp_serie.diff()[1:] # diff of unique values (NOT sorting indexes -> there can be gaps in case multiple datasets are aggregated)
    else :
        diff_timestamps = timestamp_serie.sort_values().diff()[1:] # diff of unique values (after sorting indexes)
    if remove_outliers:
        diff_timestamps = remove_outlier(diff_timestamps)
    diff_timestamps.reset_index(drop=True, inplace=True)
    diff_timestamps_nbvals = np.count_nonzero(diff_timestamps.unique()) # nb unique vals (should be 1...)
    if diff_timestamps_nbvals == 1:
        constant_intervals = True
        if verbose: print("OK: timestamps intervals are constant across this object's data")
    else: 
        constant_intervals = False
        if verbose: print(f"NOT OK: timestamp intervals are varying ({diff_timestamps_nbvals} different time intervals across {len(diff_timestamps)} values)")
   
    # -- compute some info summary
    timestamp_info = pd.Series(diff_timestamps).describe().to_dict()
    timestamp_info['monotonic_increasing'] = monotonic_increasing
    timestamp_info['constant_intervals'] = constant_intervals
    timestamp_info['nb_unique_intervals'] = diff_timestamps_nbvals

    # -- display raw time intervals (diff of timestamps as a fonction of sample index)
    if verbose:
        # cast to float when input is Pandas Timestamp (necessary for display)
        if isinstance(diff_timestamps.iloc[0],pd.Timedelta):
            diff_timestamps = (diff_timestamps / pd.to_timedelta(1, unit='ms')).astype(float)
              
        fig, ax = plt.subplots(1, 1, figsize=(15, 5), dpi=150) 
        if not sort_timestamps :
            ax.set_title("raw time intervals (indexes not sorted)")
            ax.set_xlabel("sample index (not sorted)")
        else :
            ax.set_title("raw time intervals (after sorting indexes)")
            ax.set_xlabel("sample index (sorted)")
        ax.set_ylabel("time intervals (ms)")
        ax.plot(diff_timestamps, color='C0', linewidth=2)
        # fig.show() # not useful on Spyder
        
        # display time intervals distribution (hist plot)
        fig, ax = plt.subplots(1, 1, figsize=(15, 5), dpi=150) 
        if not sort_timestamps :
            ax.set_title("time intervals distribution (indexes not sorted)")
        else :
            ax.set_title("time intervals distribution (after sorting indexes)")    
        ax.set_ylabel("count")
        ax.set_xlabel("time intervals (ms)")
        sns.histplot(ax=ax,data=diff_timestamps, bins= 100, kde=False, color='C0')
        # fig.show() # not useful on Spyder

    return timestamp_info
    


def check_timestamps_multifiles(timestamp_serie, index_serie, sort_timestamps=True, verbose=True):
    """
       check timestamps in a timestamp_serie 
       
       Parameters
       ----------
       timestamp_serie : (pd Serie) with time stamps to check (assumed unit: ms)
           typically one big Serie that aggregates multiple datasets
       index_serie: (pd Serie with integer record tags)
           index specifying datasets (unique integer tag for each record), 
           used to account for time gaps between different records when checking timestamps
           'index_serie' must have the same length as 'timestamp_serie' 
       sort_index : (bool, optional, default: True) 
           when True, compute time intervals after sorting indexes within each record 
           (in case timestamps are not monotonically increasing)
       verbose: (bool, optional, default: True) 
           UI display in console + display figures

       Returns
       -------
       timestamp_info: dict with
           monotonic_increasing: bool, is True if timestamp is monotonically increasing for all datasets segments in timestamp_serie
           constant_intervals: bool, is True if all intervals are constant within all datasets segments in timestamp_serie
       """

    # -- assert input
    assert type(timestamp_serie) == pd.core.series.Series, "input 'timestamp_serie' must be valid pd Series"
    assert type(index_serie) == pd.core.series.Series, "input 'index_serie' must be valid pd Series"
    assert len(index_serie) == len(timestamp_serie), "input 'index_serie' must have the same length than 'timestamp_serie' "
    assert type(sort_timestamps) == bool, "input 'sort_index' must be True or False"
    assert type(verbose) == bool, "input 'verbose' must be True or False"
    
    # -- init
    timestamp_serie = timestamp_serie.reset_index(drop=True) # indexes of single objects extracted from multiple objects datasets are not informative
    index_serie = index_serie.reset_index(drop=True) # same here
    nb_datasets = len(index_serie.unique())
    monotonic_increasing = True
    constant_intervals = True
    non_monotonic_datasets_count = 0
    non_constant_datasets_count = 0
    diff_timestamps_nbvals = 1
    super_diff_list = []
    start_i_list = [] 
    end_i_list = []
    
    # -- cut input 'timestamp_serie' using 'index_serie' and call to check_timestamps() on each segment
    for ix, dataset_i in enumerate(index_serie.unique()):
        start_i = index_serie.index[index_serie == dataset_i][0] # index of the first sample of 'dataset_i' in 'timestamp_serie'
        end_i = index_serie.index[index_serie == dataset_i][-1] # index of the last sample of 'dataset_i' in 'timestamp_serie'
        timestamp_serie_i = timestamp_serie.iloc[start_i:end_i] # segment of 'index_serie'
        timestamp_info = check_timestamps(timestamp_serie_i, sort_timestamps, verbose=False)
        check_monotonic_i = timestamp_info['monotonic_increasing']
        check_constant_i  = timestamp_info['constant_intervals']
        non_monotonic_datasets_count += check_monotonic_i
        non_constant_datasets_count += not check_constant_i
        monotonic_increasing &= check_monotonic_i
        constant_intervals &= check_constant_i
        if verbose: 
            if not sort_timestamps :
                diff_timestamps_i = timestamp_serie_i.diff().iloc[1:] # diff of unique values (NOT sorting indexes)
            else :
                diff_timestamps_i = timestamp_serie_i.sort_values().diff().iloc[1:] # diff of unique values (after sorting indexes)
            diff_timestamps_nbvals += np.count_nonzero(diff_timestamps_i.unique()) # nb unique vals (should be 1...)
            super_diff_list.append(diff_timestamps_i)
            start_i_list.append(start_i-ix) # (end-ix) as we must remove one value for each diff realised so far
            end_i_list.append(end_i-ix-1) # (end-ix-1) as we must remove one value +1 for each diff realised so far
            
    # -- UI print
    if verbose: 
        if monotonic_increasing: print(f"OK: timestamps for this object are monotonically increasing across all (x{nb_datasets}) datasets")
        else: print(f"NOT OK: timestamps are NOT monotonically increasing (x{non_monotonic_datasets_count})/{nb_datasets} datasets are non monotonic")
        if constant_intervals: print(f"OK: timestamps intervals for this object are constant across all (x{nb_datasets}) datasets")
        else: print(f"NOT OK: timestamp intervals are varying ({diff_timestamps_nbvals} different time intervals across {len(timestamp_serie)} values (x{non_constant_datasets_count}/{nb_datasets} datasets with jitter)")
        
    # -- display raw time intervals (diff of timestamps as a fonction of sample index)
    if verbose:
        if len(timestamp_serie) > 1000000 : print(f"Warning: display is going to take some time... ({len(timestamp_serie)} time samples)")
        diff_timestamps = pd.concat(super_diff_list, axis=0)
        diff_timestamps.reset_index(drop=True, inplace=True)
        # cast to float when input is Pandas Timestamp (necessary for display)
        if isinstance(diff_timestamps.iloc[0],pd.Timedelta):
            diff_timestamps = (diff_timestamps / pd.to_timedelta(1, unit='ms')).astype(float)
        
        # create fig and display params 
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(1, 1, figsize=(15, 5), dpi=200) 
        if not sort_timestamps :
            ax.set_title("raw time intervals (indexes not sorted)")
            ax.set_xlabel("sample index (not sorted)")
        else :
            ax.set_title("raw time intervals (after sorting indexes")
            ax.set_xlabel("sample index (sorted)")
        ax.set_ylabel("time intervals (ms)")
        
        # plot raw time intervals and fill areas to distinguish between datasets 
        sns.lineplot(data=diff_timestamps, palette="tab10", linewidth=1)
        # TODO: consider plotting using Vispy for very large datasets ?
        # TODO: use multicolored lines instead https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
        cmap = mpl.cm.get_cmap('Pastel1', nb_datasets)
        ylims = ax.get_ylim()
        # breakpoint()
        for dataset_i in range(nb_datasets): 
            ax.fill_betweenx(ylims, start_i_list[dataset_i], end_i_list[dataset_i]+1, facecolor=cmap(dataset_i))
        # fig.show() # not useful on Spyder
        
        # display time intervals distribution (hist plot)
        fig, ax = plt.subplots(1, 1, figsize=(15, 5), dpi=200) 
        if not sort_timestamps :
            ax.set_title("time intervals distribution (indexes not sorted)")
        else :
            ax.set_title("time intervals distribution (after sorting indexes)")    
        ax.set_ylabel("count")
        ax.set_xlabel("time intervals (ms)")
        sns.histplot(ax=ax,data=diff_timestamps, bins=100, kde=False, color='C0')
        # fig.show() # not useful on Spyder
    
    # put data in shape to return it
    timestamp_info = {'monotonic_increasing': monotonic_increasing, 
                      'constant_intervals': constant_intervals}
            
    return timestamp_info
    

def timestamps_diagnosis(metafile_df, target='cameraData'):
    '''
    make a complete check of timestamps in listed files 
    return timestamp info summary as a new column of metafile_df (or update existing column) 
    also return Pandas DataFrame with aggregated timestamp information from all analysed datasets

    Parameters
    ----------
    metafile_df : Pandas DataFrame 
        contains all metadata structure (subject_id, condition, scene and all file paths) 
    target : str, optional
        specifies target file type (name of column in metafile_df) 
        possible single value is : 'cameraData' or 'eyetrackingData' 
        default is 'cameraData'.
    
    Returns
    -------
    metafile_df = metadata structure updated with a new column with timestamp info
    all_timestamps_info_df : Pandas DataFrame with aggregated timestamp information from all analysed datasets: 
        time interval (count, mean, min, max, median, std), monotonic_increasing, constant_intervals, nb_unique_intervals 
        
    Notes about jitter in objectData and eyetrackingData
    -------
    objectsData consist in usually multiple tracked objets, sampled only when they move >epsilon, 
    or implied in some action or event. As a consequence, for a single object 
    jitter (varying time intervals) is expected. 
    This is not the case for eyetracking data, which is supposed to be evenly sampled without condition.
    
    '''
    
    # -- assert inputs
    assert type(metafile_df) is pd.core.frame.DataFrame, "input 'metafile_df' must be a valid Pandas dataframe"
    assert type(target) == str, "input 'target' must be a single string"
    assert target in {'cameraData', 'eyetrackingData'}, f"'{target}' unknown, input target must be 'objectsData' or 'eyetrackingData'"
    
    # -- add timestamp_info column(s) if not already in metafile_df
    if 'cameraData' in target and not 'camera_timestamp_info' in metafile_df.columns:
        metafile_df = metafile_df.assign(camera_timestamp_info=None)
    elif 'eyetrackingData' in target and not 'eyetracking_timestamp_info' in metafile_df.columns:
        metafile_df = metafile_df.assign(eyetracking_timestamp_info=None)    

    # -- run timestamp check on all datafiles listed in metafile_df
    for meta_ix, meta_row in metafile_df.iterrows():
        # check cameraData
        if 'cameraData' in target:
            if type(meta_row.file_objectsData) == list or type(meta_row.file_objectsData) == str :
                objectsData_df, objectsFormat_df = importObjectData(meta_row)
                df_sel = objectsData_df.loc[objectsData_df.objectId == 0] # objectId = 0 = VR camera = player's HMD movements
                timestamp_serie = df_sel['timestamp']*1000 # timestamps in objectsData are in microsec -> convert it in ms
                metafile_df.at[meta_ix,'camera_timestamp_info'] = check_timestamps(timestamp_serie, sort_timestamps=True, verbose=False)
            else: 
                print(f"\no file_objectsData for sub {meta_row.subject_id} / condition '{meta_row.condition}' / scene: '{meta_row.scene}'!")

         # check eyetrackingData   
        elif 'eyetrackingData' in target:
            if type(meta_row.file_HTCViveProEyeData) == list or type(meta_row.file_HTCViveProEyeData) == str :
                eyetracking_df = importEyetrackingData(meta_row)  
                timestamp_serie = eyetracking_df['time(100ns)']
                metafile_df.at[meta_ix,'eyetracking_timestamp_info'] = check_timestamps(timestamp_serie, sort_timestamps=True, verbose=False)    
            else: 
                print(f"\no file_eyetrackingData for sub {meta_row.subject_id} / condition '{meta_row.condition}' / scene: '{meta_row.scene}'!")
        
        # UI STUFF
        percent_completion = 100*(meta_ix/metafile_df.shape[0])
        if round(percent_completion % 10 , 0) == 0.0:
            print(f'\nchecking timestamps in {target}... {int(percent_completion)}%', end='')
        print('.', end='')     
    
    # -- create output summary dataframe
    if 'cameraData' in target:
        all_timestamps_info_df = pd.DataFrame(columns=list(metafile_df.iloc[0].camera_timestamp_info.keys()))
        for meta_ix, meta_row in metafile_df.iterrows():
            if meta_row.camera_timestamp_info : # ensure timestamp_info dict is there
                all_timestamps_info_df.loc[meta_ix] = list(meta_row.camera_timestamp_info.values())
    elif 'eyetrackingData' in target:
        all_timestamps_info_df = pd.DataFrame(columns=list(metafile_df.iloc[0].eyetracking_timestamp_info.keys()))
        for meta_ix, meta_row in metafile_df.iterrows():
            if meta_row.eyetracking_timestamp_info :  # ensure timestamp_info dict is there
                all_timestamps_info_df.loc[meta_ix] = list(meta_row.eyetracking_timestamp_info.values())
    
    return metafile_df, all_timestamps_info_df


    
def report_timestamp_info(all_timestamps_info_df):
    """
    Generate a small report from timestamp analysis

    Parameters
    ----------
    all_timestamps_info_df : Pandas Dataframe (one row per analysed file)
        columns:   'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
                   'monotonic_increasing', 'constant_intervals', 'nb_unique_intervals'
        rows:     

    Returns
    -------
    summary_timestamp_df : Pandas Series with timestamp info summary
    """
    
    # summarize timestamp infos
    #breakpoint()
    summary_timestamp_df = pd.Series(dtype=float)
    summary_timestamp_df['grand_mean'] = all_timestamps_info_df['mean'].mean()
    summary_timestamp_df['grand_min'] = all_timestamps_info_df['min'].min()
    summary_timestamp_df['grand_max'] = all_timestamps_info_df['max'].max()
    summary_timestamp_df['mean_std'] = all_timestamps_info_df['std'].mean()
    summary_timestamp_df['mean_25%'] = all_timestamps_info_df['25%'].mean()
    summary_timestamp_df['mean_50%'] = all_timestamps_info_df['50%'].mean()
    summary_timestamp_df['mean_75%'] = all_timestamps_info_df['75%'].mean()
    summary_timestamp_df['count'] = all_timestamps_info_df['count'].sum()
    summary_timestamp_df['nb_unique_intervals'] = all_timestamps_info_df['nb_unique_intervals'].sum()
    summary_timestamp_df['percent_varying_intervals'] = 100*summary_timestamp_df['nb_unique_intervals']/summary_timestamp_df['count']
    
    # UI print
    print(f"nb analysed datasets: {len(all_timestamps_info_df)}")
    print(f"nb total intervals: {round(summary_timestamp_df['count'])}")
    print(f"percent varying intervals: {round(summary_timestamp_df['percent_varying_intervals'],2)}%")
    print(f"intervals grand mean: {round(summary_timestamp_df['grand_mean'],2)} ms")
    print(f"intervals mean(median): {round(summary_timestamp_df['mean_50%'],2)} ms")
    print(f"intervals mean(std): {round(summary_timestamp_df['mean_std'],2)} ms")
    print(f"intervals mean(25th perc): {round(summary_timestamp_df['mean_25%'],2)} ms")
    print(f"intervals mean(75th perc): {round(summary_timestamp_df['mean_75%'],2)} ms")
    print(f"intervals min(min): {round(summary_timestamp_df['grand_min'],2)} ms")
    print(f"intervals max(max): {round(summary_timestamp_df['grand_max'],2)} ms")

    return summary_timestamp_df
    

def show_object_repartition(objectsData_df, objectsFormat_df, labels_dict = None):
    """
    show object repartition (with grouping of rare objects with a small representation)

    Parameters
    ----------
    objectsData_df : Pandas DataFrame
        single objects datafile.
    objectsFormat_df : Pandas DataFrame
        associated format info
    labels_dict: dict
        any label to show on the figure
        example: 
            labels_dict = {'subject': sub_sel, 'condition': 'condition_sel', 'scene': 'scene_sel'}

    Returns
    -------
    None.

    """
    
    # --  checking that objects present in objectsData are not listed in objectsFormat
    object_sel = objectsData_df
    unknown_objectId_list, unknown_objectMask = maskUnknownObjects(objectsData_df, objectsFormat_df)
    print('/!\ These objects present in objectsData are not listed in objectsFormat: ' + str(unknown_objectId_list))

    # -- remove unknown objects
    if unknown_objectMask != None and unknown_objectMask.any():
        object_sel = objectsData_df.loc[~unknown_objectMask]
        
    # -- aggregate rare objects
    groupBy_objectId = object_sel.groupby(by="objectId")['objectId'].count() * (100/len(objectsData_df)) # select by object_id and aggregate to get percentage / total count
    groupBy_objectId = pd.DataFrame(data=groupBy_objectId)
    groupBy_objectId = groupBy_objectId.rename(columns={'objectId': 'data_count'})
    groupBy_objectId['objectNames'] = getObject_NameFromId(objectsFormat_df, groupBy_objectId.index.to_list())
    rare_mask = (groupBy_objectId.data_count.sort_values().cumsum() < 10).sort_index() # this selects objects whose cumulative sum is <10% total representation in objectsData_df
    rare_objects = rare_mask[rare_mask].index.to_list()
    rare_summed = pd.DataFrame({'data_count': groupBy_objectId[rare_mask].data_count.sum(),'objectNames' : 'other objects whose cum sum <10%'},index = [0])
    repartition_objectId = pd.concat([groupBy_objectId[~rare_mask], rare_summed], ignore_index=True, axis=0)
    
    # -- plot pie chart
    fig, ax = plt.subplots(dpi=150)
    ax.pie(repartition_objectId['data_count'], labels=repartition_objectId['objectNames'], autopct='%.0f%%', shadow=False);
    
    # -- show labels / title info if given
    if labels_dict:
        sub_sel = labels_dict['subject']
        condition_sel = labels_dict['condition']
        scene_sel = labels_dict['scene']
        title_str = fig.suptitle(f'object repartition in data for subject "{sub_sel}" / condition "{condition_sel}" / scene "{scene_sel}"')
    
    # -- display
    plt.show()
    # bar plot ()
    # object_sel = objectsData_df.loc[~unknown_objectMask]
    # object_sel.loc[object_sel.actionId.isin(rare_objects),'actionId']= -1 # here we replace all small represented objects to aggregate them with -1
    # ax = sns.countplot(x="actionId", data=object_sel)
    # plt.show()
    
    
    
    