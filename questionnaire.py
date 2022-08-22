#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file implements classes used for backend of questionnaire presentation.

*** Architecture: 
    - Questionnaire has Blocs (dict) and Scores (dict)
    - Bloc has Items (dict)
    - Score has a reference to Questionnaire (allows getting all Items required for score computation)
    
*** Pseudo-code for frontend :
    - 1. Questionnaire init: Questionnaire(config_path,save_path)
    - 2. for each bloc, following bloc order in 'Questionnaire.bloc_order':
        - presentation of items in Bloc.item_dict, following item order in 'Bloc.item_order'
        - upon each item completion, call to Item.answered(answer, answer_duration) 
    - 3. for each condition, after all questionnaires completion
        - call save_experimental_data(questionnaire_list, subject_id, site_location, condition, condition_order)
    - 4. upon questionnaire completion: 
        - call Questionnaire.completed(completion_timestamp) at questionnaire completion
        - call Questionnaire.save_results() to write results to disk


*** File format: 
Questionnaire answers and answers durations are stored in two separate .csv files
with exactly the same structure. We use a 2-rows header that allows for further 
2-level columns indexing in Pandas.

Format for column names is:
    - header row 0: [question_category_names, score_category_names]
    - header row 1: [question_items_names, score_items_names]
with for header row 0:
    - question_category_names = 'title_items'
    - score_category_names = 'title_scores'     
    OR (when no score is defined in this questionnaire):
    - question_category_names = score_category_names = 'title' 
with for header row 1:
    - question_items_names and score_items_names defined in the config file
example: 
    - row 0: ['DFS_items', 'DFS_items', ..., 'DFS_scores', 'DFS_scores']
    - row 1: ['SO1', 'A02', ..., 'time_transformation', 'autotelic_experience']
    
Name of the file is given by frontend (save_path), with following rules:
    - [subject_id]_[answers]_[title]_[condition].csv for answers
    - [subject_id]_[answers-duration]_[title]_[condition].csv for answers duration (generated from previous filename)
[condition] should be given only for questionnaire that are presented again in each condition
example:'S7_answers_DFS_Narrative' and 'S7_answers-duration_DFS_Narrative'

@author: jonas
"""

import pandas as pd
import numpy as np
import pathlib
import yaml



"""
# DEBUG / EXAMPLE

# questionnaire DFS
config_path = '/Users/jonas/Dropbox/Code/Python/XREcho_analysis/config/pyquest_DFS.yaml'
save_path = '/Users/jonas/Documents/PORTRAIT_local/data_Sophie/S00/profiling/S0_answers_DFS_Narrative.csv'
DFS_quest = Questionnaire(config_path,save_path) # create Questionnaire
DFS_quest.debug_generate_answers() # generate fake data
DFS_quest.completed('2021-05-21 18:12:02') # questionnaire completion and score calculation
DFS_quest.save_results() # create file with results
# questionnaire FPT
config_path = '/Users/jonas/Dropbox/Code/Python/XREcho_analysis/config/pyquest_FPT.yaml'
save_path = '/Users/jonas/Documents/PORTRAIT_local/data_Sophie/S00/profiling/S0_answers_FPT_Narrative.csv'
FPT_quest = Questionnaire(config_path,save_path) # create Questionnaire
FPT_quest.debug_generate_answers() # generate fake data
FPT_quest.completed('2021-05-21 19:12:02') # questionnaire completion and score calculation
# questionnaire TPI
config_path = '/Users/jonas/Dropbox/Code/Python/XREcho_analysis/config/pyquest_TPI.yaml'
save_path = '/Users/jonas/Documents/PORTRAIT_local/data_Sophie/S00/profiling/S0_answers_TPI_Narrative.csv'
TPI_quest = Questionnaire(config_path,save_path) # create Questionnaire
TPI_quest.debug_generate_answers() # generate fake data
TPI_quest.completed('2021-05-21 20:12:02') # questionnaire completion and score calculation
# experimental data
expe_info_dict =  {'subject_id':0, 'condition': 'Narrative', 'condition_order': 2, 'site_location': 'INSA Lyon'}
questionnaire_list = [DFS_quest, FPT_quest, TPI_quest]
save_experimental_data(questionnaire_list, expe_info_dict)
"""


class Questionnaire:
    """Questionnaire class
    Contains all information required for its presentation and data management.
    Each questionnaire object has :
    - metadata information  (title, refs, version, etc.)
    - list of blocs with questions info (scales, items, order, etc.)
    - list of scores with score info (score computation)
   """
    
    def __init__(self, config_path, save_path):
        """
        Questionnaire constructor.

        Parameters
        ----------
        config_path : str, pathlib.Path or pathlib.PurePath
            path of the .yaml config file with questionnaire info.
        save_path : str, pathlib.Path or pathlib.PurePath
            path of the .csv file in which to store participant answers and scores 
        """
        
        # --- assert inputs
        assert isinstance(config_path,str) or isinstance(config_path,pathlib.Path) or isinstance(config_path,pathlib.PurePath), \
            "config_path must be a string or pathlib.Path or pathlib.PurePath"
        if not isinstance(config_path,pathlib.Path) : config_path = pathlib.Path(config_path) # convert to pathlib.Path if needed
        assert config_path.is_file(), f"specified file '{config_path.name}' does not exist in directory '{config_path.parent}'"
        assert pathlib.Path(config_path).suffix.lower() == '.yaml', f"questionnaire config file '{config_path.name}' must be a '.yaml' file"
        assert isinstance(save_path,str) or isinstance(save_path,pathlib.Path) or isinstance(save_path,pathlib.PurePath), \
            "save_path must be a string or pathlib.Path or pathlib.PurePath"
        if not isinstance(save_path,pathlib.Path) : save_path = pathlib.Path(save_path) # convert to pathlib.Path if needed
        assert pathlib.Path(save_path).suffix.lower() == '.csv', f"save path '{save_path.name}' must be a '.csv' file"
        assert save_path.parent.is_dir(), f"save_path directory '{save_path.parent}' does not exist"
        
        # --- read template config file
        with open(config_path, 'r') as file:
            yamlread = yaml.full_load(file) 
        
        # --- fill instance variables from config file and convert what's needed to dataFrames
        # -- questionnaire metadata
        self.title = yamlread['META']['title']  # (str) short name / abbreviation of the questionnaire
        self.fullName = yamlread['META']['fullName'] # (str) full name of the questionnaire
        self.reference = yamlread['META']['reference'] # list of references for this questionnaire
        self.version = yamlread['META']['version'] # version tag for retrocompatibility
        self.save_path = save_path # path of the .csv file in which to store participant answers and scores
        duration_filename = self.save_path.name.replace('answers', 'answers-duration') 
        self.save_path_duration = self.save_path.parent.joinpath(duration_filename) # path of the .csv file with answers duration 
    
        # -- blocs
        self.bloc_dict = dict()
        for bloc_i in yamlread['BLOCS']:
            bloc_id = bloc_i['bloc_id']
            # get all items for this bloc
            item_dict = dict()
            for item_i in yamlread['ITEMS']:
                item_id = item_i['item_id']
                if item_i['bloc_id'] == bloc_id :
                    item_dict[item_id] = Item(item_id, item_i['question']) # create new item in this bloc item_dict
            # create new bloc
            new_bloc = Bloc(bloc_id, bloc_i['scale'], bloc_i['scale_legend'], item_dict, bloc_i['item_order'], bloc_i['instructions'])
            self.bloc_dict[bloc_id] = new_bloc
        self.bloc_order = yamlread['SEQUENCE']['bloc_order'] # (str) order of bloc presentation, can be left empty (just one bloc), can be 'ordered' (following order in this file) or 'random'
           
        # -- scores
        if yamlread['SCORES']:
            self.score_dict = dict()
            for score_i in yamlread['SCORES']:
                score_id = score_i['score_id']
                new_score = Score(self, score_id, score_i['calculation'], score_i['notes'])
                self.score_dict[score_id] = new_score
        else:
            self.score_dict = None
    
        # -- state variables
        self.is_answered = False  # participant answered this questionnaire (item answers are filled)
        self.is_score_computed = False # score was computed from participant answers       
        self.completion_timestamp = None # container for questionnaire completion timestamp (time of completion, format: year-month-day HH:mm:ss)
        self.completion_duration = None # container for questionnaire completion duration (sec)
        
    
    def completed(self, completion_timestamp):
        """
        Method called upon questionnaire completion.
        Swaps questionnaire states, fills completion timestamp and durations, 
        compute all scores from participant answers.
        
        Parameters
        ----------
        completion_timestamp : str
            format is 'yyyy-mm-dd hh:mm:ss' (-> precision is second unit)
            example: '2021-05-21 18:12:02'
        """
        # -- swap answered state and fill completion timestamp
        self.is_answered = True     
        self.completion_timestamp = completion_timestamp 
        
        # -- compute overall completion duration
        all_items_dict = self.get_items()
        self.completion_duration = 0
        for item_id, item in all_items_dict.items():
            self.completion_duration += item.answer_duration
        
        # -- compute all scores
        for score_id, score in self.score_dict.items():
            score.compute_score()
            

    def get_items(self):
        """
        Getter method that returns a dict with all items in this Questionnaire
        """
        all_items_dict = dict()
        for bloc_id, bloc in self.bloc_dict.items():
            for item_id, item in bloc.item_dict.items():
                all_items_dict[item_id] = item      
        return all_items_dict


    def save_results(self):
        """
        Save questionnaire answer to .csv file in save_path
        Typically called by frontend after questionnaire completion
        """
        # -- assert participant answered and score was computed
        assert self.is_answered, f"Cannot save results for questionnaire {self.title}: no participant answer yet"
        
        # -- get names of all items and scores in this questionnaire
        all_items_dict = self.get_items() # dict with Item objects
        question_items_names = list(all_items_dict.keys())   # list with question item names 
        if self.score_dict:
            question_category_names = [self.title+'_items']*len(question_items_names)
            score_items_names = list(self.score_dict.keys()) # list with score item names
            score_category_names = [self.title+'_scores']*len(score_items_names)        
        else:   # when no score is defined in this questionnaire
            question_category_names = [self.title]*len(question_items_names)
            score_items_names = []
            score_category_names = []      
        category_names = question_category_names + score_category_names # column name level-0 (names categories) to write on file header row 0
        item_names = question_items_names + score_items_names # column name level-1 (names items) to write on file header row 1
        
        # -- get values of all items and scores in this questionnaire
        answer_vals = []
        answer_duration = []
        score_vals = []
        for item_id, item in all_items_dict.items():
            answer_vals.append(item.answer)
            answer_duration.append(item.answer_duration)
        for score_id, score in self.score_dict.items():
            score_vals.append(score.value)
        item_vals =  answer_vals + score_vals # concatenate answer and score values
        
        # -- make a Pandas dataFrame with questionnaire answers and scores
        multicol_list = list(zip(category_names, item_names))
        multicol_index = pd.MultiIndex.from_tuples(multicol_list, names=['profiling_categories', 'profiling_items'])
        quest_answers_df = pd.DataFrame(columns=multicol_index) # create Pandas dataFrame from multi-column index
        quest_answers_df.loc[0] = item_vals # fill in data (participant answers and corresponding calculated scores)

        # -- make another Pandas dataFrame with questionnaire answers duration (using same structure except for scores)
        multicol_list_questions_only = list(zip(question_category_names, question_items_names))
        multicol_index_questions_only = pd.MultiIndex.from_tuples(multicol_list_questions_only, names=['profiling_categories', 'profiling_items'])
        quest_answers_duration_df = pd.DataFrame(columns=multicol_index_questions_only) 
        quest_answers_duration_df.loc[0] = answer_duration # fill in data (participant answer durations)
        
        # -- write answers and answers duration dataFrames to disk
        quest_answers_df.to_csv(path_or_buf=self.save_path, index= False, encoding="iso8859_15")  # save answers and scores on disk
        quest_answers_duration_df.to_csv(path_or_buf=self.save_path_duration, index= False, encoding="iso8859_15") # save completion duration for each item in separate file 
        #quest_answers_df_read = pd.read_csv(self.save_path, header=[0,1], encoding="iso8859_15") # (debug check)
        #quest_answers_duration_df_read = pd.read_csv(self.save_path_duration, header=[0,1], encoding="iso8859_15") # (debug check)

    
    def load_results(self):
        """
        load questionnaire answer from .csv file in save_path
        used for importing profiling data
        
        -- Outputs
        quest_answers_df : Pandas dataframe
            dataframe containing all items and scores stored in questionnaire save_path
        quest_answers_duration_df : Pandas dataframe
            dataframe containing all items completion times (None if no .csv file on disk)
        """
        
        # -- load questionnaire answers and answer duration to Pandas Dataframe
        quest_answers_df = pd.read_csv(self.save_path, header=[0,1], encoding="iso8859_15") 
        try:
            quest_answers_duration_df = pd.read_csv(self.save_path_duration, header=[0,1], encoding="iso8859_15") 
        except: 
            quest_answers_duration_df = None   
            answer_duration = None
        
        # -- load all questionnaire items and scores 
        all_items_dict = self.get_items() # dict with Item objects
        all_score_dict = self.score_dict # dict with Score objects
        for item_id, item in all_items_dict.items():   
            answer = quest_answers_df.xs(item_id, axis='columns', level=1).iat[0,0] # accessing value this way so we don't have to specify name of column level-0 
            if quest_answers_duration_df:
                answer_duration = quest_answers_duration_df.xs(item_id, axis='columns', level=1).iat[0,0]
            item.answered(answer,answer_duration)
        if self.score_dict:
            for score_id, score in all_score_dict.items():   
                score.value = quest_answers_df.xs(score_id, axis='columns', level=1).iat[0,0]
        
        # -- return dataframes
        return quest_answers_df, quest_answers_duration_df
        
     
    def debug_generate_answers(self):
        """
        debug function used to generate random values for all items in questionnaire
        """
        all_items_dict = self.get_items() # dict with Item objects
        for item_id, item in all_items_dict.items():   
            random_answer = np.random.randint(0,6)        # just like a good old Lickert-7
            random_answer_duration = np.random.random()   # float in range [0-1]
            item.answered(random_answer, random_answer_duration)
        

class Bloc:
    """Bloc class
    Each bloc defines a set of items / questions with given instructions and scale.
    Bloc order can be set or randomized, same applied for items order within bloc.
    """
    def __init__(self, bloc_id, scale, scale_legend, item_dict, item_order='ordered', instructions=''):
        self.bloc_id = bloc_id #(str) id of this bloc
        self.scale = scale # (str) type of scale for all items in this bloc, can be 'lickert-n', 'continuous_slider', 'image_choice', etc. (to be implemented)
        self.scale_legend = scale_legend # (str list) legend for each step of the scale (list with length n, can have empty elements)
        self.item_dict = item_dict # dictionnary with all Item objects for this bloc
        self.item_order = item_order # (str) item orders within that bloc, can be 'ordered' (following item ids) or 'random'
        self.instructions = instructions # (str) participant instructions for that bloc (presented once at bloc begin)
        

class Item:
    """Item class
    Each item has a question string and a unique item_id used for referencing and 
    score computation.
    """ 
    def __init__(self, item_id, question):
        self.item_id = item_id # (str) unique item id
        self.question = question # (str) unique bloc id
        self.answer = None # container for participant answer
        self.answer_duration = None # container for participant answer duration for this item (in sec, float)

    def answered(self, answer, answer_duration):
        """ Method called upon item completion
        """
        self.answer = answer
        self.answer_duration = answer_duration

class Score:
    """Score class
    Contains infos for one score calculation (expression) and its value
    """
    def __init__(self, questionnaire, score_id, calculation, notes=''):
        self.questionnaire = questionnaire # (Questionnaire) reference to quest in which this score is embedded
        self.score_id = score_id # (str) unique score id
        self.calculation = calculation # (str) math expression using item ids (e.g., 2*A - 4*B + 1.5C )
        self.notes = notes # (str) any relevant information about this score (can be left empty)
        self.value = None # container for score computation
    
        
    def compute_score(self):
        """
        Compute score in self.value using math expression from questionnaire 
        config file and participant answer to items.
        Typically called by frontend after questionnaire completion.
        """
               
        # get all items in this questionnaire
        all_items_dict = self.questionnaire.get_items() # dict with Item objects
        all_items_names = list(all_items_dict.keys())   # list with item names
        
        # get variable names (question_item_id) and values
        score_calc_str = self.calculation.replace(" ", "") # remove all whitespaces
        
        # find item names in score expression and get their values
        vars_list = []
        vars_list = [(vars_list+[item])[0] for item in all_items_dict if item in score_calc_str]
        
        # assert that score can be calculated (i.e., it uses valid item_ids in questionnaire config file)
        assert vars_list, f"Something wrong with calculation of score '{self.score_id}': no valid item_id found in '{score_calc_str}'. Please check config file for questionnaire '{self.questionnaire.title}' "
        
        # get item values
        vars_dict = {var: all_items_dict[var].answer for var in vars_list}
        
        # evaluate expression to compute score
        for key, value in vars_dict.items(): 
              score_calc_str = score_calc_str.replace(key, str(value))
        self.value = eval(score_calc_str)
        
        
        
def save_experimental_data(questionnaire_list, expe_info_dict):
    """
    Create a .csv file containing metadata and experimental items
 
    Parameters
    ----------
    questionnaire_list : list of Questionnaire objects
        used for saving completion timestamp for each filled questionnaire 
        and accessing file paths
    expe_info_dict : dictionnary
        contains all relevant experimental info
        typical data includes: 'subject_id', 'site_location', 'condition', 'condition_order'
       
    Notes
    ---------
    Because some questionnaires are presented again for each condition, this 
    should be called once for each condition after all questionnaires completion
    Experimental datafiles should typically contain the following fields:
        'subject_id' (int), 'site_location' (str), 'condition' (str), 'condition_order' (int) 
        + x1 completion timestamp for each filled questionnaire    
    /!\ any modification of these fields must also be applied to experimental data import 
    in data_io.importProfilingData(metafile_df, design_configFile)
    """

    # -- assert inputs
    assert isinstance(expe_info_dict,dict), "expe_info_dict must be a valid dictionnary"
    assert isinstance(questionnaire_list,list), "questionnaire_list must be a valid list of Questionnaire objects"
    for quest_i in questionnaire_list:
        # assert isinstance(quest_i, Questionnaire), f"error: {quest_i} is not a valid Questionnaire object"     # MARCHE PÔ?..
        assert quest_i.is_answered, f"error: questionnaire '{quest_i.title}' must be answered first before saving experimental data on disk"
    
    # -- get names of items to store from expe_info_dict
    item_names = list(expe_info_dict.keys())
    item_values = list(expe_info_dict.values())
    
    # -- append data about questionnaire completion timestamps
    for quest_i in questionnaire_list:
        item_names.append(quest_i.title+'_completion')
        item_values.append(quest_i.completion_timestamp)
    category_names = ['experimental'] * len(item_names)
    
    # -- make a Pandas dataFrame with this experimental data
    multicol_list = list(zip(category_names, item_names))
    multicol_index = pd.MultiIndex.from_tuples(multicol_list, names=['profiling_categories', 'profiling_items'])
    experimental_df = pd.DataFrame(columns=multicol_index) # create Pandas dataFrame from multi-column index
    experimental_df.loc[0] = item_values # fill in experimental data 
    
    # -- get path for saving into .csv file    
    experimental_file_name = quest_i.save_path.name.replace(quest_i.title,'experimental') # taking path from last questionnaire iterator (all questionnaire paths should have the same dir)
    experimental_path = quest_i.save_path.parent.joinpath(experimental_file_name) 
    
    # -- write .csv file with experimental data
    experimental_df.to_csv(path_or_buf=experimental_path, index= False, encoding="iso8859_15")  # save answers and scores on disk
    # experimental_df_read = pd.read_csv(experimental_path, header=[0,1], encoding="iso8859_15") # (debug check)








    