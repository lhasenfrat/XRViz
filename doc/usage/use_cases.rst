# Use cases

Here three main use cases for using the Python Analyis Toolbox are described.

## use case 1: creating a new experiment from scratch 
(design and config files) + links to config files+ dirs screenshots

- by whom? 
trainees, doctoral students, researchers

- what are the pre-requisites?
no a-priori knowledge of python is required, here we just need to fill in 
some high-level script scenario

- steps
0) (read the doc and readme where all actions to follow are described step-by-step)
1) define the structure of the experiment in file “design_xxx.yaml”
    - config file in .yaml format (human readable) and pre-filled (template is made available) 
    - this file describes the experience in the simplest possible way
    - it contains all the structural information necessary for the handovers (definition of conditions / scenes , their sequence or random, etc.) 
necessary for the VR app to run the experiment, and later for the Python analysis toolbox to analyse the obtained data: 
    - this is typically required for the correct (and automatic) formatting of data and metadata (indexes and fields to save, tree structure of saved files, etc.)
    - see file "design_template.yaml" in config dir for an in-depth example.
    - this file can be copy-paste or changed manually using any available text editor.
2) define the questionnaires content in file(s) “pyquest_xxx.yaml”
    - config file in .yaml format (human readable) and pre-filled (template is made available)
    - these files contain all the information necessary to present the questionnaires 
    (they are taken as input by the pyquest application) and compute their scores / dimensions within the python analysis toolbox.
    - see file "pyquest_FPT.yaml" in config dir for an in-depth example.
3) define specific VR content in .csv format files
    - write-in format files for objectsData and actionsData in config dir 
        - "objectsFormat.csv" (format info for the recording of all xxx_objectsData_xxx.csv)
        - "actionsFormat.csv" (format info for the recording of all xxx_actionsData_xxx.csv) 
    - here are set all info on tracked objects and user interactions in the VR environment that is specific to this experiment
    - see files "objectsFormat_template.csv" and "actionsFormat_template.csv" in config dir for an in-depth example.  
4) develop VR content 
    - matching the structure defined in step 1) in "design_template.yaml" (conditions, scenes, etc.)
    - matching the object and action formats defined in step 3)
5) write documentation and instructions
    - write about anything that is relevant about experiment design 
    - provide a centralized description of all the relevant elements of the experience (structure, special cases, etc.)
    - useful for futur scientist to get a full picture of the choices that were made regards to experiment design in addition to config files
    - also write all instructions for the experiment recording (passations): participant instructins, consent forms, flyers, gift vouchers, etc.
    - why not also write methods and hypotheses before data collection? (#openscience! see: https://osf.io/4fczx/download/)


## use case 2: data collection 
links to config files+ dirs screenshots

- by whom? 
trainees, doctoral students

- what are the pre-requisites?
no programming knowledge 

- steps
0) read information from doc and instructions about this experiment
1) setup the computer used for data collection
    - this step should be realized once for all after experiment design and before data collection, 
    and it should be realized again if using a new computer (e.g., new passation site)
    - run data_io.newExperienceConfig(basePath, design_configFile) to generate everything required by Unity to run the experiment:
        - create all participant directories within root 'basePath'
        - "experiment_metafile.csv": a metafile with all directory paths for all participants on this local machine (see example here)
        - "xxx_S01_sequencing.csv": experimental sequence for each participant in participant directory (see example here)
    - this function generates, for instance, the latin square randomization used for sequencing conditions across participants.
    - TODO: make a standalone executable from this function ?
2) launch app for running and managing VR environment (Unity, Unreal, etc.)
    - (automatic) + app should launch using the correct parameters for this participant (participant id + sequencing of condition / scenes / etc.)
    - (automatic) + VR app will launch pyquest for the right questionnaire at the right time, using info from “design_xxx.yaml” and “pyquest_xxx.yaml”
3) (if relevant) launch app for physio data collection (i.e., viewing and recording of physio data)
    - in the future, this might be automatic if using in-house python code for physio acquisition (e.g., using timeflux.io)
4) end of the data collection
    - (manual or automatic) closing Unity + Python apps + physio app
    - writing any relevant info to the "carnet d'expérience" (#openscience)


## use case 3: analyzing experiment data 
(import, verify and process, select and analyze)+ links to config files + tutorial script

- by whom? 
    - trainees, doctoral students, researchers
    - not necessarily the same people as for the data collection (therefore: document document document !)

- what are the pre-requisites?
    - basic python when using the GUI (Graphical User Interface)
    - advanced python when using scripting and analysis API 

- steps
[USING GUI]: 
    - install and run the GUI
    - follow successive steps of data import, processing and visualization using available visual panels

[USING SCRIPTING]:
0) (read the doc and readme where all actions to follow are described step-by-step)
1) install API with all requirements (library dependencies, etc.) -> see doc "Getting Started"
2) enjoy the extensive API tour by following all steps in the tutorial script "tutorial_analysis.py" 
3) read the API reference, and move on with your own analysis with trial-and-error !
4) typical analysis may include the following steps:
    - set various parameters (root dir, config file paths, etc.)
    - parse all data files and create "metafile_df", a dataframe summing up all metadata structure (subject_id, condition, scene and all file paths)
    metafile_df = data_io.createMetaFile(params["dataPath"], params["configFile"])
    - import profiling/subjective data into "profiling_df", a dataframe summing up all subjective info
    profiling_df, metafile_df = data_io.importProfilingData(metafile_df, params["configFile"]) 
    - check data health, e.g., run timestamp diagnosis, or else depending on your data 
    timestamp_info = data_proc.check_timestamps(timestamp_serie)
    - preprocess data, e.g., resample uneven time series, filter, etc.:
    eyetracking_resampled_df = resample_timestamps(eyetracking_df, timestamp_col='pd_datetime', target_fs='inferred', fill_gap_ms=100, interpolator='pchip')
    - create some data selection (participant, scenes, timesegments, etc.) by analysing or selecting profiling data and / or user interaction data
    - query, filter or aggregate behavioral data (e.g., body motion or eyetracking) based on this selection
    - compute relevant metrics from behavior (e.g., fixation rate) and compare them across conditions / groups
    - visualize and export graphics and tables
    
    
