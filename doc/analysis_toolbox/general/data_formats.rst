# Data formats


## File formats

All **data files** are stored into **standard .csv format**.
This format is super convenient to store tabular data, with file in/out being 
straightforward with most data analysis and manipulation tools such as Pandas.
Different types of data are stored into separate files in order to minimize 
memory occupation, redundancy, and in/out processes: 
- questionnaires data
- objects data
- actions data
- events data (obsolet)
- eyetracking data
(- physio data)
In addition, each .csv data file is associated with a single .csv format file 
that usually describes which values and parameters are used here.

In future releases of XRecho, using **binary file format** maybe relevant to 
improve performance during storing and loading large datasets 
(e.g., for objects data or eyetracking data with relatively high sampling rate)

All **configuration files** are stored into **standard . yaml format**.
This is a human-readable file format that can also be read easily 
(YAML is a superset of JSON) see e.g., https://en.wikipedia.org/wiki/YAML


### questionnaire data
Each questionnaire is configured in dedicated .YAML file (one for each questionnaire, see use_cases).
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


### objects data

- xxx_objectsData_xxx.csv
This file stores all raw movement information concerning objects that are tracked in the VR environnement.
For each frame, each time a tracked object moves more than some given epsilon, 
XRecho adds a line to this file with: 
    - timestamp Unity, timestamp UNIX ticksSince1970 (100ns)
    - actionId: object index in objectsFormat starting from zero which allows to find what object we are talking about
    - position.x,position.y,position.z,rotation.x,rotation.y,rotation.z

- objectsFormat.csv
This file describes which objects are tracked in the VR environement, and what 
information is tracked for these objects (e.g., wether change of space / room is tracked). 
/!\ Note future release: 
Format info is set once for all and will usually be the same for all condition / scene across an experiment.
As thus, there is an important and unnecessary redundancy with having one single 
objectsFormat file for each corresponding objectsData file.
To adress this, in a future version of XREcho, a single format file will be 
provided for each data type within the "CONFIG" dir.


### action data
**action data** correspond to one-time events initiated by the user (most of the time), 
or more rarely initiated by an object whose some action is attached to.

- xxx_actionData_xxx.csv
Data file where all the actions are saved, with both their generic information 
(timestamp, ID of the action and ID of the object with which the user interacts) 
and specific information (parameters that are specific to a given type of action)
columns are:
    - timestamp (float): timestamp with reference to the beginning of the scene
    - ticksSince1970 (float): timestamp in absolute reference (UNIX datetime)
    - objectId (int): unique ID of the object that performed the action (e.g., 1 for the right hand…)
    - actionId (int): unique ID of the action itself
    - param1,2,3,... (various types): values that are related to parameters specific to this action
    
- actionsFormat.csv
Format file that describes the structure of actions and their associated parameters.
This file has a modular structure (you can add as many actions and parameters as you want)
Content here is ideally agnostic of what happens within Unity: action names should be 
as transparent as possible, with a user-centric point of view. 
columns are:
    - actionId (int): unique ID of the action
    - actionName (str): action name (human readable)
    - parameter_1-2-3-... (str): parameter name
    - description (str): text that describes the action and its parameters (readable by a human)    

Description of generic actions: 
    - **Grab / Hover / Activate**
        - definition: 
            - Grab: grab an object (/!\ objects tracked only) 
            - Hover: hover the controller over an object (/!\ tracked objects only) 
            - Activate: activate an object (/!\ tracked objects only) 
        - parameters: 
            - activated (bool): code for the start of the action (value 1) or the end of the action (value 0)
            - interactableID (integer): unique objectID of the object being interacted with
            - distanceToInteractable (float): distance between the object that did the action ('objectId' in actionData.csv) and the object we interact with ('interactableID') -> typically 0 if we interact with the controller, but sometimes non-zero if we interact remotely
            - distanceToHMD (float): distance between the VR headset and the object we interact with interacts ('interactableID') -> useful to know if we are acting close or far from ourselves (from our head in any case)
        - notes : 
            - in case of Grab in a vacuum: 'interactableID' remains null, distanceToInteractable is typically 0 if we are acting on the object directly with the controller, but sometimes non-zero if we act remotely
            - distanceToHMD is useful to know if you are acting near or far from you (from your head in any case)
    - **Move_with_joystick**
        - definition: 
            - the user moves using the joystick
        - parameters: 
            - activated (bool): code for the start of the action (value 1) or the end of the action (value 0)
    - **Turn** 
        - definition: 
            - the user starts and ends running through a command (eg, click on a button)
        - parameters :
             - turn_orientation (required?) (3 float): orientation vector of the HMD at the time of the Turn
    - **Teleport_Aim / Teleport / Teleport_Cancel**
        - definition:
            - Teleport_Aim: the user activates the teleportation interface / enters the teleportation mode
            - Teleport: the user teleports
            - Teleport_Cancel: the user cancels the teleportation
        - parameters:
            - activated (bool): code for the start of the action (value 1) or the end of the action (value 0)
            - interactableID: integer objectID with which we interact
            - startPosition : (vector3) start position
            - endPosition : (vector3) end position
        - notes: 
            - we refer here to the teleportation proposed in a standard way under Unity
            - a specific teleportation (eg with a particular object) will require to add a specific action on a case-by-case basis 
    - **EnterSpace**
        - definition:
            - the user (and more rarely certain tracked objects) enters or leaves a volume defines
        - parameters:
            - activated (bool): code for entry into a volume (value 1) or its exit (value 0)
            - volumeID (int): unique ID (objectID) of the volume in which one enters or exits
    - **UI_press**
        - definition
            - press / selection of a menu item (button)
        - parameters
            - activated (bool): code for the start of the action (value 1) or the end of the action (value 0) (required?)
            - UI_elementID (int): unique ID (objectID) of the menu item that was just selected (must be a tracked object)
        - notes
            - each relevant element of the interface is defined with a single objectId (must be a tracked object)
            - possibly many different things to consider: buttons, sliders, etc. 
            -> generically, for the moment we only consider the press UI (button press), other more specific UI elements will require defining new actions
    - **GazedAt**
        - definition
            - the user looks at some tracked object
        - parameters
            - activated (bool): code for the start of the action (value 1) or the end of the action (value 0)
            - distance: distance between the HMD and the object viewed
            - dispersion: dispersion of the gaze within the object… (future feature?)
        - notes
            - for the moment calculated as follows:
                - at each frame we take the gaze vector (raycast)
                - we check if this vector encounters a tracked object (first object touched: takes into account the occlusion)
                - however in the long term it will be necessary to refine the calculation of the GazedAt to take into account a foveal field of view of 3-5° (large threshold on Gaussian 2D)
    - **changeHMD**
        - definition
            - the user puts on or takes off the VR headset
        - parameters
            - activated (bool): code for the start of the action (value 1 ) or the end of the action (value 0)
            - state (str): putHMD or removeHMD


### event data (obsolet)
Data on events that are external to the experiment, e.g., remove the head mounted device, load the scenes, etc.
Contains Unity timestamp, universal timestamp, event id.
/!\ This will be removed in a future release (redundant with actionsData).


### eyetracking data
- eyetracking data is recorded into a standard .csv file 
- /!\ data format (columns) in this file is specific to the HTC Vive Pro Eye, 
which will change in future releases of XREcho (i.e., keeping standard column names whichever eyetracking device is used)



## data structure on disk
For a given experiment, data structure is hard-set as follows

- root dir: base directory
    - file "experiment_metafile.csv"
        - a metafile with all directory paths for all participants on this local machine
	- files "xxxFormat.csv"
    	- these files contain formatting information for each data type: 
        	- objectsFormat.csv (format info for xxx_objectsData_xxx.csv)
        	- actionsFormat.csv (format info for xxx_actionsData_xxx.csv)        	
	- dir "CONFIG": a directory with all configuration files used to run and analyse this experiment
		- file "exp-structure_title-exp.yaml": one config file with experiment structure (names of questionnaires, names of conditions and scenes, sequence, etc.)
		- file "pyquest_title-quest.yaml": one config file for each questionnaire that is used (contains all info required to present and analyse that questionnaire)
	- dir "DATA": main directory containing all participant data for this experiment (questionnaires + behavior + eyetracking + physio)
		- dir "SUBJECT": one directory per participant with name "S"+(index participant)
			- file "xxx_S03_sequencing.csv"
    			- this file is used by the VR env application (e.g., XREcho) to run the experiment. 
    			- it contains sequencing information for this participant (from info in experiment design config file)
    			- it has one row for each item to present 
			- dir "profiling": inside each subject dir, one dir containing questionnaire answers
				- one .csv file containing participant answer for each questionnaire
				- name of these files are: "S"+(index participant) + '_' +  title_questionnaire (defined in pyquest_title-quest.yaml)
			- dir [condition]: inside each subject dir, one dir for each condition:
				- folder name is the same as condition name defined in exp-structure_title-exp.yaml 
				- this folder includes the following datasets (.csv files, one for each scene):
				- "objectsData": VR objects data (one line per frame for each moving tracked object)
				- "objectsFormat": XR-Echo info for VR replay (object type, trackedData, position, etc.)
				- "actionsData": user action in the VR environement (one line per frame for each action performed)
				- "actionsFormat": defines action structure and parameters for this experiment / condition / scene
				- "eventsData": data about contextual VR events (removing headset, loading scene, etc.)
				- "eventsFormat": event metadata (event id + event name)
				- "HTCViveProEyeData": HTC Vive Pro Eye data in original format

    see also: 
    - function data_io.createMetaFile()
    - function data_io.newExperienceConfig()
				
		
				
				
				
				
				
				