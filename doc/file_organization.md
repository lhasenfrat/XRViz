## The project is organised in multiple files :

- XRViz.kv : This file contains the kivy structure. Few modifications are required. 

- AnalysisToolBox.py : Contains link between the analysis tool box and XRViz.

- generateCriteriaList.py : Generate the criteria list in yaml format. Must be changed for every experiment. 

- plotCreation.py : Contains methods for plotting data. Needs a rework.

- main.py : Contains the logic of the Kivy app. 

- data_io.py,data_proc.py, mainJonas.py, questionnaire.py, tutorial_analysis.py,transformQuestionnaires.py : Files related to analysis toolbox, more informations on 
https://github.com/Plateforme-VR-ENISE/XREcho_analysis_toolbox

- graph_widget.py : File related to matplotlib graph insertion in Kivy.

- EnrichData.py : Enrich XREcho data with actions data. Uncommented and should not be useful in most cases. 