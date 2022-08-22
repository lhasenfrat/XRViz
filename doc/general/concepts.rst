# Concepts





## Questionnaires
Questionnaire presentation uses a separate Python toolbox for display (pyquest).
In turn, Pyquest relies itself on the backend implemented within the current 
Python analysis toolbox (see questionnaire.py class).


## notions of volume / space and analysis of the user's spatial movements
so far, only the raw movements (spatial coordinates) of the users have been recorded,
but it is important and useful to to be able to describe these movements with a higher-level view, which takes into account a notion of space / volume crossed (e.g. to know when the user enters or leaves a given room within a given scene)
in this way, we will mainly analyze the spatial movements of the user, but also sometimes of certain objects of interest
to do this, we carry out a segmentation of space with volume gameObjects, and we generate an action 'enterSpace' each time the user (or tracked object) changes space.
in practice, this feature requires several additions:
in Unity: when creating the environment of a new experience, integrating these volumes in the form of tracked objects that the user / the objects of interest will pass through (Louis will create a volume template with Unity prefab)
in the “objectFormat.csv” file: addition of these volume objects themselves as tracked objects + addition for each object if we track its room change (bool trackEnterSpace)
obsolete elements
events : previously described the scene changes
-> we group this in actionFormat
declareCamera (formerly in objectFormat)
required for the replay -> we move what is necessary in the JSON exp_structure.json
also allows to know when the user has put or removed the headset -> we rename and keep this action in actionFormat.csv
start / end of actions
all actions do not necessarily have a start and end
for actions that do, we save this with a par “activated” meter (bool) which codes for the start of the action (value 1) or the end of the action (value 0)
global parameters
the recording of the movements of objects (including the movements of the user) do as soon as the object (its position vector) has moved by a certain epsilon 
the epsilon is currently hard-coded in Unity… -> we have chosen to pass it as a global parameter in the json exp_structure.json
