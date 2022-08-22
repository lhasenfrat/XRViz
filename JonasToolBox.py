from data_proc import *
from questionnaire import *

class JonasToolBox():
    params={}
    current_group=None
    def setupConfig(self,path):
        self.params["configDir"]=pathlib.Path(path)
        self.params["configExperiment"] = self.params["configDir"].joinpath('design_Sophie.yaml')
        self.params["configCriteria"] = self.params["configDir"].joinpath('criteria_list.yaml')

        self.config_design = loadConfigExperiment(self.params["configExperiment"])
        self.config_criterias = loadConfigCriterias(self.params["configCriteria"])

        self.setupDataRoot(path+"/../Enregistrements")

    def reloadConfig(self):
        self.config_criterias = loadConfigCriterias(self.params["configCriteria"])

    def setupDataRoot(self,path):
        self.params["dataPath"] = pathlib.Path(path)
        self.params["profilingFile_Sophie"] = self.params["dataPath"].joinpath(
            'traits_dfs_df.csv')  # loading Sophie's prearranged dataset (v0)
        self.metafile_df = createMetaFile(self.params["dataPath"], self.params["configExperiment"], isDataSophie=True)
        self.profiling_df, self.metafile_df = importProfilingDataSophie(self.metafile_df, self.params["profilingFile_Sophie"])
        self.current_profiling_df= self.profiling_df
        self.profiling_summary=self.profiling_df.describe(include="all")

    def updateProfilingDf(self,newdf):
        self.current_profiling_df = newdf
        self.profiling_summary = self.current_profiling_df.describe(include="all")

    def resetFilter(self):
        self.current_profiling_df = self.profiling_df
        self.profiling_summary = self.current_profiling_df.describe(include="all")

    def changeGroup(self,categorie,item):
        self.current_group=(categorie,item)

    def resetGroup(self):
        self.current_group=None

    def updateMetaFileMask(self):
        data_sel = self.current_profiling_df.droplevel(axis='columns', level='profiling_categories')
        metafile_mask = (self.metafile_df.subject_id.isin(data_sel["subject_id"])) & (self.metafile_df.condition.isin(data_sel["condition"]))
        metafile_sel = self.metafile_df.loc[metafile_mask].squeeze()
        self.data=[]
        for i in metafile_sel.index:
            row=[]
            row.append(metafile_sel.subject_id[i])
            row.append(metafile_sel.condition[i])
            objectsData, objectsFormat = importDataFile(metafile_sel.file_objectsData[i],
                                                          metafile_sel.file_objectsFormat[i], data_type='objects')
            row.append(objectsData)
            row.append(objectsFormat)
            actionsData, actionsFormat = importDataFile(metafile_sel.file_actionsData[i],
                                                          metafile_sel.file_actionsFormat[i], data_type='actions')
            row.append(actionsData)
            row.append(actionsFormat)
            self.data.append(row)
