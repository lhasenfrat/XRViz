from data_proc import *
from questionnaire import *

class AnalysisToolBox():
    params={}
    current_group=None
    group_type = None

    def setupConfig(self,path):
        self.params = loadConfigPaths(pathlib.Path(path))
        self.config_design = loadConfigExperiment(self.params["configExperiment"])
        self.config_criterias = loadConfigCriterias(self.params["configCriteria"])

        self.setupDataRoot()

    def reloadConfig(self):
        self.config_criterias = loadConfigCriterias(self.params["configCriteria"])

    def setupDataRoot(self,):
        # loading Sophie's prearranged dataset (v0)
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
        self.group_type = str(self.current_profiling_df.dtypes[self.current_group])
    def resetGroup(self):
        self.current_group=None
        self.group_type=None

    def updateMetaFileMask(self):
        data_sel = self.current_profiling_df.droplevel(axis='columns', level='profiling_categories')
        metafile_mask = (self.metafile_df.subject_id.isin(data_sel["subject_id"])) & (self.metafile_df.condition.isin(data_sel["condition"]))
        metafile_sel = self.metafile_df.loc[metafile_mask].squeeze()
        self.data={}
        self.correspondance_table={}
        for i in metafile_sel.index:
            if metafile_sel.scene[i]=="Tutorial":
                continue
            if metafile_sel.profiling_index[i] not in self.correspondance_table:
                self.correspondance_table[metafile_sel.profiling_index[i]]=[]
            self.correspondance_table[metafile_sel.profiling_index[i]].append(i)
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

            self.data[i]=row
        self.computeActionMetrics()
    def computeActionMetrics(self):
        lengthlist=[]
        nbhoverlist=[]
        profilingdic={}
        maxlength=0
        maxnbhover=0
        for profiling_index in sorted(list(self.correspondance_table.keys())):
            length=0
            nbHover=0
            for meta_index in self.correspondance_table[profiling_index]:
                length+=float(self.data[meta_index][4].timestamp.tail(1))
                nbHover+=int(len(self.data[meta_index][4][self.data[meta_index][4].ActionId == 3])/2)
            lengthlist.append(length)
            maxlength=max(maxlength,length)
            nbhoverlist.append(nbHover)
            maxnbhover=max(maxnbhover,nbHover)

        self.current_profiling_df.loc[:,('actionMetrics','length')]=lengthlist
        self.current_profiling_df.loc[:,('actionMetrics','nbHover')]=nbhoverlist
        profilingdic['length']={'from':'0','to':maxlength,'type':'numeric_range'}
        profilingdic['nbHover']={'from':'0','to':maxnbhover,'type':'numeric_range'}

        self.config_criterias["PROFILING"]['actionMetrics']=profilingdic
    def computeObjectsMetrics(self):
        for i in self.current_profiling_df :
            print(i)