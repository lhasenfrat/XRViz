import pathlib

import numpy
import pandas as pd
import yaml

import data_io
configExperiment = pathlib.Path("./config/design_Sophie.yaml")
dataPath = pathlib.Path("./Enregistrements")
metafile_df = data_io.createMetaFile(dataPath, configExperiment, isDataSophie=True)

profiling_df,b = data_io.importProfilingDataSophie(metafile_df,"Enregistrements/traits_dfs_df.csv")
output={"PROFILING":{}}
for categorie in profiling_df.columns.levels[0]:
    if "item" in categorie:
        continue
    output["PROFILING"][categorie]={}
    for item in profiling_df[categorie].columns:
        if "completion" in item:
            continue
        current_item={}

        if str(profiling_df.dtypes[categorie][item])=="int64" or str(profiling_df.dtypes[categorie][item])=="float64":
            current_item["type"]="numeric_range"
            current_item["from"]=min(list(profiling_df[categorie][item]))
            current_item["to"]=max(list(profiling_df[categorie][item]))
        elif str(profiling_df.dtypes[categorie][item])=="object":
            current_item["type"]="string_list"
            current_item["list"]=list(set(profiling_df[categorie][item]))
        else:
            print(type(profiling_df.dtypes[categorie][item]))
        output["PROFILING"][categorie][item]=current_item


f = open("criteria_list.yaml", "w")
f.write(yaml.dump(output,allow_unicode=True))
f.close()