import pandas as pd

DFS_df = pd.read_csv("Enregistrements/Dispositional Flow Scale + TPI (réponses).csv")
DFS_df["Quel est votre nom de famille ?"]=DFS_df["Quel est votre nom de famille ?"].map(lambda x : x.capitalize())
DFS_df["Quel est votre prenom ?"]=DFS_df["Quel est votre prenom ?"].map(lambda x : x.capitalize())
DFS_df["Quel est votre nom de famille ?"]=DFS_df["Quel est votre nom de famille ?"].map(lambda x : x.strip())
DFS_df["Quel est votre prenom ?"]=DFS_df["Quel est votre prenom ?"].map(lambda x : x.strip())

print(DFS_df["Quel est votre nom de famille ?"])
factors = pd.read_csv("Enregistrements/Les Cinq Facteurs des Traits De Joueurs (réponses).csv")
factors["Quel est votre nom de famille ?"]=factors["Quel est votre nom de famille ?"].map(lambda x : x.capitalize())
factors["Quel est votre prenom ?"]=factors["Quel est votre prenom ?"].map(lambda x : x.capitalize())
factors["Quel est votre nom de famille ?"]=factors["Quel est votre nom de famille ?"].map(lambda x : x.strip())
factors["Quel est votre prenom ?"]=factors["Quel est votre prenom ?"].map(lambda x : x.strip())

print(factors["Quel est votre prenom ?"])

df_merged = pd.merge(DFS_df, factors, on=['Quel est votre nom de famille ?', 'Quel est votre prenom ?'], how='left')
df_merged.pop("Quel est votre nom de famille ?")
df_merged.pop("Quel est votre prenom ?")
df_merged.pop("Adresse e-mail")

df_merged.to_csv("Enregistrements/traits_dfs_df.csv")