import pandas as pd

# adjust path/sep/column names to your files
df_train = pd.read_csv("/home/sarath_25/DATASETS/dakshina_translit/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv", sep="\t", header=None, names=["native", "roman","freq"])
df_dev   = pd.read_csv("/home/sarath_25/DATASETS/dakshina_translit/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv",   sep="\t", header=None, names=["native", "roman","freq"])

# strip and drop empties
for d in (df_train, df_dev):
    d["roman"]  = d["roman"].astype(str).str.strip()
    d["native"] = d["native"].astype(str).str.strip()
    d.dropna(inplace=True)
    d = d[(d["roman"] != "") & (d["native"] != "")]
    
print(df_train.head(), len(df_train), len(df_dev))
