import pandas as pd

# adjust path/sep/column names to your files
def build_data_df():
    df_train = pd.read_csv("/home/sarath_25/DATASETS/dakshina_translit/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv", sep="\t", header=None, names=["native", "roman","freq"])
    df_dev   = pd.read_csv("/home/sarath_25/DATASETS/dakshina_translit/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv",   sep="\t", header=None, names=["native", "roman","freq"])

# strip and drop empties
    for d in (df_train, df_dev):
        d["roman"]  = d["roman"].astype(str).str.strip()
        d["native"] = d["native"].astype(str).str.strip()
        d.dropna(inplace=True)
        d = d[(d["roman"] != "") & (d["native"] != "")]
        
    print(df_train.head(), len(df_train), len(df_dev))
    
    return df_train, df_dev

def build_char_vocab(texts):
    PAD, SOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
    specials=(PAD, SOS, EOS, UNK)
    chars = set()
    for t in texts:
        chars.update(list(t))
    # stable order
    base = list(sorted(chars))
    itos = list(specials) + base
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos

df_train, df_dev = build_char_vocab()
src_stoi, src_itos = build_char_vocab(df_train["roman"])
tgt_stoi, tgt_itos = build_char_vocab(df_train["native"])

# pad_idx = src_stoi[PAD]   # same index exists in both; we’ll also grab target pad below
# tgt_pad_idx = tgt_stoi[PAD]
# tgt_sos = tgt_stoi[SOS]
# tgt_eos = tgt_stoi[EOS]

# print(src_itos)
# print(src_stoi)
# len(src_itos), len(tgt_itos)