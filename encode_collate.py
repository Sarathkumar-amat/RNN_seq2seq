import torch
from torch.utils.data import Dataset, DataLoader
from load_data_build_vocab import build_data_df, build_char_vocab

df_train, df_dev = build_data_df()
src_stoi, src_itos = build_char_vocab(df_train["roman"])
tgt_stoi, tgt_itos = build_char_vocab(df_train["native"])
PAD, SOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"

print(f"vocab size input:{len(src_stoi)}")
print(f"vocab size target:{len(tgt_stoi)}")

def encode(texts,stoi,add_sos_eos=False,sos=False, eos=False, UNK="<unk>"):
    ids = [stoi.get(ch,stoi[UNK])  for ch in texts]
    if add_sos_eos:
        ids = ([sos] if sos is not None else []) + ids + ([eos] if eos is not None else [])
    return ids

class DakshinaDataset(Dataset):
    def __init__(self,df,src_stoi,tgt_stoi,tgt_sos,tgt_eos):
        self.src = df["roman"].tolist()
        self.tgt = df["native"].tolist()
        self.src_stoi = src_stoi
        self.tgt_stoi = tgt_stoi
        self.tgt_sos = tgt_sos
        self.tgt_eos = tgt_eos

    def __len__(self): return len(self.src)

    def __getitem__(self, i):
        src_ids = encode(self.src[i],self.src_stoi)
        tgt_ids = encode(self.tgt[i],self.tgt_stoi,add_sos_eos=True,sos=self.tgt_eos,eos=self.tgt_eos)
        return torch.tensor(src_ids,dtype=torch.long),torch.tensor(tgt_ids,dtype=torch.long)
       
def pad_batch(seqs,pad):
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs),max_len),pad,dtype=torch.long)
    lengths = torch.tensor([len(s) for s in seqs],dtype=torch.long)
    for i,s in enumerate(seqs):
        out[i,:len(s)]=s
    return out,lengths

def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_pad = src_stoi[PAD]
    tgt_pad = tgt_stoi[PAD]

    src_pad_batch, src_len = pad_batch(src_seqs,src_pad)
    tgt_pad_batch, tgt_len = pad_batch(tgt_seqs,tgt_pad)

    dec_in = tgt_pad_batch[:,:-1]
    dec_out = tgt_pad_batch[:,1:]

    return src_pad_batch, src_len, dec_in, dec_out

train_data = DakshinaDataset(df_train,src_stoi,tgt_stoi,tgt_stoi[SOS],tgt_stoi[EOS])
dev_data = DakshinaDataset(df_dev,src_stoi,tgt_stoi,tgt_stoi[SOS],tgt_stoi[EOS])

BATCH_SIZE = 64

# train_loader = DataLoader(train_data,BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
# dev_loader = DataLoader(dev_data,BATCH_SIZE,shuffle=False,collate_fn=collate_fn)

# batch = next(iter(train_loader))

# for x in batch:
#     print(type(x), x.shape)
