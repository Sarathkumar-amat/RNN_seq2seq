def encode(texts,stoi,add_sos_eos=False,sos=False, eos=False, UNK="<unk>"):
    ids = [stoi.get(ch,stoi[UNK])  for ch in texts]
    if add_sos_eos:
        ids = ([sos] if sos is not None else []) + ids + ([eos] if eos is not None else [])
    return ids