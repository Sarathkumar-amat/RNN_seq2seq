import torch.nn as nn
import random
import torch

def make_rnn(cell_type,input_size,hidden_size,num_layers,batch_first=True):
    if cell_type=="rnn":
        return nn.RNN(input_size,hidden_size,num_layers=num_layers,batch_first=batch_first)
    elif cell_type=="gru":
        return nn.GRU(input_size,hidden_size,num_layers=num_layers,batch_first=batch_first)
    elif cell_type=="lstm":
        return nn.LSTM(input_size,hidden_size,num_layers=num_layers,batch_first=batch_first)
    else:
        raise ValueError("cell_type must be 'rnn', 'gru', or 'lstm'")

class Encoder(nn.Module):
    def __init__(self,cell_type,vocab_size, embed_dim,hidden_size,num_layers,pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.rnn = make_rnn(cell_type,embed_dim,hidden_size,num_layers,batch_first=True)
        self.cell_type = cell_type
    def forward(self,src,src_len=None):
        x=self.embedding(src)
        outputs,hidden = self.rnn(x)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self,cell_type,vocab_size,embed_dim,hidden_size,num_layers,pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.rnn = make_rnn(cell_type,embed_dim,hidden_size,num_layers,batch_first=True)
        self.out = nn.Linear(hidden_size,vocab_size)
        self.cell_type = cell_type
    
    def forward(self,y_in,hidden,tgt_len=None):
        x = self.embedding(y_in)
        outputs,hidden = self.rnn(x,hidden)
        logits = self.out(outputs)
        return logits,hidden

class translit_Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,cell_type):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type
    def forward(self,src,src_len,dec_in,teacher_forcing_ratio=1.0):
        B,T_dec = dec_in.size()
        _,enc_hidden = self.encoder(src,src_len)

        hidden = enc_hidden

        if teacher_forcing_ratio>=1.0:
            logits,_ = self.decoder(dec_in,enc_hidden)
            return logits
        
        logits_all=[]
        h = hidden
        y_t=dec_in[:,:1] # shape of dec_in = [B,_len_target_sequence-1] --> not considering the end token
        # only the first elements in the batch
        for t in range(1,T_dec+1):
            logit_t,h  = self.decoder(y_t,h)
            hidden=h
            logits_all.append(logit_t) # (B,1,V)
            teacher = (random.random() < teacher_forcing_ratio) and (t < T_dec)

            if teacher:
                y_t=dec_in[:,t:t+1] 
            else:
                y_t=logit_t.argmax(dim=-1) # (B,1)
        return torch.cat(logits_all,dim=1) # (B, T_dec, V)

