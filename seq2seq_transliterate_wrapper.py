import torch.nn as nn
import random
import torch

def make_rnn(cell_type,input_size,hidden_size,num_layers,batch_first=True,dropout=0.0):
    if cell_type=="rnn":
        return nn.RNN(input_size,hidden_size,num_layers=num_layers,batch_first=batch_first,dropout=dropout)
    elif cell_type=="gru":
        return nn.GRU(input_size,hidden_size,num_layers=num_layers,batch_first=batch_first,dropout=dropout)
    elif cell_type=="lstm":
        return nn.LSTM(input_size,hidden_size,num_layers=num_layers,batch_first=batch_first,dropout=dropout)
    else:
        raise ValueError("cell_type must be 'rnn', 'gru', or 'lstm'")


def beam_search_decode(decoder, hidden, sos_idx, eos_idx,
                       max_len, beam_size, device):
        beams = [(torch.tensor([[sos_idx]], device=device), hidden, 0.0)]

        for _ in range(max_len):
            new_beams = []

            for seq, h, score in beams:
                logits, h_new = decoder(seq[:, -1:], h)
                log_probs = torch.log_softmax(logits[:, -1], dim=-1)

                topk_probs, topk_idx = log_probs.topk(beam_size)

                for k in range(beam_size):
                    next_tok = topk_idx[:, k].unsqueeze(1)
                    new_seq = torch.cat([seq, next_tok], dim=1)
                    new_score = score + topk_probs[:, k].item()
                    new_beams.append((new_seq, h_new, new_score))

            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]

            # stop if all beams ended
            if all(b[0][:, -1].item() == eos_idx for b in beams):
                break

        return beams[0][0]  # best sequence

class Encoder(nn.Module):
    def __init__(self,cell_type,vocab_size, embed_dim,hidden_size,num_layers,pad_idx,dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.rnn = make_rnn(cell_type,embed_dim,hidden_size,num_layers,batch_first=True,dropout=dropout if num_layers > 1 else 0.0)
        self.cell_type = cell_type
    def forward(self,src,src_len=None):
        x=self.embedding(src)
        x = self.emb_dropout(x)
        outputs,hidden = self.rnn(x)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self,cell_type,vocab_size,embed_dim,hidden_size,num_layers,pad_idx,dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.rnn = make_rnn(cell_type,embed_dim,hidden_size,num_layers,batch_first=True,dropout=dropout if num_layers > 1 else 0.0)
        self.out_dropout = nn.Dropout(dropout)
        # self.out = nn.Linear(hidden_size, vocab_size)
        self.out = nn.Linear(hidden_size,vocab_size)
        self.cell_type = cell_type
    
    def forward(self,y_in,hidden,tgt_len=None):
        x = self.embedding(y_in)
        x = self.emb_dropout(x)
        outputs,hidden = self.rnn(x,hidden)
        outputs = self.out_dropout(outputs)
        logits = self.out(outputs)
        return logits,hidden

class translit_Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, cell_type):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type

    def _init_decoder_hidden(self, enc_hidden):
        dec_layers = self.decoder.rnn.num_layers
        if isinstance(enc_hidden, tuple):
            h, c = enc_hidden
            h = h[-1:].repeat(dec_layers, 1, 1)
            c = c[-1:].repeat(dec_layers, 1, 1)
            return (h, c)
        else:
            return enc_hidden[-1:].repeat(dec_layers, 1, 1)

    def forward(self, src, src_len, dec_in,
                teacher_forcing_ratio=1.0,
                beam_size=1,
                sos_idx=None,
                eos_idx=None):

        device = src.device
        B, T_dec = dec_in.size()

        _, enc_hidden = self.encoder(src, src_len)
        hidden = self._init_decoder_hidden(enc_hidden)

        if teacher_forcing_ratio >= 1.0 or beam_size == 1:
            logits, _ = self.decoder(dec_in, hidden)
            return logits

        assert sos_idx is not None and eos_idx is not None

        outputs = []
        for b in range(B):
            seq = beam_search_decode(
                self.decoder,
                hidden,
                sos_idx,
                eos_idx,
                max_len=T_dec,
                beam_size=beam_size,
                device=device
            )
            outputs.append(seq)

        return outputs
    
        # logits_all=[]
        # h = hidden
        # y_t=dec_in[:,:1] # shape of dec_in = [B,_len_target_sequence-1] --> not considering the end token
        # # only the first elements in the batch
        # for t in range(1,T_dec+1):
        #     logit_t,h  = self.decoder(y_t,h)
        #     hidden=h
        #     logits_all.append(logit_t) # (B,1,V)
        #     teacher = (random.random() < teacher_forcing_ratio) and (t < T_dec)

        #     if teacher:
        #         y_t=dec_in[:,t:t+1] 
        #     else:
        #         y_t=logit_t.argmax(dim=-1) # (B,1)
        # return torch.cat(logits_all,dim=1) # (B, T_dec, V)

