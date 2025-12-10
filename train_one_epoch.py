import torch.optim as optim
import torch

# device=next(model.parameters()).device
# optimizer = optim.Adam(model.parameters(),lr=3e-4)
# criterion=CrossEntropyLoss(ignore_index=tgt_pad_idx)

def train_one_epoch(model,loader,device,optimizer,criterion,tgt_pad_idx,clip=1.0,tfr=1.0):
    device=next(model.parameters()).device
    model.train()
    total_loss, total_tokens=0.0,0
    for src,src_len,dec_in,dec_out in loader:
        src,dec_in,dec_out=src.to(device),dec_in.to(device),dec_out.to(device)
        optimizer.zero_grad()
        logits=model(src,src_len,dec_in,teacher_forcing_ratio=tfr)
        B,T,V = logits.size()
        # print(B,T,V)
        # print(dec_out.shape)
        loss = criterion(logits.reshape(B*T,V),dec_out.reshape(B*T))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimizer.step()
        non_pad = (dec_out!=tgt_pad_idx).sum().item()
        total_loss+=loss.item()*non_pad
        total_tokens+=non_pad
    model.eval()
    
    return total_loss/max(1,total_tokens)
