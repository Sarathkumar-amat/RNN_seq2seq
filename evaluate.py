
import torch

@torch.no_grad()
def evaluate(model, loader,device,criterion,tgt_pad_idx,beam_size,sos_idx,eos_idx):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, src_len, dec_in, dec_out in loader:
        src, dec_in, dec_out = src.to(device), dec_in.to(device), dec_out.to(device)
        logits = model(src, src_len, dec_in, teacher_forcing_ratio=0.0,beam_size=beam_size,sos_idx=sos_idx,eos_idx=eos_idx)
        # print(logits)
        B, T, V = logits.size()
        # print(B,T,V)
        # print(dec_out.shape)
        loss = criterion(logits.reshape(B*T, V), dec_out.reshape(B*T))
        non_pad = (dec_out != tgt_pad_idx).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    return total_loss / max(1, total_tokens)

@torch.no_grad()
def char_accuracy(model, loader,device,criterion,tgt_pad_idx):
    model.eval()
    correct, total = 0, 0
    for src, src_len, dec_in, dec_out in loader:
        src, dec_in, dec_out = src.to(device), dec_in.to(device), dec_out.to(device)
        logits = model(src, src_len, dec_in, teacher_forcing_ratio=0.0)
        preds = logits.argmax(-1)
        mask = (dec_out != tgt_pad_idx)
        correct += ((preds == dec_out) & mask).sum().item()
        total   += mask.sum().item()
    return correct / max(1, total)

