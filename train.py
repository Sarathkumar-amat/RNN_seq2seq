from train_one_epoch import train_one_epoch
from encode_collate import train_loader,dev_loader,src_itos,tgt_itos,src_stoi,tgt_stoi,PAD
from seq2seq_transliterate_wrapper import Encoder,Decoder,translit_Seq2Seq
from evaluate import evaluate,char_accuracy

import torch.optim as optim
import torch
import torch.nn as nn

EPOCHS = 10

tgt_pad_idx = tgt_stoi[PAD]
SRC_V, TGT_V = len(src_itos), len(tgt_itos)
EMB, HID, LAYERS = 128, 256, 1
CELL = "gru"  # try "lstm" later

enc = Encoder(SRC_V, EMB, HID, num_layers=LAYERS, cell_type=CELL, pad_idx=src_stoi[PAD])
dec = Decoder(TGT_V, EMB, HID, num_layers=LAYERS, cell_type=CELL, pad_idx=tgt_stoi[PAD])
model = translit_Seq2Seq(enc, dec, cell_type=CELL).to("cuda" if torch.cuda.is_available() else "cpu")

device = next(model.parameters()).device
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(1, EPOCHS+1):
    train_loss = train_one_epoch(model, train_loader, clip=1.0, tfr=1.0)  # start with 1.0
    val_loss   = evaluate(model, dev_loader,device,tgt_pad_idx=tgt_pad_idx)
    val_acc    = char_accuracy(model, dev_loader,device,tgt_pad_idx=tgt_pad_idx)
    print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f}")
