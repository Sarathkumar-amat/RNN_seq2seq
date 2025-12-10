from train_one_epoch import train_one_epoch
from encode_collate import train_loader,dev_loader,src_itos,tgt_itos,src_stoi,tgt_stoi,PAD
from seq2seq_transliterate_wrapper import Encoder,Decoder,translit_Seq2Seq
from evaluate import evaluate,char_accuracy

import torch.optim as optim
import torch
import torch.nn as nn

EPOCHS = 30
PATIENCE = 5                       # stop after 5 epochs without improvement
MIN_DELTA = 1e-4                   # minimum improvement to consider as progress
BEST_MODEL_PATH = "best_model.pt"  # where to save best checkpoint

best_val_loss = float("inf")
epochs_no_improve = 0

train_losses = []
val_losses   = []
val_accs     = []

tgt_pad_idx = tgt_stoi[PAD]
SRC_V, TGT_V = len(src_itos), len(tgt_itos)
EMB, HID, LAYERS = 128, 256, 1
CELL = "gru"  # try "lstm" later

enc = Encoder(CELL,SRC_V, EMB, HID, num_layers=LAYERS, pad_idx=src_stoi[PAD])
dec = Decoder(CELL,TGT_V, EMB, HID, num_layers=LAYERS, pad_idx=tgt_stoi[PAD])
model = translit_Seq2Seq(enc, dec, cell_type=CELL).to("cuda" if torch.cuda.is_available() else "cpu")

device = next(model.parameters()).device
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

train_one_epoch(model, train_loader, device,optimizer,criterion,tgt_pad_idx,clip=1.0, tfr=1.0)  # start with 1.0
for epoch in range(1, EPOCHS+1):
    train_loss = train_one_epoch(model, train_loader, device,optimizer,criterion,tgt_pad_idx,clip=1.0, tfr=1.0)  # start with 1.0
    val_loss   = evaluate(model, dev_loader,device,criterion,tgt_pad_idx=tgt_pad_idx)
    val_acc    = char_accuracy(model, dev_loader,device,tgt_pad_idx=tgt_pad_idx)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f}")
    
    # -------- Early Stopping logic --------
    if val_loss + MIN_DELTA < best_val_loss:
        print(f"➤ Validation improved from {best_val_loss:.4f} → {val_loss:.4f}. Saving checkpoint.")
        best_val_loss = val_loss
        epochs_no_improve = 0

        # Save CHECKPOINT
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, BEST_MODEL_PATH)

    else:
        epochs_no_improve += 1
        print(f"➤ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= PATIENCE:
            print("\nEARLY STOPPING TRIGGERED!")
            break
print(f"\nTraining complete. Best model saved to: {BEST_MODEL_PATH}")