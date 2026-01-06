from train_one_epoch import train_one_epoch
from torch.utils.data import Dataset, DataLoader

# from encode_collate import train_loader,dev_loader,src_itos,tgt_itos,src_stoi,tgt_stoi,PAD

from encode_collate import train_data, dev_data,src_itos,tgt_itos,src_stoi,tgt_stoi,collate_fn,PAD,SOS,EOS

from seq2seq_transliterate_wrapper import Encoder,Decoder,translit_Seq2Seq
from evaluate import evaluate,char_accuracy
from torch.utils.tensorboard import SummaryWriter
import wandb


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



# train_one_epoch(model, train_loader, device,optimizer,criterion,tgt_pad_idx,clip=1.0, tfr=1.0)  # start with 1.0

log_dir = "runs/rnn_seq2seq"
writer = SummaryWriter(log_dir)
wandb.init()
config = wandb.config

# wandb.init(
#     project="rnn-seq2seq",
#     name="baseline-rnn",
#     config={
#         "epochs": EPOCHS,
#         "batch_size": train_loader.batch_size,
#         "learning_rate": config.learning_rate,
#         "clip": 1.0,
#         "teacher_forcing": 1.0,
#         "model": "RNN-Seq2Seq"
#     }
# )
print("W&B config:", dict(config))
CELL_TYPE = config.cell_type
enc = Encoder(CELL_TYPE,SRC_V, config.embed, config.hidden, num_layers=config.enc_layers, pad_idx=src_stoi[PAD],dropout=config.dropout)
dec = Decoder(CELL_TYPE,TGT_V, config.embed, config.hidden, num_layers=config.dec_layers, pad_idx=tgt_stoi[PAD],dropout=config.dropout)
model = translit_Seq2Seq(enc, dec, cell_type=CELL_TYPE).to("cuda" if torch.cuda.is_available() else "cpu")


device = next(model.parameters()).device
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

train_loader = DataLoader(train_data,config.batch_size,shuffle=True,collate_fn=collate_fn)
dev_loader = DataLoader(dev_data,config.batch_size,shuffle=False,collate_fn=collate_fn)

for epoch in range(1, EPOCHS+1):
    train_loss = train_one_epoch(model,train_loader,device,optimizer,criterion,tgt_pad_idx,clip=1.0, tfr=1.0)  # start with 1.0
    val_loss   = evaluate(model, dev_loader,device,criterion,tgt_pad_idx=tgt_pad_idx,beam_size=config.beam_size,sos_idx=SOS,eos_idx=EOS)
    val_acc    = char_accuracy(model, dev_loader,device,criterion,tgt_pad_idx=tgt_pad_idx)


     # -------- W&B logging --------
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]["lr"]
    })
    # -------- TensorBoard logging --------
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val",   val_loss,   epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

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

        # upload checkpoint to W&B
        wandb.save(BEST_MODEL_PATH)

    else:
        epochs_no_improve += 1
        print(f"➤ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= PATIENCE:
            print("\nEARLY STOPPING TRIGGERED!")
            break
wandb.finish()
writer.close()
print(f"\nTraining complete. Best model saved to: {BEST_MODEL_PATH}")