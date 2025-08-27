import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import argparse 


class EmbeddingDataset(Dataset):
    def __init__(self, sample_list, data_dir):
        self.samples = []
        self.labels = []
        self.data_dir = data_dir 

        for name in sample_list:
            embed_path = os.path.join(self.data_dir, name, "concat_embed.npy")
            if os.path.exists(embed_path):
                embedding = np.load(embed_path).astype(np.float32)
                label = int(name.split("_")[0])
                self.samples.append((embedding, label))
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        return torch.tensor(data), torch.tensor(label, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.final = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        residual = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = x + residual
        x = self.dropout(x)
        return self.final(x).squeeze(1)


def load_names(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_weighted_sampler(dataset):
    labels = torch.tensor(dataset.labels)
    class_counts = torch.bincount(labels.long())
    weights = len(dataset) / (2.0 * class_counts)
    sample_weights = weights[labels.long()]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)



def train_model(args):

    train_list_path = os.path.join(args.data_dir, "train_list.txt")
    test_list_path = os.path.join(args.data_dir, "test_list.txt")

    if not os.path.exists(train_list_path):
        raise FileNotFoundError(f"Training list file could not be found : {train_list_path}")
    if not os.path.exists(test_list_path):
        raise FileNotFoundError(f"Test list file could not be found : {test_list_path}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Making output directory: {args.output_dir}")

    train_list_names = load_names(train_list_path)
    test_list_names = load_names(test_list_path)

    train_dataset = EmbeddingDataset(train_list_names, args.data_dir)
    test_dataset = EmbeddingDataset(test_list_names, args.data_dir)

    if len(train_dataset) == 0:
        print(f"Warining: training dataset is empty. Please check documents in {train_list_path} and {args.data_dir} ")
        return
    if len(test_dataset) == 0:
        print(f"Warining: test dataset is empty. Please check documents in {test_list_path} and {args.data_dir}")
        test_loader = None
    else:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=get_weighted_sampler(train_dataset)
    )


    device = torch.device(args.device)
    model = ResidualMLP().to(device)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    patience_counter = args.patience

    print("\n--- Training start ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss, correct, total = 0, 0, 0
        if test_loader: 
            val_pbar = tqdm(test_loader, desc=f"Epoch {epoch}/{args.epochs} [Validate]", leave=False)
            with torch.no_grad():
                for x, y in val_pbar:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    val_loss += loss.item()
                    pred = (torch.sigmoid(out) > 0.5).float()
                    correct += (pred == y).sum().item()
                    total += len(y)
                    val_pbar.set_postfix(loss=loss.item())

            avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else 0
            acc = correct / total if total > 0 else 0
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = args.patience
                save_path = os.path.join(args.output_dir, "model_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f"Save best model to : {save_path}")
            else:
                patience_counter -= 1
                print(f"Loss did not improve, patience: {patience_counter}")
                if patience_counter == 0:
                    print("Early stop triggered")
                    break
        else:
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Skip (test dataset is empty) ")


    print("--- Training complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MLP model")

    parser.add_argument("--data_dir", type=str, default="/dataset/outputs/path/",
                        help="dataset root directory train_list.txt and test_list.txt.")
    parser.add_argument("--output_dir", type=str, default="./models/",
                        help="Save training model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size setting")
    parser.add_argument("--epochs", type=int, default=7,
                        help="Epochs setting")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate setting")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal Loss alpha ")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Training device ('cuda' or 'cpu')。")

    args = parser.parse_args()

    print("---  Training parameter ---")
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    print("----------------")

    train_model(args)
    
