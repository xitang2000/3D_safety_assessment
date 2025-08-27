import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

ROOT_DIR = "/dataset/output/path/"
MODEL_PATH = "/your/model/path/model.pth"
TEST_LIST = os.path.join(ROOT_DIR, "test_list.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

class EmbeddingDataset(Dataset):
    def __init__(self, sample_list):
        self.samples = []
        for name in sample_list:
            embed_path = os.path.join(ROOT_DIR, name, "concat_embed.npy")
            if os.path.exists(embed_path):
                embedding = np.load(embed_path).astype(np.float32)
                label = int(name.split("_")[0])
                self.samples.append((embedding, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        return torch.tensor(data), torch.tensor(label, dtype=torch.float32)

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

def compute_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    total_pairs = len(pos) * len(neg)
    count = 0.0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    auc = count / total_pairs
    return auc


def load_names(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def evaluate():
    test_list = load_names(TEST_LIST)
    test_loader = DataLoader(EmbeddingDataset(test_list), batch_size=64, shuffle=False)

    model = ResidualMLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > THRESHOLD).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs)


    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    auc = compute_auc(y_true, y_score)


    print(f" Accuracy : {acc:.4f}")
    print(f" AUC      : {auc:.4f}")

if __name__ == "__main__":
    evaluate()
