import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# -------------------------------
# Model Definitions
# -------------------------------

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        input_dim = latent_dim + num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(gen_input)
        return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        input_dim = num_classes + int(np.prod(img_shape))
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), -1)
        return self.model(d_in)


class Classifier(nn.Module):
    def __init__(self, num_classes, img_channels=1):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (img_shape[1]//4) * (img_shape[2]//4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# -------------------------------
# Utility Functions
# -------------------------------

def get_dataset(name, root='../data'):
    if name.lower() in ['emnist', 'fashionmnist']:
        transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        if name.lower() == 'emnist':
            return datasets.EMNIST(os.path.join(root, 'emnist'), split='balanced', train=True, download=True, transform=transform), datasets.EMNIST(os.path.join(root, 'emnist'), split='balanced', train=False, download=True, transform=transform), 26
        else:
            return datasets.FashionMNIST(root, train=True, download=True, transform=transform), datasets.FashionMNIST(root, train=False, download=True, transform=transform), 10

    elif name.lower() in ['cifar10', 'svhn']:
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
        if name.lower() == 'cifar10':
            return datasets.CIFAR10(root, train=True, download=True, transform=transform), datasets.CIFAR10(root, train=False, download=True, transform=transform), 10
        else:
            return datasets.SVHN(root, split='train', download=True, transform=transform), datasets.SVHN(root, split='test', download=True, transform=transform), 10
    else:
        raise ValueError(f"Unknown dataset {name}")


def partition_indices(dataset, num_clients):
    # IID partition
    num_items = len(dataset)
    indices = np.random.permutation(num_items)
    return np.array_split(indices, num_clients)

# -------------------------------
# Federated One-Shot
# -------------------------------

def train_local_gan(dataloader, latent_dim, num_classes, img_shape, device, epochs=5):
    G = Generator(latent_dim, num_classes, img_shape).to(device)
    D = Discriminator(num_classes, img_shape).to(device)
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=2e-4)
    optimizer_D = optim.Adam(D.parameters(), lr=2e-4)

    for epoch in range(epochs):
        for imgs, labels in dataloader:
            batch_size = imgs.size(0)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Train G
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_imgs = G(z, gen_labels)
            g_loss = criterion(D(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train D
            optimizer_D.zero_grad()
            real_loss = criterion(D(real_imgs, labels), valid)
            fake_loss = criterion(D(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

    return G


def generate_synthetic(G_clients, latent_dim, num_classes, samples_per_class, device):
    synth_imgs = []
    synth_labels = []
    for cls in range(num_classes):
        z = torch.randn(samples_per_class, latent_dim, device=device)
        labels = torch.full((samples_per_class,), cls, dtype=torch.long, device=device)
        # average generation across clients
        imgs_sum = 0
        for G in G_clients:
            G.eval()
            with torch.no_grad():
                imgs_sum += G(z, labels)
        imgs = (imgs_sum / len(G_clients)).cpu()
        synth_imgs.append(imgs)
        synth_labels.append(labels.cpu())
    return torch.cat(synth_imgs), torch.cat(synth_labels)


def train_classifier(train_imgs, train_labels, num_classes, img_channels, device, epochs=10, batch_size=128):
    clf = Classifier(num_classes, img_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(train_imgs, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = clf(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return clf


def evaluate(clf, test_loader, device):
    clf.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = clf(imgs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    # multiclass ROC AUC (one-vs-rest)
    try:
        auc = roc_auc_score(pd.get_dummies(all_labels), all_probs, average='macro')
    except ValueError:
        auc = float('nan')
    return acc, auc

# -------------------------------
# Main Federated Routine
# -------------------------------

def federated_one_shot(dataset_name, num_clients=5, local_epochs=5, synth_per_class=100, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    train_set, test_set, num_classes = get_dataset(dataset_name)
    global img_shape
    if hasattr(train_set, 'data'):
        # get shape from first sample
        sample, _ = train_set[0]
        img_shape = sample.shape
    else:
        sample, _ = train_set[0]
        img_shape = sample.shape

    # Partition data among clients
    client_idxs = partition_indices(train_set, num_clients)
    G_clients = []
    for i, idxs in enumerate(client_idxs):
        logging.info(f"Training client {i} GAN on {len(idxs)} samples...")
        subset = Subset(train_set, idxs)
        dataloader = DataLoader(subset, batch_size=64, shuffle=True)
        G = train_local_gan(dataloader, latent_dim=100, num_classes=num_classes, img_shape=img_shape, device=device, epochs=local_epochs)
        G_clients.append(G)

    # Server: generate synthetic data
    logging.info("Generating synthetic data at server...")
    synth_imgs, synth_labels = generate_synthetic(G_clients, latent_dim=100, num_classes=num_classes, samples_per_class=synth_per_class, device=device)

    # Train global classifier
    logging.info("Training global classifier on synthetic data...")
    img_channels = img_shape[0]
    clf = train_classifier(synth_imgs, synth_labels, num_classes, img_channels, device, epochs=10)

    # Evaluate on global test set
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    acc, auc = evaluate(clf, test_loader, device)
    logging.info(f"Final Test Accuracy: {acc:.4f}, Test AUROC: {auc:.4f}")

    # Save metrics
    metrics = pd.DataFrame([{'dataset': dataset_name, 'accuracy': acc, 'auc': auc}])
    out_file = f"metrics_{dataset_name}.csv"
    metrics.to_csv(out_file, index=False)
    logging.info(f"Metrics saved to {out_file}")

if __name__ == '__main__':
    os.makedirs('./data', exist_ok=True)
    for ds in ['EMNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']:
        federated_one_shot(ds)
