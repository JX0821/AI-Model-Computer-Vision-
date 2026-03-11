import os
import csv
from tqdm import tqdm
import torch
import argparse
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F


class MiniPlaces(Dataset):
    def __init__(self, root_dir, split, transform=None, label_dict=None):
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.filenames = []
        self.labels = []
        
        self.label_dict = label_dict if label_dict is not None else {}

        with open(os.path.join(self.root_dir, self.split + '.txt')) as r:
            lines = r.readlines()
            for line in lines:
                line = line.split()
                self.filenames.append(line[0])
                if split == 'test':
                    label = line[0]
                else:
                    label = int(line[1])
                self.labels.append(label)
                if split == 'train':
                    text_label = line[0].split('/')[2]
                    self.label_dict[label] = text_label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(
                Image.open(os.path.join(self.root_dir, "images", self.filenames[idx])))
        else:
            image = Image.open(os.path.join(self.root_dir, "images", self.filenames[idx]))
        label = self.labels[idx]
        return image, label    


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1), nn.ReLU(),
            nn.Conv2d(c//r, c, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, i, o, dn=False):
        super().__init__()
        s = 2 if dn else 1
        self.conv = nn.Sequential(
            nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), SEBlock(o)
        )
        self.sc = nn.Identity() if (i==o and not dn) else \
                  nn.Sequential(nn.Conv2d(i, o, 1, s), nn.BatchNorm2d(o))
    
    def forward(self, x):
        return F.relu(self.conv(x) + self.sc(x))


class MyConv(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        self.l1 = ResidualBlock(64, 128, True)
        self.l2 = ResidualBlock(128, 256, True)
        self.l3 = ResidualBlock(256, 512, True)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_intermediate=False):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.fc(self.pool(x))


def evaluate(model, test_loader, criterion, device):
    model.eval()
    
    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(inputs)

            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)

    avg_loss = total_loss / num_samples
    accuracy = num_correct / num_samples
    
    return avg_loss, accuracy


def train(model, train_loader, val_loader, optimizer, criterion, device, scheduler, num_epochs):
    best_val_accuracy = 0.0
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        
        total_train_loss = 0.0
        num_train_samples = 0

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch+1}/{num_epochs}',
                  position=0,
                  leave=True) as pbar:
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * len(inputs)
                num_train_samples += len(inputs)

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            avg_val_loss, accuracy = evaluate(model, val_loader, criterion, device)
            avg_train_loss = total_train_loss / num_train_samples

            scheduler.step()

            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_accuracy': best_val_accuracy
                }, 'model.ckpt')

            print(f'Epoch: {epoch+1}: Train Loss = {avg_train_loss:.4f}; '
                  f'Val Loss = {avg_val_loss:.4f}, Val Acc = {accuracy:.4f}, '
                  f'Best Acc = {best_val_accuracy:.4f}')


def test(model, test_loader, device):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        all_preds = []

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            _, predictions = torch.max(logits, dim=1)
            preds = list(zip(labels, predictions.tolist()))
            all_preds.extend(preds)
    
    return all_preds


def main(args):
    # ImageNet statistical values
    image_net_mean = [0.485, 0.456, 0.406]
    image_net_std = [0.229, 0.224, 0.225]
    
    # 优化后的训练数据增强 - 适合场景分类
    # augementation suitable for scene classification
    image_size = 224
    # data_transform = transforms.Compose([
    #     transforms.Resize((image_size+16, image_size+16)),  # resize to larger size
    #     transforms.RandomCrop(128),  
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(degrees=10), 
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #     transforms.RandomGrayscale(p=0.1),  
    #     transforms.ToTensor(),
    #     transforms.Normalize(image_net_mean, image_net_std),
    #     transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))  
    # ])

  
    # val_transform = transforms.Compose([
    #     transforms.Resize((image_size, image_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(image_net_mean, image_net_std)
    # ])

    data_transform = transforms.Compose([
    transforms.Resize((image_size+16, image_size+16)),
    transforms.RandomResizedCrop(image_size, scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    data_root = 'data'
    
    # create datset
    miniplaces_train = MiniPlaces(data_root, split='train', transform=data_transform)
    miniplaces_val = MiniPlaces(data_root, split='val', transform=val_transform,
                                label_dict=miniplaces_train.label_dict)

    # hyperparameters
    batch_size = 32
    num_workers = 2
    num_epochs = 70  # number of epochs
    
    # create DataLoader
    # train_loader = DataLoader(miniplaces_train, batch_size=batch_size,
    #                           num_workers=num_workers, shuffle=True, pin_memory=True)
    # val_loader = DataLoader(miniplaces_val, batch_size=batch_size,
    #                         num_workers=num_workers, shuffle=False, pin_memory=True)

    train_loader = DataLoader(miniplaces_train,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(miniplaces_val,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            drop_last=False,
                            persistent_workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = MyConv(num_classes=len(miniplaces_train.label_dict))
    print(len(miniplaces_train.label_dict))
    
    # AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,  # initial lr
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # Label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Cosine Annealing to adjust lr
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # initial restartcycle
        T_mult=2,  
        eta_min=1e-6  # lowest lr
    )

    if not args.test:
        print(f'Starting training for {num_epochs} epochs...')
        train(model, train_loader, val_loader, optimizer, criterion, device, scheduler, num_epochs)
    else:
        miniplaces_test = MiniPlaces(data_root, split='test', transform=val_transform)
        test_loader = DataLoader(miniplaces_test, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)
        
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} "
              f"with accuracy {checkpoint.get('best_accuracy', 'unknown'):.4f}")
        
        preds = test(model, test_loader, device)
        write_predictions(preds, 'predictions.csv')
        print('Predictions saved to predictions.csv')


def write_predictions(preds, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for im, pred in preds:
            writer.writerow((im, pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--checkpoint', default='model.ckpt', help='Path to checkpoint')
    args, _ = parser.parse_known_args()
    main(args)