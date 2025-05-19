import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt  # Importar matplotlib

# -------------------- Dataset Personalizado --------------------
class InundacionDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = mask.point(lambda p: 1 if p > 128 else 0)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

# -------------------- Transformaciones con Aumento de Datos --------------------
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class ToTensor(object):
    def __call__(self, img, mask):
        return transforms.ToTensor()(img), transforms.ToTensor()(mask)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        return transforms.Normalize(self.mean, self.std)(img), mask

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        return transforms.Resize(self.size)(img), transforms.Resize(self.size, interpolation=Image.NEAREST)(mask)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if torch.rand(1) < self.p:
            return transforms.functional.hflip(img), transforms.functional.hflip(mask)
        return img, mask

class RandomRotation(object):
    def __init__(self, degrees):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees) # Convertir a rango si es un solo número
        else:
            self.degrees = degrees

    def __call__(self, img, mask):
        angle = transforms.RandomRotation.get_params(self.degrees)
        return transforms.functional.rotate(img, angle), transforms.functional.rotate(mask, angle, interpolation=Image.NEAREST)

# -------------------- Bloques de la UNet --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# -------------------- Arquitectura UNet --------------------
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 256)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 512)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DoubleConv(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.conv4(self.down4(x4))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        output = self.sigmoid(logits)
        return output

# -------------------- Función de Dice Loss --------------------
def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def bce_dice_loss(pred, target, bce_weight=0.5):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return (bce * bce_weight) + (dice * (1 - bce_weight))

# -------------------- Cálculo de Métricas --------------------
def calcular_metricas(pred, target):
    pred_bin = (pred > 0.5).float()
    tp = torch.logical_and(target == 1, pred_bin == 1).sum()
    fp = torch.logical_and(target == 0, pred_bin == 1).sum()
    fn = torch.logical_and(target == 1, pred_bin == 0).sum()
    tn = torch.logical_and(target == 0, pred_bin == 0).sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)

    return precision.item(), recall.item(), f1.item(), accuracy.item(), dice.item(), iou.item()

# -------------------- Preparación de Datos y DataLoaders --------------------
#img_dir = 'path/to/your/images'
#mask_dir = 'path/to/your/masks'

# Defino los path de las imagenes y las mascaras
img_dir = '/home/224F8578gianfranco/UNET/Image/'
mask_dir = '/home/224F8578gianfranco/UNET/Mask/'
output_dir = '/home/224F8578gianfranco/UNET/'

os.makedirs(output_dir, exist_ok=True)

# Transformaciones para entrenamiento con aumento de datos
train_transforms = Compose([
    Resize((512, 512)),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformaciones para validación (sin aumento de datos)
val_transforms = Compose([
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Crear el Dataset completo
full_dataset = InundacionDataset(img_dir, mask_dir, transform=train_transforms)
val_dataset = InundacionDataset(img_dir, mask_dir, transform=val_transforms)

# Dividir el dataset en entrenamiento y validación
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# -------------------- Definición de Pérdida y Optimizador --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()
criterion = bce_dice_loss

# -------------------- Bucle de Entrenamiento con Early Stopping y Seguimiento de Métricas --------------------
num_epochs = 50
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0

train_losses = []
val_losses = []
val_ious = []
val_precisions = []
val_recalls = []
val_f1s = []
val_accuracies = []
val_dices = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}')
    train_losses.append(avg_train_loss)

    # Evaluación en el conjunto de validación
    model.eval()
    val_loss = 0
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_accuracies = []
    all_dices = []
    all_ious = []

    with torch.no_grad():
        for images_val, masks_val in val_loader:
            images_val = images_val.to(device)
            masks_val = masks_val.to(device)
            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, masks_val)
            val_loss += loss_val.item()

            precision, recall, f1, accuracy, dice, iou = calcular_metricas(outputs_val, masks_val)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            all_accuracies.append(accuracy)
            all_dices.append(dice)
            all_ious.append(iou)

    avg_val_loss = val_loss / len(val_loader)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)
    avg_accuracy = np.mean(all_accuracies)
    avg_dice = np.mean(all_dices)
    avg_iou = np.mean(all_ious)

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_iou:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-Score: {avg_f1:.4f}, Accuracy: {avg_accuracy:.4f}, Dice: {avg_dice:.4f}')
    val_losses.append(avg_val_loss)
    val_ious.append(avg_iou)
    val_precisions.append(avg_precision)
    val_recalls.append(avg_recall)
    val_f1s.append(avg_f1)
    val_accuracies.append(avg_accuracy)
    val_dices.append(avg_dice)

    # Early stopping (basado en la pérdida de validación)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), output_dir + 'Gemini_best_unet_inundacion.pth')
        print('Modelo guardado (mejor validación)')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f'Épocas sin mejora: {epochs_no_improve}')
        if epochs_no_improve == patience:
            print('Early stopping activado!')
            break

print('¡Entrenamiento finalizado!')

# -------------------- Visualización y Guardado de las Curvas --------------------
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs')
plt.savefig(os.path.join(output_dir, 'Gemini_loss_vs_epochs.png'))

plt.subplot(1, 3, 2)
plt.plot(val_ious, label='Validation IoU')
plt.plot(val_f1s, label='Validation F1-Score')
plt.plot(val_dices, label='Validation Dice')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.title('IoU, F1-Score, and Dice vs. Epochs')
plt.savefig(os.path.join(output_dir, 'Gemini_iou_f1_dice_vs_epochs.png'))

plt.subplot(1, 3, 3)
plt.plot(val_precisions, label='Validation Precision')
plt.plot(val_recalls, label='Validation Recall')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel