    # Importamos todas las bibliotecas necesarias

import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

# Inicializamos el generador de numeros aleatorios
random.seed(1234)
torch.manual_seed(1234)

# Definimos la clase UNet
class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x =  self.conv(x)

        x = torch.softmax(x, dim=1)
        return x

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Probamos el modelo
input_image = torch.rand((1,3,512,512))
model = UNet(3,1)
output = model(input_image)
print(output.size())
# You should get torch.Size([1, 1, 512, 512]) as a result

# Defino los path de las imagenes y las mascaras
path_images = '/home/224F8578gianfranco/UNET/Image/'
path_masks = '/home/224F8578gianfranco/UNET/Mask/'
path_res = '/home/224F8578gianfranco/UNET/'

# Vamos a crear una clase para cargar las imagenes y las mascaras
class FloodDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform

        self.images = os.listdir(images_path)
        self.masks = os.listdir(masks_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = os.path.join(self.images_path, self.images[idx])
        mask_name = os.path.join(self.masks_path, self.masks[idx])

        image = Image.open(image_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    
# Definimos la variable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("GPU is available")
    num_workers = torch.cuda.device_count() * 4
    print(f"Number of workers: {num_workers}")

# Definimos algunos hiperparametros
Batch_Size = 8
Learning_Rate = 0.00001
Num_Epochs = 100

print("Hiperparametros:")
print(f"Batch Size: {Batch_Size}")
print(f"Learning Rate: {Learning_Rate}")
print(f"Num Epochs: {Num_Epochs}")
print(f"Device: {device}")
print(f"Num Workers: {num_workers}")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #transforms.Lambda(lambda x: x * 255),  # Convertir a rango [0, 255]
    #transforms.Lambda(lambda x: x.type(torch.uint8)),  # Convertir a uint8
    #transforms.Lambda(lambda x: x / 255),  # Volver a normalizar a rango [0, 1]
])

# Creamos el dataset
dataset = FloodDataset(images_path=path_images, masks_path=path_masks, transform=transform)

# Dividimos el dataset en train y test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1234))

# Creamos los dataloaders
#train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, num_workers=num_workers)
#test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False, num_workers=num_workers)
train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)

# Vamos a mostrar el numero de batches
print(f"Numero de batches de train: {len(train_loader)}")
print(f"Numero de batches de test: {len(test_loader)}")

# Definimos el modelo, el optimizador y la funcion de perdida y cargamos el modelo en la GPU
model = UNet(n_channels=3, n_classes=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=Learning_Rate)
criterion = nn.BCEWithLogitsLoss()

torch.cuda.empty_cache()

# Vamos a definir un bloque de entrenamiento y evaluacion del modelo guaradando los datos para las graficas de perdida y accuracy utilizando el IOU
def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs):
    train_losses = []
    test_losses = []
    iou_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Evaluamos el modelo
        model.eval()
        running_loss = 0.0
        iou_score_epoch = 0.00

        with torch.no_grad():
            for images, masks in tqdm(test_loader):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                running_loss += loss.item()

                # Calculamos el IOU
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()
                intersection = (outputs * masks).sum()
                union = outputs.sum() + masks.sum() - intersection
                iou_score = intersection / union
                iou_score_epoch += iou_score.item()

        # Guardamos la perdida y el IOU    
        test_loss = running_loss / len(test_loader)
        test_losses.append(test_loss)
        iou_scores.append(iou_score_epoch / len(test_loader))

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, IOU: {iou_scores[-1]:.4f}")

    return train_losses, test_losses, iou_scores

# Vamos a entrenar el modelo
train_losses, test_losses, iou_scores = train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=Num_Epochs)

# Vamos a mostrar y guardar la curva de loss vs. epochs
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs')
plt.savefig(path_res + 'loss_vs_epochs.png')
plt.show()

# Vamos a mostrar y guardar la curva de IOU vs. epochs
plt.figure(figsize=(12, 5))
plt.plot(iou_scores, label='IOU Score')
plt.xlabel('Epochs')
plt.ylabel('IOU Score')
plt.legend()
plt.title('IOU Score vs Epochs')
plt.savefig(path_res + 'iou_vs_epochs.png')
plt.show()

