import torch
import torch.nn as nn

from modules import ResNet50
from dataset import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

num_epochs = 30

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        running_correct += (predicted == labels).sum().item()

        running_total += labels.size(0)

    epoch_loss = (
        running_loss / len(train_loader.dataset)
    )

    epoch_accuracy = (
        running_correct / running_total
    )

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += (loss.item() * images.size(0))

            _, predicted = torch.max(outputs, 1)

            val_correct += (predicted == labels).sum().item()

            val_total += labels.size(0)

    val_epoch_loss = (val_loss / len(val_loader.dataset))

    val_accuracy = (val_correct / val_total)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_accuracy:.4f} - Val Loss: {val_epoch_loss:.4f} - Val Accuracy: {val_accuracy:.4f}"
    )
