import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix)
from modules import ResNet50, ConvNeXt
from dataset import train_loader, val_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_ResNet50():
    model = ResNet50(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 30

    best_val_accuracy = 0.0
    best_epoch = 0

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

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

        epoch_loss = (running_loss / len(train_loader.dataset))

        epoch_accuracy = (running_correct / running_total)

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

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            torch.save(model.state_dict(), "resnet50_best.pth")

            print(f"Saved new best model at epoch {best_epoch} with validation accuracy {best_val_accuracy:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_accuracy:.4f} - Val Loss: {val_epoch_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

    print(f"Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}"
)

    epochs = range(1, num_epochs + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet50 Training and Validation Loss")
    plt.legend()
    plt.savefig("resnet50_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet50 Training and Validation Accuracy")
    plt.legend()
    plt.savefig("resnet50_accuracy.png")
    plt.close()

def train_ConvNeXt():
    model = ConvNeXt(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    num_epochs = 30

    best_val_accuracy = 0.0
    best_epoch = 0

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

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

            running_loss += (loss.item() * images.size(0))

            _, predicted = torch.max(outputs, 1)

            running_correct += (predicted == labels).sum().item()

            running_total += labels.size(0)

        epoch_loss = (running_loss / len(train_loader.dataset))

        epoch_accuracy = (running_correct / running_total)

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

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            torch.save(model.state_dict(),"convnext_best.pth")

            print(f"Saved new best ConvNeXt model at epoch {best_epoch} with validation accuracy {best_val_accuracy:.4f}"
            )

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_accuracy:.4f} - Val Loss: {val_epoch_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

    print(f"Best ConvNeXt validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}"
    )

    epochs = range(1, num_epochs + 1)
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ConvNeXt Training and Validation Loss")
    plt.legend()
    plt.savefig("convnext_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ConvNeXt Training and Validation Accuracy")
    plt.legend()
    plt.savefig("convnext_accuracy.png")
    plt.close()

def evaluate_model(model, data_loader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_confidences = []
    all_correct = []

    running_loss = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            correct = predicted == labels

            running_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

            confidence, predicted = torch.max(probabilities, dim=1)

            probabilities = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().tolist())

            all_predictions.extend(predicted.cpu().tolist())
            
            all_confidences.extend(confidence.cpu().tolist())
            
            all_correct.extend(correct.cpu().tolist())

            # NC is class 1 in your current mapping
            all_probabilities.extend(probabilities[:, 1].cpu().tolist())

    accuracy = (sum(pred == label for pred, label in zip(all_predictions, all_labels)) / len(all_labels))

    precision = precision_score(all_labels, all_predictions, average="macro")

    recall = recall_score( all_labels, all_predictions, average="macro")

    f1 = f1_score( all_labels, all_predictions, average="macro")

    precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)

    recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)

    auroc = roc_auc_score( all_labels, all_probabilities)

    cm = confusion_matrix( all_labels, all_predictions )

    average_loss = (running_loss / total_samples)

    return {"loss": average_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "precision_per_class": precision_per_class, 
            "recall_per_class": recall_per_class, "auroc": auroc, "confusion_matrix": cm, "labels": all_labels, "predictions": all_predictions, 
            "confidences": all_confidences, "correct": all_correct}

def analyse_confidence(results):

    correct_confidences = [confidence for confidence, correct in zip(results["confidences"], results["correct"]) if correct]

    incorrect_confidences = [confidence for confidence, correct in zip(results["confidences"],results["correct"]) if not correct]

    average_correct = (sum(correct_confidences) / len(correct_confidences))

    average_incorrect = (sum(incorrect_confidences) / len(incorrect_confidences))

    print(f"Average confidence when correct: {average_correct:.4f}")

    print(f"Average confidence when incorrect: {average_incorrect:.4f}")

if __name__ == "__main__":

    i = input("ResNet or ConvNeXt: ").strip().lower()

    if i == "resnet":
        train_ResNet50()

        model = ResNet50(num_classes=2).to(device)

        model.load_state_dict(torch.load("resnet50_best.pth", map_location=device))
    
    elif i == "convnext":
        
        train_ConvNeXt()
        
        model = ConvNeXt(num_classes=2).to(device)
        
        model.load_state_dict(torch.load("convnext_best.pth", map_location=device))

    else:
        print("Invalid choice. Enter 'resnet' or 'convnext'.")
        raise SystemExit

    results = evaluate_model(model, test_loader, device)

    analyse_confidence(results)

    print(f"\nBest {i} Validation Results")
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"Macro F1:  {results['f1']:.4f}")
    print(f"AUROC:     {results['auroc']:.4f}")
    print(f"AD Precision: {results['precision_per_class'][0]:.4f}")
    print(f"NC Precision: {results['precision_per_class'][1]:.4f}")
    print(f"AD Recall: {results['recall_per_class'][0]:.4f}")
    print(f"NC Recall: {results['recall_per_class'][1]:.4f}")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])
