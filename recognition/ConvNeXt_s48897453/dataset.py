from pathlib import Path
import random
import time

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ADNI" / "AD_NC" / "train"

def get_patient_id(image_path):
    return image_path.stem.split("_")[0]


def build_patient_split(data_dir, validation_ratio=0.2, seed=42):
    samples = []

    class_map = {
        "AD": 0,
        "NC": 1
    }

    for class_name, label in class_map.items():
        class_dir = data_dir / class_name

        for image_path in class_dir.glob("*.jpeg"):
            patient_id = get_patient_id(image_path)

            samples.append({
                "path": image_path,
                "label": label,
                "patient_id": patient_id
            })

    patient_ids = sorted(set(sample["patient_id"] for sample in samples))

    rand = random.Random(seed)
    rand.shuffle(patient_ids)

    validation_count = int(len(patient_ids) * validation_ratio)

    validation_patients = set(patient_ids[:validation_count])
    train_patients = set(patient_ids[validation_count:])

    train_samples = [
        sample for sample in samples
        if sample["patient_id"] in train_patients
    ]

    validation_samples = [
        sample for sample in samples
        if sample["patient_id"] in validation_patients
    ]

    return train_samples, validation_samples, train_patients, validation_patients

class ADNIDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample["path"]).convert("RGB")
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

train_samples, val_samples, train_patients, val_patients = build_patient_split(DATA_DIR)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ADNIDataset(
    train_samples,
    transform=transform
)

val_dataset = ADNIDataset(
    val_samples,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

'''
assert train_patients.isdisjoint(val_patients), \
    "Patient leakage detected between training and validation sets"

print("Train patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Train images:", len(train_samples))
print("Validation images:", len(val_samples))
'''