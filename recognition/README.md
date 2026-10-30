# Alzheimer’s Disease Classification Between ConvNeXt And ResNet50

This report investigates CNN architectures for binary classification of Alzheimer’s Disease using the ADNI image dataset.

The primary model is a custom implementation of ConvNeXt, with ResNet50 implemented as a baseline for comparison.

ConvNeXt is a modern CNN that build on top of the classic ResNet. It incorporates the architectural advancements of Vision Transformers. It modernised ResNet by using the same design choices, macro-structures, and training techniques that are used in Vision Transformers. Using this architectural design, it allowed for a CNN that matches transformer accuracy, whilst keeping the efficiency, simplicity and strong inductive bias of a CNN.

The baseline model is ResNet50 [1], a deep CNN built around residual connections. ResNet50 uses bottleneck blocks 3 layers of convolutions are used. A beginning 1x1 convolution that reduces the channel count, a 3x3 convolution to perform spatial processing and a final 1x1 convolution to expand the channels again. They all use BatchNorm and ReLu activations throughout. It begins with a 7x7 stride-2 convolution and max pooling with the blocks [3, 4, 6, 3]. Resolution is halved and channels double widths at each stage. It ends with adaptive average pooling and a linear classifier. ConvNeXt keeps a similar macro-structure of four stages and residual blocks, but follows the design choices of Vision Transformers. A 4x4 stride 4 stem with a 7x7 depthwise convolution instead of the 3x3 convolution. Am inverted bottle neck expands the channels inside the block instead of narrowing them, GELU activation block, LayerNorm instead of BatchNorm, and other training aids such as layer scale and stochastic depth. 

### Architecture for ResNet50 and ConvNeXt

![Architecture](Images/architecture.png)

#### ConvNeXt
```
class ConvNeXt(nn.Module):
	def __init__(
		self, 
		in_channels = 1,
		num_classes=2, 
		depths=(3, 3, 9, 3), 
		dims=(96, 192, 384, 768), 
		drop_path_rate=0.1
	):
```

## Feasbility Report
### User Need, Scope & Acceptance Criteria
The intended users for this would be clinical or research reviewers assessing Alzheimer’s screening. The prototype would classify ADNI brain MRI images into AD vs CN. It will also produce confidence scores that may help identify uncertain cases for further review. This project will compare ConvNeXt against a CNN baseline using the same data splits. 
Acceptance Criteria:
•	At least a 0.80 test accuracy
•	Continue patient-independent train, validation and test splits
•	Compare the difference between ConvNeXt with A CNN using accuracy, precision, recall, etc.
•	Measure parameter count, GPU memory and inference time.

### Model Choice & Course Concepts
ConvNeXt was selected to be used to compare against the CNN architecture. The ConvNeXt model is listed as a Hard Difficulty model. A basic CNN will be used as a baseline to compare the ConvNeXt and evaluated under identical conditions. Convolutional feature extraction, overfitting, classification metrics, and other course related content will be useful when comparting baseline models using the same data.

### Preliminary Feasibility Evidence
The ADNI dataset contained 30520 MRI slices with the training set containing 21520 slices (10400 AD and 11120 CN), while the test set contained 9000 slices (4460 AD and 4540 CN).  A simple CNN was ran with 23714 trainable parameters and completed a 20-batch CUDA smoke test. The average loss was 0.6938, with training times of 0.93 seconds, and a peak allocated GPU memory of around 0.190GB

### Risks, Budget & Fallback
Some risks include overfitting and higher computational cost of ConvNeXt. Validation checks, augmentation, regularisation and potential early stopping might be used to prevent these risks.

The first task is to train a baseline CNN model for precision, accuracy, recall, etc, before running ConvNeXt on the same splits. If ConvNeXt exceeds available resources then batch size, input resolution or model size will be reduced.

Patient IDs will be used to ensure that slices from the same patient do not appear across training, validation and test set. The training and test datasets will also be checked for potential patient-level data leakage to ensure that slices from the same patient are not present across multiple data splits.

## Project Structure
### Dataset
Dataset.py is responsible for locating the ADNI dataset, extracting patient IDs, constructing patient-level training and validation splits, loading the supplied test set, image preprocessing, data augmentation, creation of PyTorch Dataset and DataLoader objects.

### Modules
Modules.py  implements ResNet50, ResNet50 Bottleneck Block, ConvNeXt block, LayerNorms, Stochastic Depth, and ConvNeXt.

### Train
Train.py handles training and evaluating the model; implementing optimisation, validation, checkpoint selection, early stopping, metric calculations, resource measurement.

### Predict
predict.py uses the saved model to use on individual MRI images.

## Dataset
The dataset was downloaded into the directory /recognition/ConvNeXt_s4889745/ADNI.

The dataset contains two classes:
- AD = 0
- NC = 1

The data was separated into:
ADNI/
└── AD_NC/
    ├── train/
    │   ├── AD/
    │   └── NC/
    │
    └── test/
        ├── AD/
        └── NC/

The supplied test directory is kept separate from model training. The train data is split and ensured that there is no overlap between train and validation datasets.

## Dependencies
Ensure appropriate libraries are installed. 
	torch 2.14.1+cu130
	torchvision 0.29.1+cu130
	numpy 2.5.4
	Pillow 12.3.0
	matplotlib 3.11.2
	scikit-learn 1.9.1

## Reproducibility
The patient-level training and validation split uses a fixed random seed of 42. This means that the same patient IDs are assigned to the same training and validation groups each time the dataset is loaded.

For both ConvNeXt and ResNet50:
- training, validation and test patient partitions
- image resolution
- batch size
- input representation
- evaluation metrics
- test dataset
are used.




## References

[1]
Z. Liu _et al._, ‘A ConvNet for the 2020S’, Mar. 2022. Accessed: Oct. 30, 2026. [Online]. Available: https://arxiv.org/pdf/2201.03545

[2]
M. Zhao, X. Xu, X. Bao, X. Chen, and H. Yang, ‘An Automated Instance Segmentation Method for Crack Detection Integrated With CrackMover Data Augmentation’, _Sensors_, vol. 24, no. 2, p. 446, Jan. 2024, doi: 10.3390/s24020446.