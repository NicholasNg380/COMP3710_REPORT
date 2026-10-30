# Alzheimer’s Disease Classification Between ConvNeXt And ResNet50

This report investigates cnn architectures for binary classification of Alzheimer’s Disease using the ADNI image dataset.

The primary model is a custom implementation of ConvNeXt, with ResNet50 implemented as a baseline for comparison.

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

A title, a description of the algorithm and the problem that it solves (approximately a paragraph),
how it works in a paragraph, and a figure/visualisation.

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


• Any dependencies required, including exact versions, and address reproducibility of results.
• Example inputs, outputs, and visual plots of your algorithm.
• Description of any specific pre-processing used, with references if any. Justify your training,
validation, and testing splits of the data.
• Investigation of the Open Research Dilemma:
– Quantitative benchmarking of your chosen model against your implemented baseline model under
identical evaluation splits.
– Resource profiling table documenting computational efficiency (peak GPU VRAM, parameter
count, and inference runtime/latency).
– In-depth qualitative error autopsies analyzing 3–5 representative failure cases from the dataset
(diagnosing root triggers and failure modes).
– Actionable engineering recommendation and trade-off synthesis for your project manager.
• The mandatory Artificial Intelligence Usage Disclosure subsection (see Section 6).