# FDM 3D Printing Quality Monitoring and Control Research

This repository contains experimental code for image-based monitoring of Fused Deposition Modeling (FDM/FFF) 3D printing. The current primary training pipeline uses a residual-attention convolutional neural network to classify three visual quality attributes from images of the nozzle and extruded filament:

- Uniformity
- Continuity
- Adhesion

The repository also includes data preparation and capture utilities, camera streaming, inference scripts, and fuzzy-rule experiments for proposing process-parameter adjustments.

> **Important:** This repository contains code from multiple stages of experimentation. The current classifier predicts three binary quality labels. Some older scripts refer to a different four-output regression/classification interface and should not be assumed to work with the current model.

## Table of contents

- [Project overview](#project-overview)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Environment setup](#environment-setup)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Image capture and camera streaming](#image-capture-and-camera-streaming)
- [Fuzzy-rule experiments](#fuzzy-rule-experiments)
- [Limitations and reproducibility](#limitations-and-reproducibility)
- [Recommended workflow](#recommended-workflow)
- [License](#license)

## Project overview

The research investigates whether visual features around the nozzle and deposited filament can be used to classify print-quality states. The current classifier predicts three labels from an RGB image crop:

1. **Uniformity**
2. **Continuity**
3. **Adhesion**

These are image-based classification outputs. They are not direct measurements of flow rate, feed rate, Z offset, or hotend temperature, and a predicted quality label alone does not prove the physical cause of a defect.

The repository also contains experimental fuzzy logic code that maps quality-label combinations to proposed changes in process parameters. This should be treated as research/prototype logic, not validated autonomous printer-control software.

## Architecture

The current classifier pipeline is:

    Annotated image + CSV
             |
             v
    ParametersDataset
             |
             v
    Crop around nozzle tip
             |
             v
    Resize / tensor conversion / normalization
             |
             v
    ResidualAttentionModel_56 backbone
             |
       2048-D features
             |
      +------+------+ 
      |      |      |
      v      v      v
    Uniform Continuity Adhesion
      head     head    head
      |        |       |
      +--------+-------+
               |
               v
      Three binary predictions

### Model details

The main model is ParametersClassifier in model/network_module.py.

- **Backbone:** ResidualAttentionModel_56 from model/residual_attention_network.py.
- **Feature vector:** 2,048 features after adaptive average pooling.
- **Prediction heads:** Three independent fully connected heads for uniformity, continuity, and adhesion.
- **Output:** Three logits per image, ordered as uniformity, continuity, adhesion.
- **Loss:** Binary cross-entropy with logits. The adhesion loss is weighted by 2.5, then added to the uniformity and continuity losses.
- **Prediction rule:** Apply sigmoid to each logit; values greater than 0.5 are mapped to label 1, otherwise label 0.

In the current inference display, label 1 is presented as GOOD and label 0 as BAD. Confirm that the labels in your CSV follow this convention.

## Repository structure

The main files in the current source tree are:

| Path | Description |
|---|---|
| train.py | Main PyTorch Lightning training entry point |
| train_config.py | Dataset paths, normalization statistics, learning rate, batch size, epoch count, and preprocessing |
| data/dataset.py | Loads annotations and images, crops around nozzle coordinates, and returns the three quality labels |
| data/data_module.py | Creates datasets and train/validation/test DataLoaders |
| model/network_module.py | Classifier, loss functions, metrics, optimizer, and scheduler |
| model/residual_attention_network.py | Residual-attention feature extractor |
| model/attention_module.py | Attention modules used by the backbone |
| model/basic_layers.py | Basic residual layers |
| test.py | Test-set evaluation and prediction visualization |
| predict.py | Single-image inference for the current three-output classifier |
| make_data.py | Webcam capture, ROI preview, image saving, and CSV row generation |
| data_process.py | Experimental CSV filtering and balancing code |
| mapping.py | Generates rule_mapping_24.csv from predefined parameter/quality combinations |
| features_mapping.csv | Example mapping between discretized parameter classes and quality labels |
| fuzzy/ | Fuzzy rules, membership functions, parameter adjustment helpers, and experiments |
| camera_server.py | MJPEG camera stream and snapshot endpoints |
| simulation.py | Experimental webcam loop integrating classification and fuzzy rules |
| samples.py | Legacy sample inference script; see compatibility notes below |

The repository does not currently include a dependency lockfile or requirements.txt. The complete dataset and a portable, documented checkpoint download are not provided as part of the source tree.

## Dataset

The main dataset loader expects a CSV annotation file and image files on disk.

### Required CSV fields

The current ParametersDataset in data/dataset.py accesses these fields:

| Column | Description |
|---|---|
| img_path | Image path joined to the configured root directory |
| nozzle_tip_x | Horizontal coordinate used to center the crop |
| nozzle_tip_y | Vertical coordinate used to center the crop |
| uniformity | Binary target label |
| continuity | Binary target label |
| adhesion | Binary target label |

Illustrative CSV header and row:

    img_path,nozzle_tip_x,nozzle_tip_y,uniformity,continuity,adhesion
    caxton_dataset/images/example.jpg,320,240,1,1,0

Replace the example path and labels with valid annotations. The image path must resolve after the loader joins it with DATA_DIR.

### Image preprocessing

The main data module performs the following operations:

1. Loads the image as RGB.
2. Crops a region around the nozzle coordinates. The default crop dimension is 320 × 320 pixels.
3. For training, applies random rotation and random perspective before cropping.
4. Applies random resized crop, horizontal flip, and color jitter to the cropped image.
5. Converts the image to a tensor and normalizes it to 224 × 224 input size.

When the data module is created with transform=False, it uses deterministic resizing, tensor conversion, and normalization rather than the training augmentation pipeline.

The current channel-wise normalization constants in train_config.py are:

- Mean: [0.288, 0.266, 0.137]
- Standard deviation: [0.196, 0.194, 0.173]

These values should correspond to the dataset used for training. Recalculate or verify them when using a different dataset.

### Train/validation/test split

The data module uses a nominal 70% training, 20% validation, and 10% test split. However, in the current implementation, random_split is called separately when creating the training/validation split and the test split if saved splits are not loaded. These independent splits can overlap. Do not treat the resulting test score as a reliable held-out estimate until this split logic is corrected.

For reproducible evaluation, create one split using a fixed random seed, persist the sample indices or file lists, and reuse the same non-overlapping split for every experiment.

## Environment setup

The code imports PyTorch, TorchVision, PyTorch Lightning, pandas, NumPy, Pillow, OpenCV, and Matplotlib. Flask is used by the camera server.

Create and activate a virtual environment:

    git clone https://github.com/NguyenDevxD/RESEARCH.git
    cd RESEARCH
    python -m venv .venv

**Windows PowerShell**

    .venv\Scripts\Activate.ps1

**Linux/macOS**

    source .venv/bin/activate

Install PyTorch and TorchVision using the official selector for your operating system and CUDA configuration: https://pytorch.org/get-started/locally/

Install the other dependencies:

    python -m pip install --upgrade pip
    python -m pip install pytorch-lightning pandas numpy pillow opencv-python matplotlib flask

This is a starting environment, not a pinned or fully tested dependency specification. Record the versions of Python, PyTorch, TorchVision, PyTorch Lightning, CUDA, and the GPU used for each reproducible experiment.

## Configuration

### train_config.py

Review these settings before training:

| Setting | Current source value | Purpose |
|---|---|---|
| DATA_DIR | Absolute Windows path | Root directory used to resolve image paths |
| DATASET_NAME | dataset_full | Dataset/log naming identifier |
| DATA_CSV | DATA_DIR + caxton_dataset/full_data.csv | Annotation CSV |
| DATASET_MEAN | [0.288, 0.266, 0.137] | Image normalization mean |
| DATASET_STD | [0.196, 0.194, 0.173] | Image normalization standard deviation |
| INITIAL_LR | 1e-3 | Learning rate passed to the model |
| BATCH_SIZE | 32 | Batch size passed to the data module |
| MAX_EPOCHS | 40 | Default maximum epoch count |
| NUM_GPUS | 1 | GPU count requested when CUDA is available |

**Change DATA_DIR and verify DATA_CSV on your machine.** The committed path is machine-specific and will not work on a different system without editing.

### Batch size

BATCH_SIZE is defined in train_config.py and passed by train.py into ParametersDataModule. The data module uses this batch size for training, validation, and test DataLoaders.

Change BATCH_SIZE in train_config.py to adjust it. If GPU memory is insufficient, try a smaller batch size. The DataLoaders currently use num_workers=7 and pin_memory=True; reduce the worker count in data/data_module.py if multiprocessing causes problems on your platform.

### Learning rate, optimizer, and scheduler

The current implementation uses **one shared learning rate**. The model's configure_optimizers method creates a single AdamW optimizer over the model parameters with:

- Learning rate from INITIAL_LR (currently 1e-3 in train_config.py).
- Weight decay of 1e-4.
- OneCycleLR scheduling, with max_lr equal to the configured learning rate.
- pct_start=0.1, cosine annealing, div_factor=25, and final_div_factor=100.
- Scheduler updates at every training step.

The current source does **not** implement separate learning rates for the CNN backbone and the classification heads, despite this being a possible experiment. To test separate rates, modify configure_optimizers in model/network_module.py to create separate optimizer parameter groups for attention_model and the three fc_* heads, then expose the corresponding values in the configuration. Ensure each trainable parameter appears in exactly one group. Compare the shared-rate and separate-rate settings using the same fixed validation split.

### Checkpoint initialization

train.py contains a hard-coded Windows CHECKPOINT_PATH. If the path exists, the script attempts to load backbone weights from that checkpoint; otherwise it trains from scratch. Edit or remove this path before running on another machine. Only use a checkpoint compatible with the ResidualAttentionModel_56 backbone.

## Training

After updating the data paths and checking that the CSV and images are valid, run:

    python train.py --seed 1234 --epochs 40

Arguments:

- --seed: random seed (default: 1234).
- --epochs: maximum number of epochs (defaults to MAX_EPOCHS in train_config.py).

The script creates a PyTorch Lightning Trainer with mixed precision (16-mixed), gradient clipping at 1.0, TensorBoard logging, and a rich progress bar. It uses a GPU if CUDA is available and otherwise falls back to CPU.

Outputs are written to directories based on the dataset name and seed:

- Logs: logs/<dataset_name>_classifier_<seed>/
- Checkpoints: checkpoints/<dataset_name>_classifier_<seed>/

The checkpoint callback monitors val_acc_adhesion and saves the best checkpoint according to that metric. Therefore, checkpoint selection prioritizes adhesion validation accuracy rather than an aggregate score across all three outputs.

### TensorBoard

Install TensorBoard if needed and start it with:

    python -m pip install tensorboard
    tensorboard --logdir logs

Open the local URL shown in the terminal to inspect logged metrics.

## Evaluation

test.py evaluates the classifier and visualizes sample predictions. Before running it, update the hard-coded checkpoint path in test.py and confirm that train_config.py points to the intended dataset.

Run:

    python test.py --seed 1234

The script reports accuracy for uniformity, continuity, and adhesion, calculates mean accuracy, and displays example test images with predictions and labels.

**Evaluation warning:** Fix the split overlap described above before using these metrics as a scientific test result. A valid evaluation requires a fixed, non-overlapping test set that was not used for training or hyperparameter selection.

## Inference

predict.py is the example inference script for the current three-output classifier. Before running it, configure:

- CKPT_PATH: a checkpoint created for ParametersClassifier.
- IMG_PATH: the image to analyze.
- NOZZLE_TIP: the nozzle-tip coordinate in the full image.
- IMAGE_DIM: crop size around the nozzle tip.

Then run:

    python predict.py

The script chooses CUDA when available, loads the checkpoint, crops around the configured nozzle coordinates, converts the crop to RGB, resizes and normalizes it, and prints a GOOD/BAD result and sigmoid probability for each output. It also displays the original image with the crop region and the crop itself.

The crop position and preprocessing must be consistent with training. Incorrect nozzle coordinates or a crop that differs substantially from training images can degrade predictions.

### Legacy samples.py compatibility

The current samples.py script expects four outputs and uses a legacy checkpoint-loading interface. The current ParametersClassifier returns one tensor containing three logits. Therefore, samples.py should not be assumed to run with the current classifier without modification. Use predict.py for the current model or update samples.py to match the current model interface.

## Image capture and camera streaming

### Capture images with make_data.py

make_data.py opens a camera, displays the full frame and the region of interest, saves cropped images, and appends annotations to dataset/data.csv. Its configurable fields include camera index, ROI position, crop dimensions, process-parameter metadata, quality labels, and capture interval.

The values in the script are examples. Change them to match the actual print session and annotation procedure. The capture script writes quality labels as strings such as good and bad, while the classifier dataset expects numeric binary labels. Convert and validate these labels before using captured data for training. Also verify that each saved image path resolves from DATA_DIR.

Run:

    python make_data.py

Use W/A/S/D to move the ROI and Q to exit.

### Camera server

camera_server.py provides two endpoints:

- /video.mjpg — MJPEG camera stream.
- /snapshot.jpg — latest frame as a JPEG image.

Run it with:

    python camera_server.py

The script uses camera index 1 and starts Flask on port 8081, bound to 0.0.0.0. Change the camera index as needed. Binding to all network interfaces may expose the camera feed to other devices on the network; use only in a trusted environment unless access controls are added.

## Fuzzy-rule experiments

The fuzzy directory contains rule-based experiments that map combinations of Good/Bad uniformity, continuity, and adhesion labels to qualitative actions such as Increase, Decrease, and Remain. The controller applies fixed increments to flow rate, feed rate, Z offset, and hotend temperature, while config.py clamps values to configured bounds.

simulation.py demonstrates an experimental webcam loop that classifies the three quality attributes, buffers predictions for majority voting, and passes the resulting labels to the fuzzy controller.

These components are prototypes, not validated printer-control software. Heuristic rules do not guarantee that a proposed adjustment is physically correct or safe for a specific printer, material, or print condition. Some fuzzy scripts reference older regression model modules or may need import-path adjustments.

Do not connect this experimental logic to a live printer without independent validation, conservative safety limits, a controlled test procedure, and a reliable mechanism to prevent unintended motion or temperature changes.

## Limitations and reproducibility

Before treating this repository as a reproducible research release, address the following:

1. **Machine-specific paths:** several scripts contain absolute Windows paths. Replace them with portable configuration or command-line arguments.
2. **Potential data leakage:** train/validation and test subsets may overlap because the current split code generates them in separate random_split calls.
3. **Unpinned dependencies:** no requirements.txt or environment lockfile is currently present.
4. **Legacy model interfaces:** samples.py, full_model.py, and fuzzy/command.py refer to interfaces or modules that do not align with the current three-output classifier.
5. **Hard-coded checkpoints:** training, evaluation, and inference scripts should accept checkpoint paths through configuration or CLI arguments.
6. **Capture/training schema mismatch:** make_data.py writes string labels and a capture-oriented CSV schema, while the classifier loader expects numeric binary labels and nozzle coordinates.
7. **Metric coverage:** accuracy alone may hide class imbalance. Consider per-class counts, precision, recall, F1-score, confusion matrices, and calibration where appropriate.
8. **No verified benchmark table:** publish dataset size, split protocol, hardware, runtime methodology, checkpoint, and metrics only after reproducing the measurements.
9. **Deployment scope:** camera capture and streaming utilities exist, but the repository does not provide a unified, packaged deployment service or a verified interface for sending commands to a printer.

## Recommended workflow

1. Validate the annotation CSV, binary labels, image paths, and nozzle coordinates.
2. Replace machine-specific paths and record the software/hardware environment.
3. Create and persist a deterministic, non-overlapping train/validation/test split.
4. Train a baseline using train.py and retain the configuration, logs, and selected checkpoint.
5. Evaluate on the untouched test split after correcting the split logic.
6. Inspect representative predictions using predict.py and verify the crop matches training preprocessing.
7. Experiment with batch size and learning rate in controlled runs. Treat separate backbone/head learning rates as a proposed experiment until implemented and measured.
8. Evaluate fuzzy parameter adjustments separately from image-classification performance before considering any physical printer integration.

## License

No license file is present in the current repository tree. Unless a license is added by the project owner, do not assume the code is released under a specific open-source license.

If this repository supports a publication or incorporates third-party research code, add verified citations, attribution, and applicable license notices here.
