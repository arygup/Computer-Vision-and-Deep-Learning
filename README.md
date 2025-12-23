# Computer Vision and Deep Learning

This repository contains a comprehensive collection of all tasks completed related to Computer Vision. The work spans the entire computer vision pipeline, progressing from fundamental digital image processing and manual feature extraction to advanced deep learning architectures, including Convolutional Neural Networks (CNNs), Object Detection systems, Semantic Segmentation models, and Vision Transformers.

## Repository Structure

The projects are organized by assignment number, moving from low level pixel manipulation to high level semantic understanding.

### 1: Image Processing and Fundamental Classification
This module focuses on foundational techniques in digital image processing and introductory machine learning using Multi-Layer Perceptrons (MLP).

* **Cloverleaf Bridge**
    * **Objective:** Analysis of aerial highway imagery to detect circular "cloverleaf" structures.
    * **Implementation:** Includes a custom histogram function implemented from scratch (benchmarked against OpenCV).
    * **Techniques:** Image preprocessing (noise reduction, edge detection), circular structure detection for distorted shapes, and radius/area estimation using both mathematical formulas and pixel-counting methods.

* **Line Segmentation in Historical Document**
    * **Objective:** Preprocessing pipeline for Optical Character Recognition (OCR) on ancient documents.
    * **Implementation:** Segmentation of text lines into individual image files and specialized extraction of text within circular seals.
    * **Techniques:** Grayscale conversion, binarization, morphological operations, and a comparative analysis between rectangular bounding boxes and polygonal boundaries for tight text fitting.

* **MLP Classification on Image Features**
    * **Objective:** Clothing item classification using PyTorch-based Multi-Layer Perceptrons.
    * **Implementation:** Training and hyperparameter tuning of three distinct models based on different input features.
    * **Techniques:** Feature extraction comparisons: Raw flattened pixels vs. Edge detection (Canny) vs. Histogram of Oriented Gradients (HOG). Metric logging via Weights & Biases (WandB).

### 2: CNN Architectures, Visualization, and Style Transfer
This module explores ResNet architectures, interpretability methods, and generative deep learning.

* **Convolutional Blocks of ResNet18**
    * **Objective:** Adaptation of the ResNet-18 architecture for custom small-scale datasets ($36 \times 36$).
    * **Implementation:** Training from scratch versus fine-tuning ImageNet-pretrained weights.
    * **Techniques:** Analysis of information loss in downsampling layers and implementation of custom architectural modifications (kernel adjustments in initial convolutional and pooling layers) to retain spatial resolution.

* **Network Visualisation**
    * **Objective:** Interpretability analysis to understand how CNNs process visual data.
    * **Implementation:** Generation of Saliency Maps to highlight pixel importance using gradient computations.
    * **Techniques:** Adversarial attacks using noise injection and gradient optimization to force misclassification (fooling the network) while maintaining visual similarity.

* **Style Transfer**
    * **Objective:** Implementation of Neural Style Transfer based on the Gatys et al. paper using VGG-19.
    * **Implementation:** Optimization of a target image to match the content of a photograph and the artistic style of a reference image.
    * **Techniques:** Custom loss functions (Content Loss vs. Style Loss via Gram Matrices) and optimizer comparison (L-BFGS vs. Adam).

### 3: Advanced Object Detection
This module focuses on extending the Faster R-CNN architecture for specialized detection tasks including oriented bounding boxes.

* **Oriented Bounding Boxes through Faster RCNN**
    * **Objective:** Extension of a standard Faster R-CNN to predict Oriented Bounding Boxes (OBB) for rotated objects.
    * **Implementation:** Modification of Region of Interest (ROI) heads to predict angle parameters.
    * **Techniques:** Visualization of Region Proposal Network (RPN) evolution, anchor analysis, and implementation of angle regression versus multi-bin classification. Evaluation using modified Mean Average Precision (mAP) for oriented boxes.

* **Fruit Detection and Counting**
    * **Objective:** Detection and counting of fruit instances in dense environments.
    * **Implementation:** Conversion of segmentation masks to bounding boxes and training a Faster R-CNN with a ResNet-34 backbone.
    * **Techniques:** Handling of overlapping instances, Non-Maximum Suppression (NMS) analysis, and evaluation of performance under occlusion and varying lighting conditions.

* **Human Parts Detection**
    * **Objective:** Detection of specific human body parts using deep learning detectors.
    * **Implementation:** Exploratory Data Analysis (EDA) on part scales and aspect ratios to inform model selection.
    * **Techniques:** Implementation of an object detection pipeline (e.g., DETR, YOLO, or RetinaNet) with custom anchor configurations and data augmentation strategies.

### 4: Semantic Segmentation
This module covers pixel-level classification using Fully Convolutional Networks and U-Net.

* **Fully Convolutional Networks for Semantic Segmentation**
    * **Objective:** Implementation of FCN variants for multi-class segmentation.
    * **Implementation:** Development of FCN-32s, FCN-16s, and FCN-8s architectures.
    * **Techniques:** Comparison of frozen versus fine-tuned VGG backbones and analysis of segmentation resolution across different upsampling strides.

* **Semantic Segmentation using U-Net**
    * **Objective:** Biomedical-style image segmentation using the U-Net architecture.
    * **Implementation:** Construction of a full Encoder-Decoder network with skip connections.
    * **Techniques:** Extensive ablation studies including U-Net without skip connections, Residual U-Net (ResBlocks), and Attention U-Net (Additive Attention Gates).

### 5: Vision Transformers and Multimodal Models
The final module implements state-of-the-art Transformer architectures and contrastive learning models.

* **Vision Transformer**
    * **Objective:** Implementation of a Vision Transformer (ViT) from scratch for image classification on CIFAR-10.
    * **Implementation:** Manual construction of Multi-Head Self-Attention (MSA), patch embeddings, and learnable positional encodings.
    * **Techniques:** Attention rollout visualization, analysis of [CLS] token attention, and hyperparameter tuning for patch sizes and embedding dimensions.

* **Differential Vision Tranformer**
    * **Objective:** Enhancement of the standard ViT using Differential Attention Mechanisms.
    * **Implementation:** Replacement of standard MSA with Differential Attention to reduce noise and focus on critical features.
    * **Techniques:** Comparative analysis of attention maps and convergence speed between Vanilla ViT and Diff-ViT.

* **CLIP**
    * **Objective:** Analysis of OpenAI's Contrastive Language-Image Pre-training (CLIP) model.
    * **Implementation:** Zero-shot learning evaluation on ImageNet categories.
    * **Techniques:** Comparison of CLIP ResNet-50 visual encoders against standard ImageNet-pretrained ResNet-50. Optimization analysis comparing FP32 versus FP16 inference for memory usage and speed.

## Technology Stack

* **Languages:** Python
* **Deep Learning Frameworks:** PyTorch, Torchvision
* **Computer Vision Libraries:** OpenCV, PIL
* **Data Analysis & Visualization:** NumPy, Matplotlib, Weights & Biases (WandB)
* **Architectures:** ResNet, VGG-19, Faster R-CNN, FCN, U-Net, ViT, CLIP
