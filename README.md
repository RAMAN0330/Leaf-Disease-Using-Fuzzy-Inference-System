# Leaf Disease Classification using Fuzzy Logic

This project demonstrates the classification of leaf diseases using fuzzy logic. Fuzzy logic allows for imprecise reasoning and decision-making, making it suitable for scenarios where exact mathematical models are not available.

## Overview

In this project, we implement a fuzzy logic-based classifier to classify leaf diseases based on extracted features such as lesion size and density. The classifier uses linguistic variables, membership functions, and fuzzy rules to make classification decisions.

## Features

- Preprocessing: Resize input images to a fixed size for uniformity.
- Feature Extraction: Extract features such as mean and standard deviation of pixel intensities from preprocessed images.
- Fuzzy Logic Classification: Classify leaf diseases using fuzzy logic with linguistic variables, membership functions, and fuzzy rules.
- Interpretability: Obtain interpretable classification results based on the degree of membership.

## Usage

1. **Preprocessing**: Use the `Preprocessor` class to resize input images.
2. **Feature Extraction**: Extract features using the `FeatureExtractor` class.
3. **Fuzzy Logic Classification**: Classify leaf diseases using the `FuzzyLeafDiseaseClassifier` class.
4. **Example Usage**: See the provided example script (`Flower_Disease.py`) for usage demonstration.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/RAMAN0330/leaf-disease-fuzzy-classification.git
