# Bird Species Classification with Convolution Neural Networks

## Overview

This assignment involves building and training a neural network to classify images of birds into one of 10 species using PyTorch. You will learn how to design, train, and evaluate a deep learning model for image classification.

**Accuracy on the trained dataset**:  94.68%
## Directory Structure

```
bird.py
environment.yml
install.sh
run.ipynb
model_path.txt
Analysis_report.pdf
```

- **bird.py**: Main Python script for training and testing the neural network.
- **environment.yml**: Conda environment file listing all dependencies.
- **install.sh**: Shell script to install dependencies (for Kaggle, uses pip).
- **run.ipynb**: Jupyter notebook with instructions and example commands.
- **model_path.txt**: Link to best model trained. 
- **Analysis_report.pdf**: Explains the model architecture and all the evalution results on the final model trained. Please refer to this for proper understanding.

## Dataset

Dataset from Kaggle: [Identify-the-Birds](https://www.kaggle.com/datasets/aayushkt/identify-the-birds)

## Setup Instructions

### On Kaggle

1. **Copy environment files to working directory:**
    ```python
    !cp /kaggle/input/setupfiles/environment.yml /kaggle/working/
    !cp /kaggle/input/setupfiles/install.sh /kaggle/working/
    !bash install.sh
    ```

2. **Run the training script:**
    ```python
    !python bird.py path_to_dataset train bird.pth
    ```

3. **Run the testing script:**
    ```python
    !python bird.py path_to_dataset test bird.pth
    ```

### Locally

1. Create the environment:
    ```sh
    conda env create -f environment.yml
    conda activate birds-env
    ```

2. Install any missing pip dependencies:
    ```sh
    pip install kaggle
    ```

3. Run the scripts as shown above.

## Usage

- **Training:**  
  `python bird.py <path_to_dataset> train <model_save_path>`
- **Testing:**  
  `python bird.py <path_to_dataset> test <model_load_path>`

The script will output predictions to `bird.csv` during testing.

## Model

The model is defined in [`bird.py`](bird.py) as a convolutional neural network (`SimpleCNN`) using PyTorch.

## Data Preparation

- The dataset should be organized in folders by class, compatible with `torchvision.datasets.ImageFolder`.
- The script automatically applies data augmentation and normalization.

## Output

- After training, the model is saved to the specified path.
- After testing, predictions are saved in `bird.csv` with a column `Predicted_Label`.

## Notes

- Make sure your dataset path is correct and accessible.
- The environment setup is designed for Kaggle but can be adapted for local use.
- For any issues, check the dependencies in [`environment.yml`](environment.yml) and the installation script [`install.sh`](install.sh).
- Feel free to go to the official problem statement for this project in [Problem Statement](https://lily-molybdenum-65d.notion.site/Assignment-3-Part-I-Learning-with-Neural-Networks-7388ff163f7b403482e2cc4329f03003)
---