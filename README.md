# Landmark Recognition using Deep Learning

## Project Overview

This project performs **landmark recognition using deep learning**. The notebook loads landmark image data, performs basic data analysis and visualization, prepares the image data, and builds a **VGG19-based neural network** for landmark classification.

## Features

- Loads and explores landmark training data
- Analyzes the distribution of landmark classes
- Displays sample landmark images
- Encodes landmark labels using `LabelEncoder`
- Resizes images to `224 × 224`
- Normalizes image pixel values
- Uses a VGG19-based architecture for classification
- Trains the model using RMSprop
- Saves the trained model

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Pillow
- Scikit-learn
- TensorFlow
- Keras
- Jupyter Notebook

## Project Structure

```text
Landmark/
│
├── landmarks_recognition.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── train.csv
└── images/
    └── ...
```

> `train.csv` and the `images/` directory are required by the notebook. If they are large, they should not be uploaded directly to GitHub; instead, provide the dataset source/download instructions.

## Dataset

The dataset used in this project contains landmark images and
their corresponding training labels.

Due to the large size of the image dataset, the dataset is
hosted separately on Kaggle.

**Dataset:** [Landmark Recognition Dataset](https://www.kaggle.com/datasets/sansy060604/landmark-recognition-dataset)

The dataset contains:

- `train.csv` — training labels and image information
- `images/` — landmark images organized according to the image IDs

## Model

The project uses **VGG19** as the base architecture and adds a final dense classification layer.

Input image size:

```text
224 × 224
```

Optimizer:

```text
RMSprop
```

Learning rate:

```text
0.0001
```

Loss function:

```text
sparse_categorical_crossentropy
```

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Landmark
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
landmarks_recognition.ipynb
```

Make sure `train.csv` and the `images/` directory are available in the expected project location before running the notebook.

## Training

The current notebook uses:

- Batch size: `16`
- Epochs: `1`
- Image size: `224 × 224`
- Train/validation split: `80/20`

These parameters can be changed in the notebook according to the available computational resources.

## Output

The trained model is saved by the notebook using:

```python
model.save("Model")
```

## Notes

The uploaded notebook is a Jupyter checkpoint-derived notebook. It has been copied to the cleaner project filename `landmarks_recognition.ipynb` for use in the GitHub repository.

Before presenting the repository as a fully reproducible project, verify the dataset files and run the notebook from start to finish. In particular, the notebook currently contains a function naming mismatch between `encode_laabel` and `encode_label`, which should be corrected before a fresh training run.

## Author

**Sanskriti Kumari**

