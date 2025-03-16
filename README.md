# AutoGISAXS

## Deep-learning assisted analysis of 2D grazing incidence small-angle X-Ray scattering (GISAXS) data

This repository contains the implementation of a deep learning-based approach to analyzing GISAXS data. The project uses artificial neural networks (ANNs) trained on simulated scattering images to extract critical morphological parameters, such as cluster radius and inter-cluster distance, from GISAXS patterns. The methods presented here aim to enhance the efficiency and accuracy of GISAXS data interpretation, reducing reliance on time-consuming manual fitting procedures.

## Features
- Uses deep feedforward neural networks for parameter extraction
- Incorporates prior physical knowledge to improve training effectiveness
- Preprocessing pipeline to align experimental and simulated GISAXS images
- Regularization techniques to improve model generalization

## Repository Structure
<!---
```
AutoGISAXS/
│-- data/                 # contains the raw data: experimental and                         simulated GISAXS images
│-- models/               # trained deep learning models
│-- notebooks/            # Jupyter notebooks for data exploration and model evaluation
│-- src/                  # Source code for training and inference
│   │-- preprocessing.py  # Image preprocessing steps
│   │-- train.py          # Training script for neural networks
│   │-- inference.py      # Inference script for extracting parameters from GISAXS images
│-- results/              # Outputs, evaluation results, and visualizations
│-- requirements.txt      # Python dependencies
│-- README.md             # Project documentation
```
-->
## Installation
To use this project, ensure you have Python 3.8+ installed. Clone the repository and install dependencies:
```sh
git clone https://github.com/sdan141/AutoGISAXS.git
cd AutoGISAXS
pip install -r requirements.txt
```

## Usage
### Preprocess Data
Run the preprocessing script to align and normalize experimental and simulated GISAXS images
<!---
```sh
python src/preprocessing.py --input_dir data/raw --output_dir data/processed
```
-->


### Train Model
Train the ANN on the simulated GISAXS dataset
<!---
```sh
python src/train.py --data_dir data/processed --epochs 50 --batch_size 32
```
-->

### Run Inference
Use the trained model to predict morphological parameters from 
GISAXS images
<!---
:
```sh
python src/inference.py --model_path models/best_model.pth --input_image data/test_image.png
```
-->
## Data
- **Simulated GISAXS images**: Generated using IsGISAXS software/ other with an HDF5 style database
- **Experimental GISAXS images**: CBF files

## Model
The deep learning model is a feedforward neural network designed for probabilistic parameter estimation. <!---The architecture consists of multiple fully connected layers with Leaky ReLU activations and dropout regularization.-->

## References
- TUHH Bachelor's Thesis: _Deep-Learning-Assisted Analysis of GISAXS Images_
- [IsGISAXS Software](https://w3.insp.upmc.fr/oxydes//IsGISAXS/isgisaxs.htm)
<!---- [Deep Learning for Scattering Data Analysis](https://arxiv.org/abs/2201.XXXX)-->

<!---- ## License
This project is licensed under the MIT License.-->

<!----## Contact
For questions, contact **Shachar Dan** at (mailto:shachardan94@gmail.com).-->
