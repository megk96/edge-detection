# Coursework 2

Coursework 2 contains two parts: Text Mining and Image Processing.

The README.txt contains details of the source code. This file is purely for set up and installation and running the code. 

## Installation

A virtualenv with Python3 is used for running all the sourcecode. 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

The following packages can be seen on the requirements.txt file:
```bash
matplotlib==3.3.4
numpy==1.19.5
scikit-image==0.18.1
pandas==1.2.2
scikit-learn==0.24.1
sklearn==0.0
scipy==1.6.0
```

## Directory Structure

```
Coursework
├── code
│    └── text-mining.py
│    └── image-processing.py
├── data
│    └── image_data
│    └── text_data
│── outputs 
│── README.md
│── README.txt
│── Report.pdf
│__ requirements.txt 
```

The .csv files need to be put in the data folder. 
## Running text-mining.py

Set the following constants, and vary them if you wish to experiment. 

```python
DATA_FOLDER = "../data/text_data"
FILE_NAME = "Corona_NLP_train.csv"
OUTPUT_FOLDER = "../outputs"
```
Run the command line argument:

```
python text-mining.py
```

## Running image-processing.py

Set the following constants, and vary them if you wish to experiment. 

```python
DATA_FOLDER = "../data/image_data"
OUTPUT_FOLDER = "../outputs"
Q1_FILE_NAME = "avengers_imdb.jpg"
Q2_FILE_NAME = "bush_house_wikipedia.jpg"
Q3_FILE_NAME = "forestry_commission_gov_uk.jpg"
Q4_FILE_NAME = "rolland_garros_tv5monde.jpg"
```
Run the command line argument:

```
python image-processing.py
```

The images can be found in the outputs folder. 
