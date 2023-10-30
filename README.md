# teamedward

## Installation Instructions
1. Start a Ubuntu 20.04 Large Instance of type t3.large or bigger.
2. SSH into the instance.
3. Run the following commands
```sh
sudo apt update
sudo apt -y install python3-pip
git clone https://github.com/ben-dom393/teamedward.git
cd teamedward
pip install -r requirements.txt
```

## Running Training Script on Sample Dataset (~30 seconds)
```sh
python3 train_script.py sample_dataset.json data.info
```
Output: Fitted Keras model `fitted_model.h5`, scaler `fitted_scaler.pkl` and one-hot encoder `fitted_encoder.pkl`. Stored in current directory

## Training Script Manual
```sh
Train a ML model to predict m6A modification

positional arguments:
  json_data_dir  File path for RNA-seq data (.json)
  data_info_dir  File path for m6A labels (.info)

optional arguments:
  -h, --help     show this help message and exit
```
## Running Prediction Script on Sample Dataset (~15 seconds)
```sh
python3 predict_script.py sample_dataset.json predictions.csv
```
Output: `predictions.csv` with the columns transcript_id, transcript_position and score (i.e. probability of m6A modification). Stored in current directory.

## Prediction Script Manual
```sh
Generate predictions for RNA-seq data

positional arguments:
  json_data_dir         File path for RNA-seq data (.json)
  output_dir            File path for predictions output (.csv)

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        File path for fitted model object (.h5). Default: models/fitted_model.h5
  -s SCALER, --scaler SCALER
                        File path for fitted scaler object (.pkl). Default: models/fitted_scaler.pkl
  -e ENCODER, --encoder ENCODER
                        File path for fitted one-hot encoder object (.pkl). Default: models/fitted_encoder.pkl
```
