from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import argparse
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle

def parse_json_data(json_data_dir):
    """
    Converts the raw json dataset into a list of dictionaries
    """
    dictlist = []
    with open(json_data_dir) as json_file:
        for jsonobj in json_file:
            dictlist.append(json.loads(jsonobj))
    return dictlist

def get_weighted_readings(summarized_features, readings_array):
    """
    Get Weighted Mean and SD
    """
    positions = ['-1', '0', '1']
    weighted_means = np.zeros(len(positions))
    weighted_sds = np.zeros(len(positions))
    NUM_FEATURES = 3 #Dwell time, SD, Mean

    for pos_idx, pos in enumerate(positions):
        dwell_time_col = readings_array[:, pos_idx * NUM_FEATURES]
        sd_col = readings_array[:, pos_idx * NUM_FEATURES + 1]
        mean_col = readings_array[:, pos_idx * NUM_FEATURES + 2]
        
        total_dwell_time = dwell_time_col.sum()
        
        weighted_means[pos_idx] = (dwell_time_col * mean_col).sum() / total_dwell_time
        weighted_sds[pos_idx] = np.sqrt((sd_col * sd_col * dwell_time_col).sum() / total_dwell_time)
    
    for pos_idx, pos in enumerate(positions):
        summarized_features[f'weighted_mean_{pos}'] = weighted_means[pos_idx]
        summarized_features[f'weighted_sd_{pos}'] = weighted_sds[pos_idx]
    
    return summarized_features

def get_readings_quantiles(summarized_features, readings_array):
    """
    Get Mean and SD at 25th, 50th, and 75th percentile
    """
    positions = ['-1', '0', '1']
    quantiles = [25, 50, 75]
    NUM_FEATURES = 3 #Dwell time, SD, Mean
    for pos_idx, pos in enumerate(positions):
        sd_col = readings_array[:, pos_idx * NUM_FEATURES + 1]
        mean_col = readings_array[:, pos_idx * NUM_FEATURES + 2]
        
        for quant in quantiles:
            summarized_features[f'mean_{quant}_{pos}'] = np.percentile(mean_col, quant)
            summarized_features[f'sd_{quant}_{pos}'] = np.percentile(sd_col, quant)
    
    return summarized_features

def summarise_json_data(dictlist):
    """
    Summarise the list of readings, obtaining the weighted mean (weighed by 
    dwell time), 25th, 50th, and 75th percentile of the mean and sd

    Returns a pandas dataframe
    """
    df_list = []
    
    for i in tqdm(dictlist):
        transcript_id = list(i.keys())[0]
        transcript_position = list(i[transcript_id].keys())[0]
        nucleotide_seq = list(i[transcript_id][transcript_position].keys())[0]
        readings = i[transcript_id][transcript_position][nucleotide_seq]
        
        # Create a NumPy array from readings
        readings_array = np.array(readings)
        
        summarized_features = {}
        summarized_features = get_weighted_readings(summarized_features, readings_array)
        summarized_features = get_readings_quantiles(summarized_features, readings_array)
        
        # Convert the list to include transcript information and features
        row = [transcript_id, transcript_position, nucleotide_seq]
        for feature in summarized_features.values():
            row.append(feature)
        
        df_list.append(row)
    
    # Convert list of lists into pandas dataframe
    columns = ['transcript_id', 'transcript_position', 'nucleotide_seq']
    columns += [f for f in summarized_features]
    df = pd.DataFrame(df_list, columns=columns)
    return df

def encode_nucleotides(df):
    """
    Generate features from the 7-character nucleotide sequence
    """
    WINDOW_SIZE = 5
    relevant_positions = [0,1,2,5,6] # Positions 3 and 4 are excluded as all of them have the same nucleotide
    # Maintain the list of categorical columns for OneHotEncoding
    categorical_columns = []
    # Generates a column showing the nucleotide at each of the relevant position
    for pos in relevant_positions:
        col_name = f'nucleotide_{pos}'
        categorical_columns.append(col_name)
        df[col_name] = df['nucleotide_seq'].map(lambda x:x[pos])
    # Sliding window to get the 5-mer at position -1, 0 and 1
    for i in range(7-WINDOW_SIZE+1):
        col_name = f'5-mer_window_{i-1}'
        categorical_columns.append(col_name)
        df[col_name] =  df['nucleotide_seq'].map(lambda x:x[i:i+WINDOW_SIZE])

    # Generates a column showing the frequency of each nucleotide in the sequence
    nucleotides = ['A','C','G','T']
    for nuc in nucleotides:
        df[f'{nuc}_freq'] = df['nucleotide_seq'].map(lambda x:x.count(nuc))

    return df, categorical_columns


def prepare_dataset_for_prediction(df_test, scaler, encoder, categorical_columns):
    """
    Drop irrelevant columns and scale input features
    """
    
    # Extract input features
    X_test = df_test.drop(columns = ['transcript_id','transcript_position','nucleotide_seq'])
    #One hot Encode
    X_test_categorical = X_test[categorical_columns]
    test_ohe_columns = encoder.transform(X_test_categorical)
    test_ohe_df = pd.DataFrame(test_ohe_columns.toarray(), columns=encoder.get_feature_names_out(input_features=categorical_columns))
    X_test_encoded = pd.concat([X_test.drop(columns = categorical_columns),test_ohe_df], axis = 1)
    # Scale input features
    X_test_scaled = scaler.transform(X_test_encoded)

    return X_test_scaled


def main():
    parser = argparse.ArgumentParser(description = "Generate predictions for RNA-seq data")
    parser.add_argument("json_data_dir",help = "File path for RNA-seq data (.json)")
    parser.add_argument("output_dir", help = "File path for predictions output (.csv)")
    parser.add_argument("-m","--model", help = "File path for fitted model object (.h5). Default: models/fitted_model.h5")
    parser.add_argument("-s","--scaler", help = "File path for fitted scaler object (.pkl). Default: models/fitted_scaler.pkl")
    parser.add_argument("-e","--encoder",help = "File path for fitted one-hot encoder object (.pkl). Default: models/fitted_encoder.pkl")
    args = parser.parse_args()

    if not args.model:
        model = keras.models.load_model('models/fitted_model.h5')
    else:
        model = keras.models.load_model(args.model)

    if not args.scaler:
        scaler_file = open('models/fitted_scaler.pkl','rb')
    else:
        scaler_file = open(args.scaler, 'rb')
    scaler = pickle.load(scaler_file)

    if not args.encoder:
        encoder_file = open('models/fitted_encoder.pkl','rb')
    else:
        encoder_file = open(args.encoder,'rb')
    encoder = pickle.load(encoder_file)
    

    print("=====Preprocessing JSON data=====")
    dictlist = parse_json_data(args.json_data_dir)
    df = summarise_json_data(dictlist)
    df, categorical_columns = encode_nucleotides(df)
    X_test_scaled = prepare_dataset_for_prediction(df, scaler, encoder, categorical_columns)
    print("=====Generating Predictions=====")
    df['score'] = model.predict(X_test_scaled)
    prediction_df = df[['transcript_id','transcript_position','score']]
    prediction_df.to_csv(args.output_dir, index = False)


if __name__ == "__main__":
    main()