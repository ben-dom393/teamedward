from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import argparse

def parse_json_data(json_data_dir):
    """
    Converts the raw json dataset into a list of dictionaries
    """
    dictlist = []
    with open(json_data_dir) as json_file:
        for jsonobj in json_file:
            dictlist.append(json.loads(jsonobj))
    return dictlist

def summarise_json_data(dictlist):
    """
    Summarise the list of readings, obtaining the weighted mean (weighed by 
    dwell time), 25th, 50th and 75th percentile of the mean and sd

    Returns a pandas dataframe
    """
    df_list = []
    for i in tqdm(dictlist):
        transcript_id = list(i.keys())[0]
        transcript_position = list(i[transcript_id].keys())[0]
        nucleotide_seq = list(i[transcript_id][transcript_position].keys())[0]
        readings = i[transcript_id][transcript_position][nucleotide_seq]
        readings_df = pd.DataFrame(readings)
        readings_df.columns = ['dwell_time_-1','sd_-1','mean_-1','dwell_time_0','sd_0','mean_0','dwell_time_1','sd_1','mean_1']
        positions = ['-1','0','1']
        summarized_features = {}
        for pos in positions:
            #Get Weighted Features (Mean, SD of Readings)
            summarized_features[f'weighted_mean_{pos}'] = (readings_df[f'dwell_time_{pos}']*readings_df[f'mean_{pos}']).sum()/\
                readings_df[f'dwell_time_{pos}'].sum()
            summarized_features[f'weighted_sd_{pos}'] = np.sqrt((readings_df[f'sd_{pos}']*readings_df[f'sd_{pos}']*(readings_df[f'dwell_time_{pos}'])).sum()/\
                                                                readings_df[f'dwell_time_{pos}'].sum())
            #Get Features at different quantiles (Mean, SD of Readings)
            summarized_features[f'mean_25_{pos}'] = readings_df[f'mean_{pos}'].quantile(0.25)
            summarized_features[f'mean_50_{pos}'] = readings_df[f'mean_{pos}'].quantile(0.5)
            summarized_features[f'mean_75_{pos}'] = readings_df[f'mean_{pos}'].quantile(0.75)
            summarized_features[f'sd_25_{pos}'] = readings_df[f'sd_{pos}'].quantile(0.25)
            summarized_features[f'sd_50_{pos}'] = readings_df[f'sd_{pos}'].quantile(0.5)
            summarized_features[f'sd_75_{pos}'] = readings_df[f'sd_{pos}'].quantile(0.75)

        
        df_list.append([transcript_id,transcript_position,nucleotide_seq]+[summarized_features[i] for i in summarized_features])
    #Convert list of lists into pandas dataframe
    df = pd.DataFrame(df_list)
    df.columns = ['transcript_id','transcript_position','nucleotide_seq']+[i for i in summarized_features]
    return df

def encode_nucleotides(df):
    """
    Generate features from the 7-character nucleotide sequence
    """
    relevant_positions = [0,1,2,5,6]
    #Maintain the list of categorical columns for OneHotEncoding
    categorical_columns = []
    #Generates a column showing the nucleotide at each of the relevant position
    for pos in relevant_positions:
        col_name = f'nucleotide_{pos}'
        categorical_columns.append(col_name)
        df[col_name] = df['nucleotide_seq'].map(lambda x:x[pos])
    #Sliding window to get the 5-mer at position -1, 0 and 1
    for i in range(3):
        col_name = f'5-mer_window_{i-1}'
        categorical_columns.append(col_name)
        df[col_name] =  df['nucleotide_seq'].map(lambda x:x[i:i+5])
    #One Hot Encode these categorical columns
    encoder = OneHotEncoder()
    df_categorical = df[categorical_columns]
    
    ohe_columns = encoder.fit_transform(df_categorical)
    ohe_df = pd.DataFrame(ohe_columns.toarray(), columns=encoder.get_feature_names_out(input_features=categorical_columns))
    #Join these columns back to the original dataframe, removing the original columns
    df_encoded = pd.concat([df.drop(columns = categorical_columns),ohe_df], axis = 1)
    #Generates a column showing the frequency of each nucleotide in the sequence
    nucleotides = ['A','C','G','T']
    for nuc in nucleotides:
        df_encoded[f'{nuc}_freq'] = df_encoded['nucleotide_seq'].map(lambda x:x.count(nuc))

    return df_encoded

def main():
    parser = argparse.ArgumentParser(description = "Train a ML model to predict m6A modification")
    parser.add_argument("json_data_dir",help = "File path for RNA-seq data (.json)")
    args = parser.parse_args()
    print("Preprocessing JSON data")
    dictlist = parse_json_data(args.json_data_dir)
    df = summarise_json_data(dictlist)
    df_encoded = encode_nucleotides(df)
    df_encoded.to_csv('sample_processed.csv', index = False)
    print("Training Model")

if __name__ == "__main__":
    main()