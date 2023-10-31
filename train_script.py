from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import argparse
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import average_precision_score, roc_auc_score
from pickle import dump

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

def train_test_split(df, data_info, ratio = 0.1):
    """
    train_test_split by gene_id, default ratio=0.1 (10% of the dataset used as test)
    """
    # Join with df with info
    df['transcript_position'] = df['transcript_position'].astype(int)
    df_combined = pd.merge(df, data_info, on = ['transcript_id','transcript_position'])
    
    # Split on gene id
    unique_gene_ids = pd.Series(df_combined['gene_id'].unique())
    num_unique = len(unique_gene_ids)
    train_gene_id = unique_gene_ids.sample(int(num_unique*(1-ratio)), random_state=4266)
    df_train = df_combined[df_combined['gene_id'].isin(train_gene_id)].reset_index(drop = True)
    df_test  = df_combined[~df_combined['gene_id'].isin(train_gene_id)].reset_index(drop = True)
   
    #Sanity Check
    print(f"train_test_split working: {df_train.shape[0]+df_test.shape[0]==df.shape[0]}")
    return df_train, df_test

def prepare_train_dataset(df_train, categorical_columns):
    """
    Split into X and y, fit_transform scaler and encoder objects
    """
    # Extract input and output features
    X_train = df_train.drop(columns = ['gene_id','transcript_id','transcript_position','nucleotide_seq','label'])
    y_train = df_train['label']
    
    #One Hot Encode these categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_categorical = X_train[categorical_columns]
    train_ohe_columns = encoder.fit_transform(X_train_categorical)
    train_ohe_df = pd.DataFrame(train_ohe_columns.toarray(), columns=encoder.get_feature_names_out(input_features=categorical_columns))
    # Join these columns back to the original dataframe, removing the original columns
    X_train_encoded = pd.concat([X_train.drop(columns = categorical_columns),train_ohe_df], axis = 1)
    # Scale input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    
    #Sanity check
    print(f"prepare_dataset_for_training working: {'label' not in X_train.columns}")

    return X_train_scaled, y_train, encoder, scaler

def prepare_test_dataset(df_test, categorical_columns, encoder, scaler):
    """
    Split into X and y, then use fitted scaler and encoder objects to transform
    """
    X_test = df_test.drop(columns = ['gene_id','transcript_id','transcript_position','nucleotide_seq','label'])
    y_test = df_test['label']

    #One hot encode test columns with fitted encoder
    X_test_categorical = X_test[categorical_columns]
    test_ohe_columns = encoder.transform(X_test_categorical)
    test_ohe_df = pd.DataFrame(test_ohe_columns.toarray(), columns=encoder.get_feature_names_out(input_features=categorical_columns))
    X_test_encoded = pd.concat([X_test.drop(columns = categorical_columns),test_ohe_df], axis = 1)

    #Scale Test dataset
    X_test_scaled = scaler.transform(X_test_encoded)

    #Sanity check
    print(f"prepare_dataset_for_testing working: {'label' not in X_test.columns}")

    return X_test_scaled, y_test

#Hyperparamters
INITIAL_LEARNING_RATE = 0.001
L2_REG_STRENGTH= 0.0001
BATCH_SIZE = 32 
ACTIVATION = "relu"
NUM_NODES = 64
PATIENCE = 20
EPOCHS = 200

def initialize_model(X_train_scaled):
    """
    Initialize Model Object
    """
    adam_optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
    
    model = keras.Sequential([
      keras.layers.Input(shape=(X_train_scaled.shape[1],)), 
      keras.layers.Dense(NUM_NODES, activation=ACTIVATION,kernel_regularizer=l2(L2_REG_STRENGTH)),  
      keras.layers.Dense(NUM_NODES, activation=ACTIVATION,kernel_regularizer=l2(L2_REG_STRENGTH)),  
      keras.layers.Dense(1, activation='sigmoid') 
    ])
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy')
    return model

def calculate_class_weights(y_train):
    """
    Calculate class weights for training using imbalanced dataset
    """
    neg = y_train.value_counts()[0]
    pos = y_train.value_counts()[1]
    total = neg+pos
    #Positive/minority class will have higher weights in the loss function
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}
    #Sanity Check
    print(f"calculate_class_weights working: {weight_for_0<weight_for_1}")
    return class_weights


class EarlyStoppingByScore(keras.callbacks.Callback):
    """
    Early stopping callback class. Early stop when average of ROCAUC and AP does not improve on test dataset
    """
    def __init__(self, validation_data, patience=10, restore_best_weights=True):
        super(EarlyStoppingByScore, self).__init__()
        self.validation_data = validation_data
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_score = -1
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)
        ap = average_precision_score(y_val, y_pred)
        roc = roc_auc_score(y_val, y_pred)
        #Average of AP and ROC
        score = 0.5*(ap+roc)

        if score > self.best_score:
            self.best_score = score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping due to no improvement in Score for {self.patience} epochs.")
                self.model.stop_training = True
                if self.restore_best_weights:
                    print("Restoring best weights.")
                    self.model.set_weights(self.best_weights)
                
def fit_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Fit model with early stopping
    """
    early_stopping = EarlyStoppingByScore(validation_data=(X_test_scaled, y_test), patience=PATIENCE)
    class_weights = calculate_class_weights(y_train)
    model.fit(X_train_scaled,y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True, class_weight = class_weights, callbacks = [early_stopping])
    return model

def main():
    parser = argparse.ArgumentParser(description = "Train a ML model to predict m6A modification")
    parser.add_argument("json_data_dir",help = "File path for RNA-seq data (.json)")
    parser.add_argument("data_info_dir",help = "File path for m6A labels (.info)")
    args = parser.parse_args()
    print("=====Preprocessing JSON data=====")
    dictlist = parse_json_data(args.json_data_dir)
    df = summarise_json_data(dictlist)
    df, categorical_columns = encode_nucleotides(df)
    print("=====Training Model=====")
    data_info = pd.read_csv(args.data_info_dir)
    df_train, df_test = train_test_split(df, data_info)
    X_train_scaled, y_train, fitted_encoder , fitted_scaler = prepare_train_dataset(df_train,  categorical_columns)
    X_test_scaled, y_test = prepare_test_dataset(df_test, categorical_columns, fitted_encoder, fitted_scaler)
    model = initialize_model(X_train_scaled)
    fitted_model = fit_model(model,X_train_scaled, y_train, X_test_scaled, y_test)
    #Save all output objects
    fitted_model.save('fitted_model.h5',save_format='h5')
    dump(fitted_scaler,open('fitted_scaler.pkl','wb'))
    dump(fitted_encoder,open('fitted_encoder.pkl','wb'))

if __name__ == "__main__":
    main()