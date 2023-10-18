import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def parse_json_data(data_json_dir):
    jsonlist = []
    with open(data_json_dir) as file:
        for jsonobj in file:
            jsonlist.append(json.loads(jsonobj))
    return jsonlist

def process_json_data(data_json_dir, mode):
    jsonlist = parse_json_data(data_json_dir)
    if mode == "old":
        #do old one
        df_dict = {'transcript_id' : [], 'transcript_position' : [], '5-mers': [],'readings':[]}
        for i in tqdm(jsonlist):
            transcript_id = list(i.keys())[0]
            transcript_position = list(i[transcript_id].keys())[0]
            five_mer = list(i[transcript_id][transcript_position].keys())[0]
            readings = list(i[transcript_id][transcript_position][five_mer])
            df_dict['transcript_id'].append(transcript_id)
            df_dict['transcript_position'].append(transcript_position)
            df_dict['5-mers'].append(five_mer)
            df_dict['readings'].append(readings)
        df = pd.DataFrame(df_dict)

    elif mode == "new":
        #do xg latest jsondata manipulation
        df_list = []
        count = 0
        for i in tqdm(jsonlist):
            count+=1
            transcript_id = list(i.keys())[0]
            transcript_position = list(i[transcript_id].keys())[0]
            five_mer = list(i[transcript_id][transcript_position].keys())[0]
            readings = i[transcript_id][transcript_position][five_mer]
            df = pd.DataFrame(readings)
            df.columns = ['dwell_time_-1','sd_-1','mean_-1','dwell_time_0','sd_0','mean_0','dwell_time_1','sd_1','mean_1']
            df['product_mean_dwell_-1'] = df['dwell_time_-1']*df['mean_-1']
            df['product_mean_dwell_0'] = df['dwell_time_0']*df['mean_0']
            df['product_mean_dwell_1'] = df['dwell_time_1']*df['mean_1']
            df['product_var_dwell_-1'] = df['sd_-1']*df['sd_-1']*(df['dwell_time_-1'])
            df['product_var_dwell_0'] = df['sd_0']*df['sd_0']*(df['dwell_time_0'])
            df['product_var_dwell_1'] = df['sd_1']*df['sd_1']*(df['dwell_time_1'])

            weighted_mean_neg1 = df['product_mean_dwell_-1'].sum()/df['dwell_time_-1'].sum()
            weighted_mean_0 = df['product_mean_dwell_0'].sum()/df['dwell_time_0'].sum()
            weighted_mean_1 = df['product_mean_dwell_1'].sum()/df['dwell_time_1'].sum()

            weighted_sd_neg1 = np.sqrt(df['product_var_dwell_-1'].sum()/df['dwell_time_-1'].sum())
            weighted_sd_0 = np.sqrt(df['product_var_dwell_0'].sum()/df['dwell_time_0'].sum())
            weighted_sd_1 = np.sqrt(df['product_var_dwell_1'].sum()/df['dwell_time_1'].sum())
            
            mean_25_neg1 = df['mean_-1'].quantile(0.25)
            mean_25_0 = df['mean_0'].quantile(0.25)
            mean_25_1 = df['mean_1'].quantile(0.25)

            mean_50_neg1 = df['mean_-1'].quantile(0.5)
            mean_50_0 = df['mean_0'].quantile(0.5)
            mean_50_1 = df['mean_1'].quantile(0.5)

            mean_75_neg1 = df['mean_-1'].quantile(0.75)
            mean_75_0 = df['mean_0'].quantile(0.75)
            mean_75_1 = df['mean_1'].quantile(0.75)
            
            df_list.append([transcript_id,transcript_position,five_mer,weighted_mean_neg1,weighted_mean_0,weighted_mean_1,weighted_sd_neg1,
                            weighted_sd_0,weighted_sd_1,mean_25_neg1,mean_25_0,mean_25_1,mean_50_neg1,mean_50_0,mean_50_1,mean_75_neg1,
                            mean_75_0,mean_75_1])
            """
            df_list.append([transcript_id,transcript_position,five_mer,weighted_mean_neg1,weighted_mean_0,weighted_mean_1,weighted_sd_neg1,
                            weighted_sd_0,weighted_sd_1])
            """
        df = pd.DataFrame(df_list)
        df.columns = ['transcript_id','transcript_position','five_mer','weighted_mean_neg1','weighted_mean_0','weighted_mean_1','weighted_sd_neg1',
                'weighted_sd_0','weighted_sd_1','mean_25_neg1','mean_25_0','mean_25_1','mean_50_neg1','mean_50_0','mean_50_1','mean_75_neg1',
                    'mean_75_0','mean_75_1']
    return df


def train_test_split_by_geneid(data_df, data_label_df, train_test_split_proportion):
        unique_ids = pd.Series(data_label_df['gene_id'].unique())
        train_gene_id = unique_ids.sample(int(len(unique_ids)*train_test_split_proportion), random_state = 4266)
        y_train = data_label_df[data_label_df['gene_id'].isin(train_gene_id)]
        y_test = data_label_df[~data_label_df['gene_id'].isin(train_gene_id)]
        x_train = data_df[data_df['transcript_id'].isin(y_train['transcript_id'])]
        x_train = x_train.sample(frac=1).reset_index(drop=True) # Shuffle training data
        x_test = data_df[~data_df['transcript_id'].isin(y_train['transcript_id'])]
        return x_train, x_test, y_train, y_test

def xg_generate_features_old(data_df, data_label_info):
    data_df[[f"readings_{i}" for i in range(9)]] = pd.DataFrame(data_df['readings'].tolist(), index = data_df.index)
    data_df.drop(columns = 'readings', inplace = True)
    new_columns = ['dwell_time_-1','sd_-1','mean_-1','dwell_time_0','sd_0','mean_0','dwell_time_1','sd_1','mean_1']
    data_df = data_df.rename(columns=dict(zip(list(data_df.columns)[3:], new_columns)))
    data_df = data_df.drop(columns = [i for i in data_df.columns if 'weight' in i])
    data_df['product_mean_dwell_-1'] = data_df['dwell_time_-1']*data_df['mean_-1']
    data_df['product_mean_dwell_0'] = data_df['dwell_time_0']*data_df['mean_0']
    data_df['product_mean_dwell_1'] = data_df['dwell_time_1']*data_df['mean_1']
    data_df['product_var_dwell_-1'] = data_df['sd_-1']*data_df['sd_-1']*(data_df['dwell_time_-1'])
    data_df['product_var_dwell_0'] = data_df['sd_0']*data_df['sd_0']*(data_df['dwell_time_0'])
    data_df['product_var_dwell_1'] = data_df['sd_1']*data_df['sd_1']*(data_df['dwell_time_1'])
    data_df['count'] = 1
    new_df = data_df.groupby(['transcript_id','transcript_position','5-mers']).agg({'dwell_time_-1':'sum','dwell_time_0':'sum','dwell_time_1':'sum',
                                                                    'product_mean_dwell_-1':'sum','product_mean_dwell_0':'sum','product_mean_dwell_1':'sum',
                                                                    'product_var_dwell_-1':'sum','product_var_dwell_0':'sum','product_var_dwell_1':'sum','count':'sum'}).reset_index()
    new_df['mean_dwell_time_-1'] = new_df['dwell_time_-1']/new_df['count']
    new_df['mean_dwell_time_0'] = new_df['dwell_time_0']/new_df['count']
    new_df['mean_dwell_time_1'] = new_df['dwell_time_1']/new_df['count']
    new_df['weighted_mean_-1'] = new_df['product_mean_dwell_-1']/new_df['dwell_time_-1']
    new_df['weighted_mean_0']= new_df['product_mean_dwell_0']/new_df['dwell_time_0']
    new_df['weighted_mean_1']= new_df['product_mean_dwell_1']/new_df['dwell_time_1']
    new_df['weighted_sd_-1'] = np.sqrt(new_df['product_var_dwell_-1']/(new_df['dwell_time_-1']))
    new_df['weighted_sd_0']= np.sqrt(new_df['product_var_dwell_0']/(new_df['dwell_time_0']))
    new_df['weighted_sd_1']= np.sqrt(new_df['product_var_dwell_1']/(new_df['dwell_time_1']))
    new_df = new_df.drop(columns = ['dwell_time_-1',
        'dwell_time_0', 'dwell_time_1', 'product_mean_dwell_-1',
        'product_mean_dwell_0', 'product_mean_dwell_1', 'product_var_dwell_-1',
        'product_var_dwell_0', 'product_var_dwell_1', 'count'])
    new_df['5-mer-0'] = new_df['5-mers'].map(lambda x:x[0])
    new_df['5-mer-1'] = new_df['5-mers'].map(lambda x:x[1])
    new_df['5-mer-2'] = new_df['5-mers'].map(lambda x:x[2])
    new_df['5-mer-5'] = new_df['5-mers'].map(lambda x:x[5])
    new_df['5-mer-6'] = new_df['5-mers'].map(lambda x:x[6])
    new_df = new_df.drop(columns = ['5-mers'])
    new_df_cat = new_df[['5-mer-0','5-mer-1','5-mer-2','5-mer-5','5-mer-6']]
    encoder = OneHotEncoder()
    one_hot_encoded = encoder.fit_transform(new_df_cat)
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded.toarray(), columns=encoder.get_feature_names_out(input_features=new_df_cat.columns))
    data_encoded = pd.concat([new_df.drop(columns=new_df_cat.columns), one_hot_encoded_df], axis=1)
    data_encoded['transcript_position'] = data_encoded['transcript_position'].astype(int)
    data_encoded = pd.merge(data_encoded, data_label_info, on = ['transcript_id','transcript_position'])
    return data_encoded


def xg_generate_features_new(data_df, data_label_info):
    data_df['5-mer-0'] = data_df['five_mer'].map(lambda x:x[0])
    data_df['5-mer-1'] = data_df['five_mer'].map(lambda x:x[1])
    data_df['5-mer-2'] = data_df['five_mer'].map(lambda x:x[2])
    data_df['5-mer-5'] = data_df['five_mer'].map(lambda x:x[5])
    data_df['5-mer-6'] = data_df['five_mer'].map(lambda x:x[6])
    data_df['5-mer_window-1'] = data_df['five_mer'].map(lambda x: x[0:5])
    data_df['5-mer_window0'] = data_df['five_mer'].map(lambda x: x[1:6])
    data_df['5-mer_window1'] = data_df['five_mer'].map(lambda x: x[2:7])
    data_df_cat = data_df[['5-mer-0','5-mer-1','5-mer-2','5-mer-5','5-mer-6','5-mer_window-1', '5-mer_window0', '5-mer_window1']]
    encoder = OneHotEncoder()
    one_hot_encoded = encoder.fit_transform(data_df_cat)
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded.toarray(), columns=encoder.get_feature_names_out(input_features=data_df_cat.columns))
    data_encoded = pd.concat([data_df.drop(columns=data_df_cat.columns), one_hot_encoded_df], axis=1)
    data_encoded['A_freq'] = data_encoded['five_mer'].map(lambda x:x.count('A'))
    data_encoded['C_freq'] = data_encoded['five_mer'].map(lambda x:x.count('C'))
    data_encoded['G_freq'] = data_encoded['five_mer'].map(lambda x:x.count('G'))
    data_encoded['T_freq'] = data_encoded['five_mer'].map(lambda x:x.count('T'))
    data_encoded['transcript_position'] = data_encoded['transcript_position'].astype(int)
    data_encoded = pd.merge(data_encoded, data_label_info, on = ['transcript_id','transcript_position'])
    return data_encoded

def xg_dataprep(data_json_dir, data_info_dir, output_dir, mode):
    #Read in data_info and json_data
    data_info = pd.read_csv(data_info_dir)
    df = process_json_data(data_json_dir, mode)

    #Train - test split by geneid
    train_test_split_proportion = 0.8
    df_train, df_valid,  data_info_train, data_info_valid = train_test_split_by_geneid(df, data_info, train_test_split_proportion)

    if mode == "old":
        df_train = xg_generate_features_old(df_train, data_info_train)
        df_valid = xg_generate_features_old(df_valid, data_info_valid)
    elif mode == "new":
        df_train = xg_generate_features_new(df_train, data_info_train)
        df_valid = xg_generate_features_new(df_valid, data_info_valid)
    
    
    df_valid.to_csv(f'{output_dir}/label_df_valid.csv', index = False)
    df_train.to_csv(f'{output_dir}/label_df_train.csv', index = False)
    return

def zac_generate_features(data_df, data_label_info, is_test):
    def calculate_mean(row):
        readings_array = np.array(row)
        return np.mean(readings_array, axis=0).tolist()

    min_length = 99999
    for idx, row in data_df.iterrows():
        length = len(row[3])
        if length < min_length:
            min_length = length
    full_data = data_df.merge(data_label_info, on = ['transcript_position','transcript_id'])
    positive_data = full_data[full_data['label']=='1']
    negative_data = full_data[full_data['label']=='0']

    if not is_test:
        random.seed(4266)
        ## Control sample size using n.  n <= 20
        n = min(15, min_length)
        row_list = []
        # create new positive data until we achieve 1:1 ratio of positive to nagatives
        while len(row_list) < len(negative_data):
        # sample a random row
            row = positive_data.sample(n=1)
            # sample n readings and create a new row for the new positive data
            sample_reads = random.sample(row['readings'].tolist()[0],n)
            t_id = row['transcript_id'].tolist()[0]
            t_pos = row['transcript_position'].tolist()[0]
            fmer = row['5-mers'].tolist()[0]
            gene_id = row['gene_id'].tolist()[0]
            label = row['label'].tolist()[0]
            row_dict = {'transcript_id':t_id, 'transcript_position':t_pos, '5-mers': fmer, 'readings':sample_reads, 'gene_id':gene_id, 'label':label}
            row_list.append(row_dict)

        new_positive_data = pd.DataFrame(row_list)

        df = pd.concat([new_positive_data, negative_data], ignore_index=True)
    else:
        df = full_data

    df['readings'] = df['readings'].apply(calculate_mean)

    # Split the "readings" column into separate columns
    split_readings = df['readings'].apply(lambda x: pd.Series(x))
    split_readings.columns = [f'value_{i}' for i in range(9)]

    # Concatenate the split columns with the original DataFrame
    df = pd.concat([df, split_readings], axis=1)
    # Drop readings
    df.drop(columns='readings',inplace = True)

    column_mapping = {
        'value_0': 'dwell_time_-1',
        'value_1': 'sd_-1',
        'value_2': 'mean_-1',
        'value_3': 'dwell_time_0',
        'value_4': 'sd_0',
        'value_5': 'mean_0',
        'value_6': 'dwell_time_1',
        'value_7': 'sd_1',
        'value_8': 'mean_1'
    }

    df = df.rename(columns=column_mapping)
    # df = df.drop(columns = ["label","transcript_id","gene_id"])
    df = df.drop(columns = ["transcript_id","gene_id"])#try adding in transcript_position to remove also
    # Define the possible gene types
    gene_types = ['A', 'C', 'T', 'G']

    # Create a one-hot encoding for each position and gene type
    for position in range(7):
        for gene_type in gene_types:
            col_name = f'5-mer-{position}_{gene_type}'
            df[col_name] = (df['5-mers'].str[position] == gene_type).astype(int)  # Convert to 1 or 0

    # Drop the original "5-mers" column and other useless positions
    columns_to_drop = ['5-mers', '5-mer-3_A', '5-mer-3_C', '5-mer-3_T', '5-mer-3_G',
                   '5-mer-4_A', '5-mer-4_C', '5-mer-4_T', '5-mer-4_G',
                   '5-mer-1_C', '5-mer-2_C', '5-mer-2_T', '5-mer-5_G',]

    # Drop the specified columns
    df = df.drop(columns=columns_to_drop)
    return df


def zac_dataprep(data_json_dir, data_info_dir, output_dir):
    #Read in data_info and json_data
    df = process_json_data(data_json_dir, "old")

    data_info = pd.read_csv(data_info_dir)

    #Train - test split by geneid
    train_test_split_proportion = 0.8
    df_train, df_valid,  data_info_train, data_info_valid = train_test_split_by_geneid(df, data_info, train_test_split_proportion)
    df_train = zac_generate_features(df_train, data_info_train, False)
    df_valid = zac_generate_features(df_valid, data_info_valid, True)

    df_valid.to_csv(f'{output_dir}/label_df_valid.csv', index = False)
    df_train.to_csv(f'{output_dir}/label_df_train.csv', index = False)
    return


def train_model(data_dir, sklearn_model):
    train_df = pd.read_csv(data_dir)
    x_train, y_train = train_df[[i for i in train_df.columns if i!='label']],train_df['label']
    sklearn_model.fit(x_train,y_train)
    return sklearn_model

def evaluate_model(y_pred, y_test):
    ## ROC curve
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred)

    # Get ROC curve values
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    ## PR AUC curve
    # Calculate PR AUC
    pr_auc = average_precision_score(y_test, y_pred)

    # Get precision-recall curve values
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PR AUC = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()

    pr_auc_score = auc(recall,precision)
    print(f'PR-AUC score : {pr_auc_score}')
    print(f"ROC-AUC score : {roc_auc}")
    print(f'Average of both scores : {(pr_auc_score+roc_auc)/2:.4f}')



def full_pipeline(data_json_dir, data_info_dir, output_dir,sklearn_model, who_mode):
    if who_mode =="xg_old":
        xg_dataprep(data_json_dir, data_info_dir, output_dir, "old")
    elif who_mode == "xg_new":
        xg_dataprep(data_json_dir, data_info_dir, output_dir, "new")
    elif who_mode == "zac":
        zac_dataprep(data_json_dir, data_info_dir, output_dir)
    
    sklearn_model = train_model(f"{output_dir}/label_df_train.csv",sklearn_model)

    #evaluate model and plot graphs
    test_df = pd.read_csv(f"{output_dir}/label_df_valid.csv")
    x_test,y_test = test_df[[i for i in test_df.columns if i!='label']],test_df['label']
    # y_pred = sklearn_model.predict(x_test)
    y_pred_prob = sklearn_model.predict_proba(x_test)
    evaluate_model(y_pred_prob[:, 1], y_test)

    return sklearn_model
