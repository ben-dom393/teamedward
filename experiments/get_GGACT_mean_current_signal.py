import json
from tqdm import tqdm

# Get all the means of GGACT 5mer
def subset_json_data(data_json_dir):
    jsondict = {}
    with open(data_json_dir) as file:
        for jsonobj in tqdm(file):
            dic = json.loads(jsonobj)
            for transcript_id in dic:
                for pos in dic[transcript_id]:
                    for k_mer in dic[transcript_id][pos]:
                        if k_mer[1:6] != "GGACT":
                            continue
                        #index 5 => mean of centre
                        mean_data = list(map(lambda x: x[5],dic[transcript_id][pos][k_mer]))
                        jsondict[f"{transcript_id},{pos}"] = mean_data
    return jsondict

datasets = ['SGNex_A549_directRNA_replicate5_run1_data','SGNex_K562_directRNA_replicate5_run1_data',
            'SGNex_K562_directRNA_replicate4_run1_data',"SGNex_K562_directRNA_replicate6_run1_data",
            "SGNex_MCF7_directRNA_replicate3_run1_data","SGNex_MCF7_directRNA_replicate4_run1_data",
            "SGNex_Hct116_directRNA_replicate3_run4_data","SGNex_Hct116_directRNA_replicate3_run1_data",
            "SGNex_A549_directRNA_replicate6_run1_data","SGNex_Hct116_directRNA_replicate4_run3_data",
            "SGNex_HepG2_directRNA_replicate6_run1_data","SGNex_HepG2_directRNA_replicate5_run2_data"]

for i in range(len(datasets)):
    fname = datasets[i]
    print(f"Currently at {fname}, index {i}")
    print("Parsing json data")
    jsondict = subset_json_data(f"../data/sgnex/raw_json/{fname}.json")
    print(f"Saving to json")
    with open(f"../data/sgnex/plotting/{fname}.json", "w") as file:
        json.dump(jsondict,file)
