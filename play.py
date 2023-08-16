# import json

# data = {
#     # exp
#     "seed": 0,
#     "runs": 5,
#     "exp_name": "main_run5_cv5_21data_100trees_40opt_fast",
#     "path": "../../",
#     "split": "CV", #CV, random_split
#     "cv_folds": 5,
#     "plot": False,
#     "calib_methods": ["RF_d", "RF_opt",
#                       "Platt", "ISO", "Beta", "CRF", "VA",
#                       "tlr", "Rank", 
#                       # "RF_ens_k", "RF_ens_r", 
#                       "RF_large",
#                       ],
    
#     "metrics": ["acc", "brier", "ece", "logloss"],

#     # calib param
#     "bin_strategy": "uniform",
#     "ece_bins": 20,
#     "boot_size": 1000,
#     "boot_count": 5,

#     # RF hyper opt
#     "hyper_opt": True,
#     "opt_cv":5, 
#     "opt_n_iter":40,
#     "opt_top_K": 5,
#     "search_space": {
#                     "n_estimators": [100],
#                     "max_depth": [2,3,4,5,6,7,8,10,15,20,30,40,50,60,100],
#                     "criterion": ["gini", "entropy"],
#                     "max_features": ["sqrt", "log2"],
#                     "min_samples_split": [2,3,4,5],
#                     "min_samples_leaf": [1,2,3],
#                     "oob_score": [False]
#                     },
    
#     "n_estimators": 100,
#     "oob": False,
# }

# print("data before\n", data)

# # Specify the file name
# file_name = "test_dict_data.json"

# # Open the file for writing
# with open(file_name, "w") as json_file:
#     json.dump(data, json_file)

# with open(file_name, "r") as json_file:
#     loaded_data = json.load(json_file)
    
# print("---------------------------------")
# print("data after\n", loaded_data)

import time
import random
import string

def generate_readable_short_id():
    timestamp = int(time.time())  # Get current timestamp
    random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))  # Generate a random 4-character code
    short_id = f"{timestamp}_{random_chars}"
    return short_id

if __name__ == "__main__":
    readable_short_id = generate_readable_short_id()
    print("Generated Readable Short ID:", readable_short_id)



