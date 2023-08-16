import os
import time
import random
import string
import json
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import core_calib as cal


def generate_readable_short_id():
    timestamp = int(time.time())  # Get current timestamp
    random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))  # Generate a random 4-character code
    short_id = f"{timestamp}_{random_chars}"
    return short_id

def save_params(params):
    path = f"./results/{params['exp_name']}"
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = f"{path}/params.json"

    # Open the file for writing
    with open(file_name, "w") as json_file:
        json.dump(params, json_file)

def save_results(calib_results_dict, exp_name):
    path = f"./results/{exp_name}"
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = f"{path}/results.json"

    # Open the file for writing
    with open(file_name, "w") as json_file:
        json.dump(calib_results_dict, json_file)

def read_results(file_address):
    file_name = f"{file_address}/results.json"
    with open(file_name, "r") as json_file:
        loaded_data = json.load(json_file)

    file_name = f"{file_address}/params.json"
    with open(file_name, "r") as json_file:
        params = json.load(json_file)

    tables = cal.mean_and_ranking_table(loaded_data, 
                                    params["metrics"], 
                                    params["calib_methods"], 
                                    params["exp_values"], 
                                    mean_and_rank=False, 
                                    std=False)
    return loaded_data, tables, params



def save_metrics_to_csv(tables, metrics, exp_name):
    path = f"./results/{exp_name}/metrics"
    if not os.path.exists(path):
        os.makedirs(path)

    for metric in metrics:
        tables[metric].round(5).to_csv(f"{path}/{metric}.csv") 

    for metric in metrics:
        tables[metric+ "_std"].round(5).to_csv(f"{path}/{metric}_std.csv")

def save_metrics_to_latex(tables, metrics, exp_name):
    path = f"./results/{exp_name}/metrics"
    if not os.path.exists(path):
        os.makedirs(path)

    for metric in metrics:
        tables[metric].round(5).to_latex(f"{path}/{metric}.csv") 

    for metric in metrics:
        tables[metric+ "_std"].round(5).to_latex(f"{path}/{metric}_std.csv")

def res_statistics(tables, metrics, path):
    path += "/statistics_nf"

    # tables = cal.mean_and_ranking_table(calib_results_dict, 
    #                                     params["metrics"],
    #                                     calib_methods, 
    #                                     params["exp_values"], 
    #                                     mean_and_rank=True, 
    #                                     std=True)
    
    # tables = cal.make_table(calib_results_dict, 
    #                         params["metrics"],
    #                         calib_methods, 
    #                         params["exp_values"])

    for metric in metrics:

        df = pd.DataFrame(tables[metric])
        avg_rank = df.loc["Rank"]
        df = df.drop(["Mean", "Rank"])

        # Perform the Friedman test
        statistic, p_value = friedmanchisquare(*df.values.T)

        print(f"Friedman Test Statistic: {statistic}")
        print(f"P-value: {p_value}")

        # Check for significance (you can choose your significance level, e.g., 0.05)
        alpha = 0.05
        if p_value < alpha:
            print("The differences between groups are significant.")

            # posthoc_res = sp.posthoc_conover_friedman(df)

            posthoc_res = sp.posthoc_nemenyi_friedman(df)

            # w_df = np.array(df.T)
            # posthoc_res = sp.posthoc_wilcoxon(w_df, p_adjust="hommel")
            # posthoc_res.columns = df.columns
            # posthoc_res = posthoc_res.set_index(df.columns)

            # sp.sign_plot(posthoc_res)
            # plt.show()

            plt.figure(figsize=(10, 4), dpi=100)
            plt.title(f"Critical difference diagram of average score ranks ({metric})")    
            sp.critical_difference_diagram(avg_rank, posthoc_res)

            # path = f"./results/{params['exp_name']}/statistics"
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f"{path}/CD_{metric}.pdf", format='pdf', transparent=True)
            plt.close() 
        else:
            print("The differences between groups are not significant.")

def add_rank_mean(tabels):
    for metric in tabels.keys():
        mean_res = tabels[metric].mean()
        if metric == "ece" or metric == "brier" or metric == "tce" or metric == "logloss":
            df_rank = tabels[metric].rank(axis=1, ascending = True)
        else:
            df_rank = tabels[metric].rank(axis=1, ascending = False)

        mean_rank = df_rank.mean()
        tabels[metric].loc["Mean"] = mean_res
        tabels[metric].loc["Rank"] = mean_rank
    return tabels
