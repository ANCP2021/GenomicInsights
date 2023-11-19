import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# functions start
def tR_tL(dataframe):
    tL = []
    tR = []
    for i in dataframe:
        tL.append((dataframe[i] == 1).sum())
        tR.append((dataframe[i] == 0).sum())
    
    return [tL, tR]

def cancer_tL(untamp_df):
    tL_C = []
    tL_NC = []

    for i in untamp_df:
        sample_list = list(untamp_df.loc[untamp_df[i] == 1, "Unnamed: 0"])
        tL_NC.append(len([j for j in sample_list if "NC" in j]))
        tL_C.append((len(sample_list)) - (len([j for j in sample_list if "NC" in j])))

    tL_C.pop(0)
    tL_NC.pop(0)

    return [tL_C, tL_NC]

def cancer_tR(untamp_df):
    tR_C = []
    tR_NC = []

    for i in untamp_df:
        sample_list = list(untamp_df.loc[untamp_df[i] == 0, "Unnamed: 0"])
        tR_NC.append(len([j for j in sample_list if "NC" in j]))
        tR_C.append((len(sample_list)) - (len([j for j in sample_list if "NC" in j])))

    tR_C.pop(0)
    tR_NC.pop(0)

    return [tR_C, tR_NC]

def gen_phi(dataframe):
    untampered_mutations_dataframe = dataframe.copy()
    feature_table = pd.DataFrame()

    original_mutations_dataframe = dataframe.copy()

    original_mutations_dataframe.pop(dataframe.columns[0])

    mutation_names = list(original_mutations_dataframe)
    feature_table["Mutation"] = mutation_names # gets mutation names

    childern_nodes = tR_tL(original_mutations_dataframe) # values of n(tL) and n(tR)
    feature_table["n(tL)"] = childern_nodes[0]
    feature_table["n(tR)"] = childern_nodes[1]

    left_child_cORnc = cancer_tL(untampered_mutations_dataframe) # values of n(tL, C) and n(tL, NC)
    feature_table["n(tL, C)"] = left_child_cORnc[0]
    feature_table["n(tL, NC)"] = left_child_cORnc[1]

    right_child_cORnc = cancer_tR(untampered_mutations_dataframe) # values of n(tR, C) and n(tR, NC)
    feature_table["n(tR, C)"] = right_child_cORnc[0]
    feature_table["n(tR, NC)"] = right_child_cORnc[1]

    feature_table["P(L)"] = (feature_table["n(tL)"] / (feature_table["n(tL)"] + feature_table["n(tR)"])) # P(L) value
    feature_table["P(R)"] = (feature_table["n(tR)"] / (feature_table["n(tL)"] + feature_table["n(tR)"])) # P(R) value

    feature_table["P(C | tL)"] = (feature_table["n(tL, C)"] / feature_table["n(tL)"]) # P(C | tL) value
    feature_table["P(NC | tL)"] = (feature_table["n(tL, NC)"] / feature_table["n(tL)"]) # P(NC | tL) value

    feature_table["P(C | tR)"] = (feature_table["n(tR, C)"] / feature_table["n(tR)"]) # P(C | tR) value
    feature_table["P(NC | tR)"] = (feature_table["n(tR, NC)"] / feature_table["n(tR)"]) # P(NC | tR) value

    feature_table["Q(s | t)"] = (abs(feature_table["P(C | tL)"] - feature_table["P(C | tR)"])) + (abs(feature_table["P(NC | tL)"] - feature_table["P(NC | tR)"])) # Q(s|t) value

    feature_table["Phi(s, t)"] = (2 * feature_table["P(L)"] * feature_table["P(R)"]) * (feature_table["Q(s | t)"]) # Phi function value

    top_feature_table = feature_table.nlargest(1, ["Phi(s, t)"])
    top_feature = top_feature_table["Mutation"].iat[0]
    return top_feature

def get_samples(dataframe):
    random_df = dataframe.sample(frac=1)
    split_dataframe = np.array_split(random_df, 3)
    return split_dataframe

def n_largest(dataframe, mutation, calc, n, TP_FP_list):
    top_TP_FP = pd.DataFrame(columns=[mutation, calc])
    top_TP_FP[mutation] = dataframe.columns
    top_TP_FP[calc] = TP_FP_list
    top_TP_FP = top_TP_FP.nlargest(n, calc)
    return top_TP_FP

def create_bin_tree(training_set, node_Root):
    original_mutations_dataframe = training_set.copy()

    group_A = original_mutations_dataframe.loc[original_mutations_dataframe[node_Root] == 1, "Unnamed: 0"].values.T.tolist()
    group_B = original_mutations_dataframe.loc[original_mutations_dataframe[node_Root] == 0, "Unnamed: 0"].values.T.tolist()

    groupA_df = original_mutations_dataframe[original_mutations_dataframe["Unnamed: 0"].isin(group_A)]
    node_A = gen_phi(groupA_df)


    groupB_df = original_mutations_dataframe[original_mutations_dataframe["Unnamed: 0"].isin(group_B)]
    node_B= gen_phi(groupB_df)

    return [node_A, node_B]

def get_subgroups(training_set, group_node):
    original_mutations_dataframe = training_set.copy()

    right_node_list = original_mutations_dataframe.loc[original_mutations_dataframe[group_node] == 1, "Unnamed: 0"].values.T.tolist()
    left_node_list = original_mutations_dataframe.loc[original_mutations_dataframe[group_node] == 0, "Unnamed: 0"].values.T.tolist()

    sub1_nc = sum('NC' in s for s in right_node_list)
    sub1_c = len(right_node_list) - sub1_nc
    if sub1_c >= sub1_nc:
        sub1 = "Cancer"
    else:
        sub1 = "Non-Cancer"

    
    sub2_nc = sum('NC' in s for s in left_node_list)
    sub2_c = len(left_node_list) - sub2_nc
    if sub2_c >= sub2_nc:
        sub2 = "Cancer"
    else:
        sub2 = "Non-Cancer"

    return [sub1, sub2]

def eval(sample_file, overall_top_feature, groupA_top_feature, groupB_top_feature, overall, training):
    true_negative_TN = 0
    true_positive_TP = 0
    false_negative_FN = 0
    false_positive_FP = 0

    group_A_subgroups = get_subgroups(training, groupA_top_feature)
    a1 = group_A_subgroups[0]
    a2 = group_A_subgroups[1]
    print("A1:", a1)
    print("A2:", a2)

    group_B_subgroups = get_subgroups(training, groupB_top_feature)
    b1 = group_B_subgroups[0]
    b2 = group_B_subgroups[1]
    print("B1:", a1)
    print("B2:", a2)

    for sample in sample_file:
        row_index = overall.index[overall["Unnamed: 0"] == sample]
        if overall.iat[row_index[0], overall.columns.get_loc(overall_top_feature)] == 1:
            if overall.iat[row_index[0], overall.columns.get_loc(groupA_top_feature)] == 1:
                if a1 == "Cancer":
                    if sample.startswith("NC"):
                        false_positive_FP += 1
                    else:
                        true_positive_TP += 1
                else:
                    if sample.startswith("NC"):
                        true_negative_TN += 1
                    else:
                        false_negative_FN += 1
            else:
                if a2 == "Cancer":
                    if sample.startswith("NC"):
                        false_positive_FP += 1
                    else:
                        true_positive_TP += 1
                else:
                    if sample.startswith("NC"):
                        true_negative_TN += 1
                    else:
                        false_negative_FN += 1
        else:
            if overall.iat[row_index[0], overall.columns.get_loc(groupB_top_feature)] == 1:
                if b1 == "Cancer":
                    if sample.startswith("NC"):
                        false_positive_FP += 1
                    else:
                        true_positive_TP += 1
                else:
                    if sample.startswith("NC"):
                        true_negative_TN += 1
                    else:
                        false_negative_FN += 1
            else:
                if b2 == "Cancer":
                    if sample.startswith("NC"):
                        false_positive_FP += 1
                    else:
                        true_positive_TP += 1
                else:
                    if sample.startswith("NC"):
                        true_negative_TN += 1
                    else:
                        false_negative_FN += 1

    return [true_negative_TN, false_positive_FP, false_negative_FN, true_positive_TP]

def metrics_calculate(metric_list):
    true_negative_TN = metric_list[0]
    false_positive_FP = metric_list[1]
    false_negative_FN = metric_list[2]
    true_positive_TP = metric_list[3]

    accuracy = (true_positive_TP + true_negative_TN) / (true_positive_TP + true_negative_TN + false_positive_FP + false_negative_FN)
    print("Accuracy:", round(accuracy, 2))
    sensitivity = true_positive_TP / (true_positive_TP + false_negative_FN)
    print("Sensitivity:", round(sensitivity, 2))
    specificity = true_negative_TN / (true_negative_TN + false_positive_FP)
    print("Specificity:", round(specificity, 2))
    precision = true_positive_TP / (true_positive_TP + false_positive_FP)
    print("Precision:", round(precision, 2))
    miss_rate = false_negative_FN / (false_negative_FN + true_positive_TP)
    print("Miss Rate:", round(miss_rate, 2))
    false_discorvery_rate = false_positive_FP / (false_positive_FP + true_positive_TP)
    print("False Discovery Rate:", round(false_discorvery_rate, 2))
    false_omission_rate = false_negative_FN / (false_negative_FN + true_negative_TN)
    print("False Omission Rate:", round(false_omission_rate, 2))

    return accuracy
    
# functions end

# convert excel to dataframe
excel_mutations = pd.read_excel('mutations.xlsx')
mutations_dataframe = pd.DataFrame(excel_mutations)
original_mutations_dataframe = pd.DataFrame(excel_mutations)

set_array = get_samples(mutations_dataframe)

training_set1 = pd.concat([set_array[0], set_array[1]], ignore_index=True)
testing_set1 = set_array[2]
testing_sample_set1 = list(testing_set1["Unnamed: 0"])

training_set2 = pd.concat([set_array[1], set_array[2]], ignore_index=True)
testing_set2 = set_array[0]
testing_sample_set2 = list(testing_set2["Unnamed: 0"])

training_set3 = pd.concat([set_array[0], set_array[2]], ignore_index=True)
testing_set3 = set_array[1]
testing_sample_set3 = list(testing_set3["Unnamed: 0"])

train = [training_set1, training_set2, training_set3]
test = [testing_sample_set1, testing_sample_set2, testing_sample_set3]


for i in range(0, 3):
    root_node = gen_phi(train[i])
    nodes = create_bin_tree(train[i], root_node)
    right_child_node = nodes[0]
    left_child_node = nodes[1]
    print("Root Node:", root_node)
    print("Right Child Node:", right_child_node)
    print("Left Child Node:", left_child_node)

    vals_TP_FP = eval(test[i], root_node, right_child_node, left_child_node, original_mutations_dataframe, train[i])
    metrics_calculate(vals_TP_FP)
    print("*****************************************************************")
