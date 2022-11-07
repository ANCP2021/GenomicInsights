import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# functions start
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

def get_unnamed_binaries(col_list):
    binary = []
    for feature in col_list:
        if feature.startswith("C"):
            binary.append(1)
        else:
            binary.append(0)
    return binary

def calc_TPFP_values(dataframe, binary):
    diff_TP_FP_list = []
    for col in dataframe:
        mutation_list = dataframe[col].to_list()
        mutation_confusion_matrix = pd.crosstab(binary, mutation_list, rownames=['Actual'], colnames=['Predicted'], dropna=False)
        mutation_confusion_matrix = mutation_confusion_matrix.reindex(columns=[0,1], fill_value=0)
        difference_TP_FP = mutation_confusion_matrix[1][1] - mutation_confusion_matrix[1][0]
        diff_TP_FP_list.append(difference_TP_FP)
    return diff_TP_FP_list

def create_bin_tree(training_set):
    original_mutations_dataframe = training_set.copy()

    actual = training_set["Unnamed: 0"].to_list()
    actual_binary = get_unnamed_binaries(actual)
    training_set.pop(training_set.columns[0])

    top_10_TP_FP = pd.DataFrame(columns=['Mutation', "TP-FP"])
    diff_TP_FP_list = []

    for col in training_set:
        mutation_list = training_set[col].to_list()
        mutation_confusion_matrix = pd.crosstab(actual_binary, mutation_list, rownames=['Actual'], colnames=['Predicted'], dropna=False)
        mutation_confusion_matrix = mutation_confusion_matrix.reindex(columns=[0,1], fill_value=0)
        difference_TP_FP = mutation_confusion_matrix[1][1] - mutation_confusion_matrix[1][0]
        diff_TP_FP_list.append(difference_TP_FP)

    top_10_TP_FP["Mutation"] = training_set.columns
    top_10_TP_FP["TP-FP"] = diff_TP_FP_list
    node_Root = top_10_TP_FP.nlargest(1, "TP-FP")
    node_Root = node_Root["Mutation"].iat[0]

    group_A = original_mutations_dataframe.loc[original_mutations_dataframe[node_Root] == 1, "Unnamed: 0"].values.T.tolist()
    group_B = original_mutations_dataframe.loc[original_mutations_dataframe[node_Root] == 0, "Unnamed: 0"].values.T.tolist()

    group_A_binary = get_unnamed_binaries(group_A)
    groupA_df = original_mutations_dataframe[original_mutations_dataframe["Unnamed: 0"].isin(group_A)]
    groupA_df.pop(groupA_df.columns[0])
    diff_TP_FP_groupA_list = calc_TPFP_values(groupA_df, group_A_binary)
    top_TP_FP_groupA = n_largest(groupA_df, "Mutation", "TP-FP", 1, diff_TP_FP_groupA_list)
    node_A = top_TP_FP_groupA["Mutation"].iat[0]

    group_B_binary = get_unnamed_binaries(group_B)
    groupB_df = original_mutations_dataframe[original_mutations_dataframe["Unnamed: 0"].isin(group_B)]
    groupB_df.pop(groupB_df.columns[0])
    diff_TP_FP_groupB_list = calc_TPFP_values(groupB_df, group_B_binary)
    top_TP_FP_groupB = n_largest(groupB_df, "Mutation", "TP-FP", 1, diff_TP_FP_groupB_list)
    node_B = top_TP_FP_groupB["Mutation"].iat[0]

    return [node_Root, node_A, node_B]

def eval(sample_file, overall_top_feature, groupA_top_feature, groupB_top_feature, overall):
    true_negative_TN = 0
    true_positive_TP = 0
    false_negative_FN = 0
    false_positive_FP = 0

    for sample in sample_file:
        row_index = overall.index[overall["Unnamed: 0"] == sample]
        if overall.iat[row_index[0], overall.columns.get_loc(overall_top_feature)] == 1:
            if overall.iat[row_index[0], overall.columns.get_loc(groupA_top_feature)] == 1:
                # Cancer
                if sample.startswith("NC"):
                    false_positive_FP += 1
                else:
                    true_positive_TP += 1
            else:
                # Non-Cancer
                if sample.startswith("NC"):
                    true_negative_TN += 1
                else:
                    false_negative_FN += 1
        else:
            if overall.iat[row_index[0], overall.columns.get_loc(groupB_top_feature)] == 1:
                # Cancer
                if sample.startswith("NC"):
                    false_positive_FP += 1
                else:
                    true_positive_TP += 1
            else:
                # Non-Camcer
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

average_accuracy = 0
for i in range(0, 3):
    node_list = create_bin_tree(train[i])
    node_Root = node_list[0]
    node_A = node_list[1]
    node_B = node_list[2]

    vals_TP_FP = eval(test[i], node_Root, node_A, node_B, original_mutations_dataframe)
    average_accuracy += metrics_calculate(vals_TP_FP) 
    print("**************************************")

print("Average Accuracy:", average_accuracy / 3)