import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# functions start
def n_largest(dataframe, mutation, calc, n, TP_FP_list, excel_name):
    top_TP_FP = pd.DataFrame(columns=[mutation, calc])
    top_TP_FP[mutation] = dataframe.columns
    top_TP_FP[calc] = TP_FP_list
    top_TP_FP = top_TP_FP.nlargest(n, calc)
    top_TP_FP.to_excel(excel_name)

def calc_accuracy_values(dataframe, binary):
    accuracy_overall_list = []
    for col in dataframe:
        mutation_list = dataframe[col].to_list()
        mutation_confusion_matrix = pd.crosstab(binary, mutation_list, rownames=['Actual'], colnames=['Predicted'], dropna=False)
        mutation_confusion_matrix = mutation_confusion_matrix.reindex(columns=[0,1], fill_value=0)
        accuracy = (mutation_confusion_matrix[1][1] + mutation_confusion_matrix[0][0]) / (mutation_confusion_matrix[1][1] + mutation_confusion_matrix[1][0] + mutation_confusion_matrix[0][1] + mutation_confusion_matrix[0][0])
        accuracy_overall_list.append(accuracy)
    return accuracy_overall_list

def get_unnamed_binaries(col_list):
    binary = []
    for feature in col_list:
        if feature.startswith("C"):
            binary.append(1)
        else:
            binary.append(0)
    return binary

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

def gen_heatmap(metric_list):
    confusion_matrix = [[metric_list[0], metric_list[1]], 
                        [metric_list[2], metric_list[3]]]
    ax = sns.heatmap(confusion_matrix, annot=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.show()
# functions end

# convert excel to dataframe
excel_mutations = pd.read_excel('mutations.xlsx')
mutations_dataframe = pd.DataFrame(excel_mutations)
original_mutations_dataframe = pd.DataFrame(excel_mutations)

# gets binary of C and NC of main dataframe
actual = mutations_dataframe["Unnamed: 0"].to_list()
actual_binary = get_unnamed_binaries(actual)
# pops "unnamed: 0" column
mutations_dataframe.pop(mutations_dataframe.columns[0])

# calculates top 10 accuracy values
accuracy_overall_list = calc_accuracy_values(mutations_dataframe, actual_binary) 
n_largest(mutations_dataframe, "Mutation", "Accuracy", 10, accuracy_overall_list, "top_10_accuracy.xlsx") # converts the top 10 mutations of the overal dataframe

# nodes used in decision tree for classification of samples
node_Root = "BRAF_GRCh38_7:140753336-140753336_Missense-Mutation_SNP_A-A-T"
node_A = "BRAF_GRCh38_7:140753336-140753336_Missense-Mutation_SNP_A-A-T"
node_B = "KRAS_GRCh38_12:25245350-25245350_Missense-Mutation_SNP_C-C-A_C-C-G_C-C-T_C-G-G_C-A-A"

overall_list = original_mutations_dataframe["Unnamed: 0"] # this may change depending on what list of samples is taken in
evaluation_metrics = eval(overall_list, node_Root, node_A, node_B, original_mutations_dataframe) # call to get the TP, FP, TN, & FN for the sample list
metrics_calculate(evaluation_metrics) # calculation of metric passed in

gen_heatmap(evaluation_metrics)