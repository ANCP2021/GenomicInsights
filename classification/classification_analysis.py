import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

excel_mutations = pd.read_excel('mutations.xlsx')
mutations_dataframe = pd.DataFrame(excel_mutations)

actual = mutations_dataframe["Unnamed: 0"].to_list()
actual_binary = []
for feature in actual:
    if feature.startswith("C"):
        actual_binary.append(1)
    else:
        actual_binary.append(0)

mutations_dataframe.pop(mutations_dataframe.columns[0])
max_difference = 0
genetic_mutation_max_val = ""
for col in mutations_dataframe:
    mutation_list = mutations_dataframe[col].to_list()
    mutation_confusion_matrix = pd.crosstab(actual_binary, mutation_list, rownames=['Actual'], colnames=['Predicted'])
    difference_TP_FP = mutation_confusion_matrix[1][1] - mutation_confusion_matrix[1][0]

    if difference_TP_FP > max_difference:
        max_difference = difference_TP_FP
        genetic_mutation_max_val = col

print(genetic_mutation_max_val + ":", max_difference)

print("**********************************************")

max_difference_percent = 0
genetic_mutation_max_val_percent = ""
# dataframe_sum = mutations_dataframe.values.sum()
for col in mutations_dataframe:
        mutation_list = mutations_dataframe[col].to_list()
        # mutation_list_sum = sum(mutation_list)
        mutation_confusion_matrix = pd.crosstab(actual_binary, mutation_list, rownames=['Actual'], colnames=['Predicted'])
        true_positive_percent = (mutation_confusion_matrix[1][1] / 230) * 100
        false_positive_percent = (mutation_confusion_matrix[1][0] / 230) * 100
        difference_TP_FP_percent = true_positive_percent - false_positive_percent

        if (difference_TP_FP_percent) > max_difference_percent:
            max_difference_percent = difference_TP_FP_percent
            genetic_mutation_max_val_percent = col

print(genetic_mutation_max_val_percent + ":", max_difference_percent)