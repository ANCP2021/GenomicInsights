import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def samples_by_F(mutation_name, dataframe, sample_col):
    mutation_and_sample_df = pd.DataFrame(columns=["Sample", mutation_name])
    mutation_and_sample_df["Sample"] = dataframe[sample_col]
    mutation_and_sample_df[mutation_name] = dataframe[mutation_name]
    print(mutation_and_sample_df)

excel_mutations = pd.read_excel('mutations.xlsx')
mutations_dataframe = pd.DataFrame(excel_mutations)
original_mutations_dataframe = pd.DataFrame(excel_mutations)

actual = mutations_dataframe["Unnamed: 0"].to_list()
actual_binary = []
for feature in actual:
    if feature.startswith("C"):
        actual_binary.append(1)
    else:
        actual_binary.append(0)

mutations_dataframe.pop(mutations_dataframe.columns[0])

top_10_TP_FP = pd.DataFrame(columns=['Mutation', "TP-FP"])
diff_TP_FP_list = []

for col in mutations_dataframe:
    mutation_list = mutations_dataframe[col].to_list()
    mutation_confusion_matrix = pd.crosstab(actual_binary, mutation_list, rownames=['Actual'], colnames=['Predicted'])
    difference_TP_FP = mutation_confusion_matrix[1][1] - mutation_confusion_matrix[1][0]

    diff_TP_FP_list.append(difference_TP_FP)

top_10_TP_FP["Mutation"] = mutations_dataframe.columns
top_10_TP_FP["TP-FP"] = diff_TP_FP_list
top_10_TP_FP = top_10_TP_FP.nlargest(10, "TP-FP")
most_useful_feature = top_10_TP_FP.nlargest(1, "TP-FP")
top_10_TP_FP.to_excel("top_10_FP_TP.xlsx")

group_A = original_mutations_dataframe.loc[original_mutations_dataframe["BRAF_GRCh38_7:140753336-140753336_Missense-Mutation_SNP_A-A-T"] == 1, "Unnamed: 0"].values.T.tolist()
print("Group A:", group_A)
group_B = original_mutations_dataframe.loc[original_mutations_dataframe["BRAF_GRCh38_7:140753336-140753336_Missense-Mutation_SNP_A-A-T"] == 0, "Unnamed: 0"].values.T.tolist()
print("Group B:", group_B)

braf_mutation = mutations_dataframe["BRAF_GRCh38_7:140753336-140753336_Missense-Mutation_SNP_A-A-T"].to_list()
braf_confusionM = pd.crosstab(actual_binary, braf_mutation, rownames=['Actual'], colnames=['Predicted'])
# confusion matrix
sn.heatmap(braf_confusionM, annot=True, fmt='g')
plt.title("BRAF Mutation")
plt.show()