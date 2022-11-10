import pandas as pd

def n_largest(dataframe, mutation, calc, n, TP_FP_list, excel_name):
    top_TP_FP = pd.DataFrame(columns=[mutation, calc])
    top_TP_FP[mutation] = dataframe.columns
    top_TP_FP[calc] = TP_FP_list
    top_TP_FP = top_TP_FP.nlargest(n, calc)
    top_TP_FP.to_excel(excel_name)
    return top_TP_FP

def calc_TPFP_values(dataframe, binary):
    diff_TP_FP_list = []
    for col in dataframe:
        mutation_list = dataframe[col].to_list()
        mutation_confusion_matrix = pd.crosstab(binary, mutation_list, rownames=['Actual'], colnames=['Predicted'], dropna=False)
        mutation_confusion_matrix = mutation_confusion_matrix.reindex(columns=[0,1], fill_value=0)
        difference_TP_FP = mutation_confusion_matrix[1][1] - mutation_confusion_matrix[1][0]
        diff_TP_FP_list.append(difference_TP_FP)
    return diff_TP_FP_list

def get_unnamed_binaries(col_list):
    binary = []
    for feature in col_list:
        if feature.startswith("C"):
            binary.append(1)
        else:
            binary.append(0)
    return binary

def classify(sample, overall_top_feature, groupA_top_feature, groupB_top_feature, overall):
    row_index = overall.index[overall["Unnamed: 0"] == sample]
    if overall.iat[row_index[0], overall.columns.get_loc(overall_top_feature)] == 1:
        if overall.iat[row_index[0], overall.columns.get_loc(groupA_top_feature)] == 1:
            print(sample, "Cancer")
        else:
            print(sample, "Non-Cancer")
    else:
        if overall.iat[row_index[0], overall.columns.get_loc(groupB_top_feature)] == 1:
            print(sample, "Cancer")
        else:
            print(sample, "Non-Cancer")

excel_mutations = pd.read_excel('mutations.xlsx')
mutations_dataframe = pd.DataFrame(excel_mutations)
original_mutations_dataframe = pd.DataFrame(excel_mutations)

# print(original_mutations_dataframe.index[original_mutations_dataframe["Unnamed: 0"] == "C1"].tolist())

# gets binary of C and NC of main dataframe
actual = mutations_dataframe["Unnamed: 0"].to_list()
actual_binary = get_unnamed_binaries(actual)
# pops "unnamed: 0" column
mutations_dataframe.pop(mutations_dataframe.columns[0])

diff_TP_FP_overall_list = calc_TPFP_values(mutations_dataframe, actual_binary) # calculates all TP-FP values
top_TP_FP_overall = n_largest(mutations_dataframe, "Mutation", "TP-FP", 10, diff_TP_FP_overall_list, "top_10_TP_FP.xlsx") # converts the top 10 mutations of the overal dataframe

# initializes groups A and B depending on if they have the top feature "BRAF" or not
group_A = original_mutations_dataframe.loc[original_mutations_dataframe[top_TP_FP_overall["Mutation"].iloc[0]] == 1, "Unnamed: 0"].values.T.tolist()
group_B = original_mutations_dataframe.loc[original_mutations_dataframe[top_TP_FP_overall["Mutation"].iloc[0]] == 0, "Unnamed: 0"].values.T.tolist()


group_A_binary = get_unnamed_binaries(group_A)

groupA_df = original_mutations_dataframe[original_mutations_dataframe["Unnamed: 0"].isin(group_A)]
groupA_df.pop(groupA_df.columns[0])
diff_TP_FP_groupA_list = calc_TPFP_values(groupA_df, group_A_binary)
top_TP_FP_groupA = n_largest(groupA_df, "Mutation", "TP-FP", 10, diff_TP_FP_groupA_list, "top_10_TP_FP_GroupA.xlsx")
# confusion matrix groupA
top_mutation_groupA = groupA_df[top_TP_FP_groupA["Mutation"].iloc[0]].tolist()
confusion_matrix_top_feature_A = pd.crosstab(group_A_binary, top_mutation_groupA, rownames=['Actual'], colnames=['Predicted'])
print("GroupA Confusion Matrix:")
print(confusion_matrix_top_feature_A)


group_B_binary = get_unnamed_binaries(group_B)
groupB_df = original_mutations_dataframe[original_mutations_dataframe["Unnamed: 0"].isin(group_B)]
groupB_df.pop(groupB_df.columns[0])
diff_TP_FP_groupB_list = calc_TPFP_values(groupB_df, group_B_binary)
top_TP_FP_groupB = n_largest(groupB_df, "Mutation", "TP-FP", 10, diff_TP_FP_groupB_list, "top_10_TP_FP_GroupB.xlsx")
# confusion matrix groupA
top_mutation_groupB = groupB_df[top_TP_FP_groupB["Mutation"].iloc[0]].tolist()
confusion_matrix_top_feature_B = pd.crosstab(group_B_binary, top_mutation_groupB, rownames=['Actual'], colnames=['Predicted'])
print("GroupB Confusion Matrix:")
print(confusion_matrix_top_feature_B)


classify("C1", top_TP_FP_overall["Mutation"].iloc[0], top_TP_FP_groupA["Mutation"].iloc[0], top_TP_FP_groupB["Mutation"].iloc[0], original_mutations_dataframe)
classify("C10", top_TP_FP_overall["Mutation"].iloc[0], top_TP_FP_groupA["Mutation"].iloc[0], top_TP_FP_groupB["Mutation"].iloc[0], original_mutations_dataframe)
classify("C50", top_TP_FP_overall["Mutation"].iloc[0], top_TP_FP_groupA["Mutation"].iloc[0], top_TP_FP_groupB["Mutation"].iloc[0], original_mutations_dataframe)
classify("NC5", top_TP_FP_overall["Mutation"].iloc[0], top_TP_FP_groupA["Mutation"].iloc[0], top_TP_FP_groupB["Mutation"].iloc[0], original_mutations_dataframe)
classify("NC15", top_TP_FP_overall["Mutation"].iloc[0], top_TP_FP_groupA["Mutation"].iloc[0], top_TP_FP_groupB["Mutation"].iloc[0], original_mutations_dataframe)
