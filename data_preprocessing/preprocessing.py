import pandas as pd

def n_largest(dataframe, n, col):
    top_10_dataframe = dataframe.nlargest(n, [col])
    top_10_dataframe.to_excel("./output/" + col + ".xlsx")

excel_mutations = pd.read_excel('mutations.xlsx')
mutations_dataframe = pd.DataFrame(excel_mutations)

mutation_names = list(mutations_dataframe)
mutation_names.pop(0)

overall_dataframe = pd.DataFrame()
overall_dataframe["Mutation"] = mutation_names
overall_dataframe["T"] = list(mutations_dataframe.sum(axis=0))[1:]
overall_dataframe["C"] = list(mutations_dataframe.loc[mutations_dataframe["Unnamed: 0"].str.startswith("C")].sum(axis=0))[1:]
overall_dataframe["NC"] = list(mutations_dataframe.loc[mutations_dataframe["Unnamed: 0"].str.startswith("NC")].sum(axis=0))[1:]
overall_dataframe["%C"] = (overall_dataframe["C"] / overall_dataframe["C"].sum()) * 100
overall_dataframe["%NC"] = (overall_dataframe["NC"] / overall_dataframe["NC"].sum()) * 100
overall_dataframe["%C subtraction %NC"] = (overall_dataframe["%C"] - overall_dataframe["%NC"])
overall_dataframe["%C division %NC"] = (overall_dataframe["%C"] / overall_dataframe["%NC"])

n_largest(overall_dataframe, 10, "T")
n_largest(overall_dataframe, 10, "C")
n_largest(overall_dataframe, 10, "NC")
n_largest(overall_dataframe, 10, "%C")
n_largest(overall_dataframe, 10, "%NC")
n_largest(overall_dataframe, 10, "%C subtraction %NC")
n_largest(overall_dataframe, 10, "%C division %NC")