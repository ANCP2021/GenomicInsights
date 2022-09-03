from operator import contains
import sys
import pandas as pd

excel_mutations = pd.read_excel('mutations.xlsx')
mutations_dataframe = pd.DataFrame(excel_mutations)

def count_unique_mutations(dataframe):
    unique_mutations = dataframe.shape[1] - 1
    print("Unique Mutations:", unique_mutations)

def count_individual_samples(dataframe):
    individual_samples = dataframe.shape[0]
    print("Individual Samples:", individual_samples)

def mutation_count(dataframe, individual):
    for row in dataframe.itertuples():
        if row[1] == individual:
            row = list(row)
            mutation_count = row[2:]

    print(individual, "Mutations:", sum(mutation_count))

def avg_mutations_per_individual(dataframe):
    sum_mutations = 0
    for row in dataframe.iterrows():
        list_row = list(row[1][1:])
        sum_mutations += sum(list_row)

    avg_mutations_per_indv = sum_mutations / int((dataframe.shape[0]))

    print("Avg Mutations per Individual", round(avg_mutations_per_indv, 2))

def min_max_mutation_per_individual(dataframe):
    max_value = 0
    min_value = sys.maxsize
    for row in dataframe.iterrows():
        list_row = list(row[1][1:])
        sum_list = sum(list_row)
        if sum_list < min_value:
            min_value = sum_list
        if sum_list > max_value:
            max_value = sum_list
        
    print("Maximum Mutations per Individual:", max_value)
    print("Minimum Mutations per Individual:", min_value)

def individuals_w_mutation_gene(dataframe, substring):
    number_individuals = dataframe.filter(like=substring).sum()
    print("Individuals with Mutation in", substring, "Gene:", number_individuals.sum())

def avg_individuals_per_mutation(dataframe):
    sum_mutations = 0
    for row in dataframe.iterrows():
        list_row = list(row[1][1:])
        sum_mutations += sum(list_row)

    avg_mutations_per_indv = sum_mutations / int((dataframe.shape[1] - 1))

    print("Avg Individuals per Mutation", round(avg_mutations_per_indv, 2))

def min_max_individual_per_mutation(dataframe):
    sum_df = list(dataframe.sum(axis=0))[1:]
    max_value = max(sum_df)
    min_value = min(sum_df)

    print("Maximum Individuals per Mutation:", max_value)
    print("Minimum Individuals per Mutation:", min_value)
    



count_unique_mutations(mutations_dataframe)
count_individual_samples(mutations_dataframe)
mutation_count(mutations_dataframe, "C1")
mutation_count(mutations_dataframe, "NC1")
avg_mutations_per_individual(mutations_dataframe)
min_max_mutation_per_individual(mutations_dataframe)
individuals_w_mutation_gene(mutations_dataframe, "BRAF")
individuals_w_mutation_gene(mutations_dataframe, "KRAS")
avg_individuals_per_mutation(mutations_dataframe)
min_max_individual_per_mutation(mutations_dataframe)