import pandas as pd
import matplotlib.pyplot as plt

excel_mutations = pd.read_excel('mutations.xlsx')
mutations_dataframe = pd.DataFrame(excel_mutations)

def createList(r1, r2):
    return [item for item in range(r1, r2 + 1)]

row_name_list = list(mutations_dataframe["Unnamed: 0"])
row_name_list_index = createList(0, (len(row_name_list) - 1))
mutation_per_individual = []
for row in mutations_dataframe.itertuples():
    mutation_per_individual.append(sum(row[2:]))

column_name_list = list(mutations_dataframe)
column_name_list.pop(0)
column_name_list_index = createList(0, (len(column_name_list) - 1)) 

samples_per_mutation = list(mutations_dataframe.sum(axis=0))
samples_per_mutation = samples_per_mutation[1:]

fig, ax = plt.subplots(2, figsize=(10, 6))
ax[0].scatter(x = row_name_list_index, y = mutation_per_individual)
ax[0].set_xlabel("Individual Samples")
ax[0].set_ylabel("Mutations per Sample")

ax[1].scatter(x = column_name_list_index, y = samples_per_mutation)
ax[1].set_xlabel("Mutations")
ax[1].set_ylabel("Samples per Mutation")

plt.show()