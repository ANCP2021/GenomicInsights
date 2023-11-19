import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import time
start_time = time.time()


def get_bootstrap(dataframe):
    bootstrap = pd.DataFrame(columns=list(dataframe))
    for i in range(0, 230):
        index = random.randint(0,229)
        bootstrap = pd.concat([bootstrap, dataframe.iloc[[index]]])
    
    overall_samples = dataframe['Unnamed: 0'].tolist()
    bootstrap_samples = bootstrap['Unnamed: 0'].tolist()
    out_of_bag_samples = [x for x in overall_samples if x not in bootstrap_samples]
    print("Length of Out of Bag:", len(out_of_bag_samples))
    print("Out of Bag List:", out_of_bag_samples)

    return bootstrap 

excel_mutations = pd.read_excel('mutations.xlsx')
mutations_dataframe = pd.DataFrame(excel_mutations)
original_mutations_dataframe = pd.DataFrame(excel_mutations)

bootstrap_df = get_bootstrap(mutations_dataframe)

print("--- %s seconds ---" % (time.time() - start_time))