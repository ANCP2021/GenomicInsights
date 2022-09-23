import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

excel_mutations = pd.read_excel('mutations.xlsx')
mutations_dataframe = pd.DataFrame(excel_mutations)

rnfa43_mutation = mutations_dataframe["RNF43_GRCh38_17:58357800-58357800_Frame-Shift-Del_DEL_C-C--"].to_list()
tp53_mutation = mutations_dataframe["TP53_GRCh38_17:7675088-7675088_Missense-Mutation_SNP_C-T-T_C-C-T"].to_list()

actual = mutations_dataframe["Unnamed: 0"].to_list()
actual_binary = []
for feature in actual:
    if feature.startswith("C"):
        actual_binary.append(1)
    else:
        actual_binary.append(0)
        
# Actual    Prediction
# 1         1          = True Positive
# 1         0          = False Negative
# 0         1          = False Positive
# 0         0          = True Negative
rnfa43_confusionM = pd.crosstab(actual_binary, rnfa43_mutation, rownames=['Actual'], colnames=['Predicted'])
tp53_confusionM = pd.crosstab(actual_binary, tp53_mutation, rownames=['Actual'], colnames=['Predicted'])
print(rnfa43_confusionM)
print(tp53_confusionM)
# confusion matrix
sn.heatmap(rnfa43_confusionM, annot=True, fmt='g')
plt.title("RNFA43 Mutation")
plt.show()
sn.heatmap(tp53_confusionM, annot=True, fmt='g')
plt.title("TP53 Mutation")
plt.show()

# stacked bar charts
figure, axis = plt.subplots(1, 2)
axis[0].bar(["RNFA43", "TP53"], [rnfa43_confusionM[1][1], tp53_confusionM[1][1]], label="True Positive")
axis[0].bar(["RNFA43", "TP53"], [rnfa43_confusionM[1][0], tp53_confusionM[1][0]], label="False Positive", bottom=[rnfa43_confusionM[1][1], tp53_confusionM[1][1]])
axis[0].set_title("TP & FP")
axis[0].legend()
axis[1].bar(["RNFA43", "TP53"], [rnfa43_confusionM[0][0], tp53_confusionM[0][0]], label="True Negative")
axis[1].bar(["RNFA43", "TP53"], [rnfa43_confusionM[0][1], tp53_confusionM[0][1]], label="False Negative", bottom=[rnfa43_confusionM[0][0], tp53_confusionM[0][0]])
axis[1].set_title("TN & FN")
axis[1].legend()
plt.show()

rnfa43_1D = [rnfa43_confusionM[0][0], rnfa43_confusionM[0][1], rnfa43_confusionM[1][0], rnfa43_confusionM[1][1]]
tp53_1D = [tp53_confusionM[0][0], tp53_confusionM[0][1], tp53_confusionM[1][0], tp53_confusionM[1][1]]

plt.pie(rnfa43_1D, labels=['TN', 'FN', 'FP', 'TP'], autopct='%1.1f%%', pctdistance=0.85)
circle = plt.Circle((0, 0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(circle)
plt.title('RNFA43 Mutation')
plt.show()

plt.pie(tp53_1D, labels=['TN', 'FN', 'FP', 'TP'], autopct='%1.1f%%', pctdistance=0.85)
circle = plt.Circle((0, 0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(circle)
plt.title('TP53 Mutation')
plt.show()