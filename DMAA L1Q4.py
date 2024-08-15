import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('transactions.csv')

transactions = df.groupby('TransactionID')['Item'].apply(list).tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

min_support = 0.4  

frequent_itemsets = fpgrowth(df_onehot, min_support=min_support, use_colnames=True)

print("Frequent Itemsets:")
print(frequent_itemsets)

min_confidence = 0.7  

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

print("\nAssociation Rules:")
print(rules)
