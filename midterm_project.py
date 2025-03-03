import pandas as pd
import time
from itertools import combinations
from mlxtend.frequent_patterns import apriori, association_rules

# Welcome message
print("Welcome to the Frequent Itemset Mining and Association Rule Generation Program!")
selected_store = input("Please select your store:\n1. Amazon\n2. BestBuy\n3. Nike\n4. Kmart\n5. Walmart\n6. Quit program\n")
if selected_store == '6':
    quit()

stores = ['Amazon', 'BestBuy', 'Nike', 'Kmart', 'Walmart']

# Verify input
try:
    selected_store = int(selected_store)
    if selected_store < 1 or selected_store > len(stores):
        print("Invalid store selection. Please enter a valid number.")
        quit()
except ValueError:
    print("Invalid input. Please enter a valid number.")
    quit()

store_name = stores[selected_store - 1]

# Load datasets based on user-selected store
df_tr = pd.read_csv(f"csv files/{store_name}_transactions.csv")
df_items = pd.read_csv(f"csv files/{store_name}_items.csv")

print(f"\nYou have selected {store_name}!\n")
print("\nLoaded Items:")
print(df_items, "\n")

# Set a limit for max items per transaction
MAX_ITEMS_PER_TRANSACTION = 5  

# Prepare transactions with item limitation
transactions = df_tr['Transaction'].dropna().apply(lambda x: sorted(set(x.split(',')))[:MAX_ITEMS_PER_TRANSACTION]).tolist()

for i, transaction in enumerate(transactions, 1):
    print(f"Transaction {i}: {transaction}")

# User input for support and confidence thresholds
minimum_support = int(input("\nPlease enter a Minimum Support value (1 to 100): "))
minimum_confidence = int(input("Please enter a Minimum Confidence value (1 to 100): "))

minSupCount = (minimum_support / 100) * len(transactions)
minConfidence = minimum_confidence / 100

def brute_force_algorithm():
    print("\nInitiating Brute-Force Algorithm...\n")
    start_time = time.time()
    all_items = sorted(set(item for transaction in transactions for item in transaction))
    frequent_itemsets = {}

    def count_occurrences(itemset, transactions):
        return sum(1 for transaction in transactions if set(itemset).issubset(transaction))

    for item in all_items:
        support = count_occurrences([item], transactions)
        if support >= minSupCount:
            frequent_itemsets[(item,)] = support / len(transactions)

    k = 2
    while True:
        candidate_itemsets = list(combinations(frequent_itemsets.keys(), k))
        candidate_itemsets = [tuple(sorted(set().union(*itemset))) for itemset in candidate_itemsets]
        new_frequent_itemsets = {}
        for itemset in candidate_itemsets:
            support = count_occurrences(itemset, transactions)
            if support >= minSupCount:
                new_frequent_itemsets[itemset] = support / len(transactions)
        if not new_frequent_itemsets:
            break
        frequent_itemsets.update(new_frequent_itemsets)
        k += 1

    print("\nFrequent Itemsets (Brute-Force):")
    if not frequent_itemsets:
        print("No frequent itemsets found with the given support threshold.")
    else:
        for idx, (itemset, support) in enumerate(frequent_itemsets.items(), 1):
            print(f"Itemset {idx}: {list(itemset)}, Support: {support:.2f}")

    association_rules_list = []
    for itemset, support in frequent_itemsets.items():
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                consequent = tuple(set(itemset) - set(antecedent))
                antecedent_support = count_occurrences(antecedent, transactions) / len(transactions)
                confidence = support / antecedent_support if antecedent_support > 0 else 0
                if confidence >= minConfidence:
                    association_rules_list.append((antecedent, consequent, confidence, support))

    print("\nGenerated Association Rules (Brute-Force):")
    num_rules = len(association_rules_list)

    if not association_rules_list:
        print("No association rules found with the given confidence threshold.")
    else:
        for i, (antecedent, consequent, confidence, support) in enumerate(association_rules_list, 1):
            print(f"Rule {i}: {list(antecedent)} -> {list(consequent)}")
            print(f"Confidence: {confidence * 100:.2f}%")
            print(f"Support: {support * 100:.2f}%\n")

    execution_time = round(time.time() - start_time, 4)
    return num_rules, execution_time

def apriori_algorithm():
    print("\nInitiating Apriori Algorithm...")
    start_time = time.time()
    all_items = sorted(set(item for transaction in transactions for item in transaction))
    encoded_data = pd.DataFrame([{item: (item in transaction) for item in all_items} for transaction in transactions])

    frequent_itemsets = apriori(encoded_data, min_support=minimum_support / 100, use_colnames=True)

    if frequent_itemsets.empty:
        print("\nNo frequent itemsets found with the given support threshold.")
    else:
        print("\nFrequent Itemsets (Apriori):")
        for i, row in frequent_itemsets.iterrows():
            print(f"Itemset {i + 1}: {list(row['itemsets'])}, Support: {row['support']:.2f}")

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minConfidence)
    num_rules = len(rules)

    print("\nGenerated Association Rules (Apriori):")
    if rules.empty:
        print("No association rules found with the given confidence threshold.")
    else:
        for i, row in rules.iterrows():
            print(f"Rule {i + 1}: {list(row['antecedents'])} -> {list(row['consequents'])}")
            print(f"Confidence: {row['confidence'] * 100:.2f}%")
            print(f"Support: {row['support'] * 100:.2f}%\n")

    execution_time = round(time.time() - start_time, 4)
    return num_rules, execution_time

def run_algorithms():
    brute_force_rules, brute_force_time = brute_force_algorithm()
    apriori_rules, apriori_time = apriori_algorithm()

    print("\n" + "="*40)
    print("FINAL SUMMARY\n")
    print(f"Brute-Force Algorithm:")
    print(f"  - Total Rules Generated: {brute_force_rules}")
    print(f"  - Execution Time: {brute_force_time} seconds\n")
    
    print(f"Apriori Algorithm:")
    print(f"  - Total Rules Generated: {apriori_rules}")
    print(f"  - Execution Time: {apriori_time} seconds")
    print("="*40)

run_algorithms()