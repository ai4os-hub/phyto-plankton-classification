import pandas as pd

# Load the CSV file
df = pd.read_csv('columns.csv')


# Load the CSV file
new_order= pd.read_csv('new_order.csv')

# Function to normalize names: spaces to underscores, plus to slashes, hyphen to underscore
# Function to normalize names: spaces to underscores, plus to slashes, hyphen to underscore, colon to hyphen
# Function to normalize names: spaces to underscores, plus to slashes, hyphen to underscore, colon to hyphen
def normalize_name(name):
    name = name.replace(' ', '_')  # Replace spaces with underscores
    name = name.replace('+', '/')  # Replace plus signs with slashes
    name = name.replace('-', '_')  # Replace hyphen with underscore
    name = name.replace(':', '_')  # Replace colon with hyphen
    return name

# Apply the normalization to both 'names' columns (df and new_order)
df['names_normalized'] = df['names'].apply(normalize_name)
new_order['new_order_normalized'] = new_order['new_order'].apply(normalize_name)

# Check if the normalized 'names' in df match any in new_order
df['match_in_new_order'] = df['names_normalized'].isin(new_order['new_order_normalized'])
# df[df["names"]=="Crustaceae:part"]["names"]="Crustaceae-part"
# Filter rows where names don't match
df

# First, create a mapping from normalized name to original row in df
df_indexed = df.set_index('names_normalized')

# Then, use new_order to reindex df
reordered_df = df_indexed.loc[new_order['new_order_normalized']].reset_index()

# Optional: drop the normalized columns if you don’t need them anymore
# reordered_df = reordered_df.drop(columns=['names_normalized', 'match_in_new_order'])

print(reordered_df)

reordered_df.to_csv('reordered_columns.csv', index=False)
