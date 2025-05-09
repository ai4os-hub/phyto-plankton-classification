import os
import random

def split_train_file(input_file, num_clients, split):
    # Read the content of the train.txt file
    with open(input_file, 'r') as f:
        lines = f.read().splitlines()
    random.shuffle(lines)
    
    # Calculate how many lines each client should get
    lines_per_client = len(lines) // num_clients
    remainder = len(lines) % num_clients
    
    # Split the data and write to separate files
    start_idx = 0
    for i in range(num_clients):
        # Add an extra line to some clients if there's a remainder
        extra = 1 if i < remainder else 0
        end_idx = start_idx + lines_per_client + extra
        
        # Get the current client's data
        client_data = lines[start_idx:end_idx]
        
        output_dir = os.path.dirname(input_file)

        
        # Write to a new file
        with open(f'{output_dir}/site-{i+1}_{split}.txt', 'w') as f:
            f.write('\n'.join(client_data))
       
        start_idx = end_idx
    
    print(f"Successfully split data among {num_clients} clients.")

# Example usage
input_file = '/hkfs/home/project/hk-project-test-p0023500/mp9809/plankton/plankton/app/custom/phyto-plankton-classification/data/dataset_files/val.txt'
num_clients = 2  # Change this to your desired number of clients
split = "train"
split_train_file(input_file, num_clients, split) 