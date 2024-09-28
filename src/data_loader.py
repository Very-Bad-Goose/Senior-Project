#change this line to match our formatting if needed
'''features = torch.tensor([sample['Data1']... etc'''

''' Add to trainer script
from data_loader import get_data_loader

train_loader = get_data_loader('train.json', batch_size=32)
test_loader = get_data_loader('test.json', batch_size=32)
'''

import torch
from torch.utils.data import Dataset, DataLoader
import json

class JSONDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as file:
            self.data = json.load(file)  # Load the JSON file into a list of dictionaries
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract the features and label from the dictionary
        sample = self.data[idx]
        features = torch.tensor([sample['Data1'], sample['Data2'], sample['Data3']], dtype=torch.float32)
        label = torch.tensor(sample['dataLabel'], dtype=torch.long)  # Adjust the key as per your JSON format

        sample = {'features': features, 'label': label}

        # Apply any transformations if specified
        if self.transform:
            sample = self.transform(sample)

        return sample


# Create the DataLoader
def get_data_loader(json_file, batch_size=32, shuffle=True, num_workers=0):
    dataset = JSONDataset(json_file=json_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# Test the DataLoader
if __name__ == '__main__':
    json_path = 'path\to\file.json'  # Replace with our json file path
    dataloader = get_data_loader(json_path, batch_size=4)

    # Iterate through the DataLoader
    for i, batch in enumerate(dataloader):
        print(f'Batch {i}:')
        print('Features:', batch['features'])
        print('Labels:', batch['label'])
