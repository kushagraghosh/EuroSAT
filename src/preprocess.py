from sklearn.preprocessing import StandardScaler 

def preprocess(Dataset):
    # Standardize features by removing mean and scaling to unit variance:
    Dataset.scaler = StandardScaler()
    Dataset.train['x'] = Dataset.scaler.fit_transform(Dataset.train['x'])
    Dataset.test['x'] = Dataset.scaler.transform(Dataset.test['x'])
    return Dataset