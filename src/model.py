from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self) -> None:
        self.model = []
        self.initialize()
    
    def initialize(self):
        self.model = RandomForestClassifier()
    
    def train(self,Dataset):
        self.model.fit(Dataset.train['x'],Dataset.train['y'])
        
    def predict_proba(self,Dataset):
        prediction = self.model.predict_proba(Dataset.test['x'])
        return prediction

    def predict(self,Dataset):
            prediction = self.model.predict(Dataset.test['x'])
            classes = self.model.classes_
            return [prediction, classes]