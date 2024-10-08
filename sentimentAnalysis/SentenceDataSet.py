from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, phrases, sentiments=None):
        self.phrases = phrases
        self.sentiments = sentiments

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        if self.sentiments is not None:
            sentiment = self.sentiments[idx]
            return phrase, sentiment
        return phrase
