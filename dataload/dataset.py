from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, train_stage=True):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.train_stage = train_stage

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence_origin = self._tokenize_function(self.df.loc[idx, '원문'], label=False)
        sentence_trans = self._tokenize_function(self.df.loc[idx, '번역문'], label=False)
        return sentence_origin, sentence_trans

    def _tokenize_function(self, text, label=True):
        tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=256)
        tensor_token = torch.tensor(tokenized.data['input_ids'])
        if label:
            tensor_token = tensor_token.float()
        return tensor_token


if __name__ == '__main__':
    import pandas as pd
    from utils.common.project_paths import GetPaths
    DF = pd.read_excel(GetPaths.get_data_folder('1_구어체(1).xlsx'))
    ds = CustomDataset(DF)