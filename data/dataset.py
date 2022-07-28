from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(self, data, label, windows_size, step_size):
        self.data = data
        self.label = label
        self.windows_size = windows_size
        self.step_size = step_size

    def __getitem__(self, index):
        index = index * self.step_size
        X = self.data[index: index + self.windows_size, :]
        y = self.label[index: index + self.windows_size]
        return X, y

    def __len__(self):
        return (len(self.data) - self.windows_size) // self.step_size + 1
