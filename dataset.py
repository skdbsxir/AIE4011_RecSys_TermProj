"""
interaction dataset인 big_matrix.csv & small_matrix.csv를 변환
"""
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle

class BasicDataset(Dataset):
    def __init__(self, user_list, item_list, interaction_list):
        super(BasicDataset, self).__init__()

        self.user_list = user_list
        self.item_list = item_list
        self.interaction_list = interaction_list

    def __len__(self):
        return len(self.user_list)
    
    def __getitem__(self, index):
        user = self.user_list[index]
        item = self.item_list[index]
        interaction = self.interaction_list[index]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(interaction, dtype=torch.float)
        )

class MyDataset(object):
    def __init__(self, args, dataframe, type):
        """
        big/small .csv를 읽어서 -> dataset 변환 및 dataloader 구성
        type: big_matrix.csv, small_matrix.csv에서 _를 기준으로 split 했을 때, 'big' 또는 'small'

        사용할 column:
            user_id, video_id, watch_ratio/play_duration (==rating)
        """
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.type = type

        if self.type == 'big':
            dataframe = shuffle(dataframe, random_state=self.seed)
            num_valid = int(0.1 * len(dataframe))
            self.valid_data = dataframe[:num_valid]
            self.train_data = dataframe[num_valid:]

        elif self.type == 'small':
            self.test_data = dataframe

        else:
            raise Exception("Error")

    def load_train_data(self):
        users, items, interactions = [], [], []
        for row in self.train_data.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.video_id))
            interactions.append(float(row.watch_ratio))
        
        dataset = BasicDataset(users, items, interactions)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def load_valid_data(self):
        users, items, interactions = [], [], []
        for row in self.valid_data.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.video_id))
            interactions.append(float(row.watch_ratio))
        
        dataset = BasicDataset(users, items, interactions)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def load_test_data(self):
        users, items, interactions = [], [], []
        for row in self.test_data.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.video_id))
            interactions.append(float(row.watch_ratio))
        
        dataset = BasicDataset(users, items, interactions)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)   
