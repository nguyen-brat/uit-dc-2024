from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from qwen_vl_utils import process_vision_info


class MSDDataloader(Dataset):
    def __init__(self, path):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        '''
        Return message and a labels
        '''
        return
    
    def load_dataset(self):
        pass