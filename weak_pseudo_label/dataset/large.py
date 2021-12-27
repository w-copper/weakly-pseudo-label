import torch.utils.data.dataset as dataset

class LargePatchDataset(dataset.Dataset):
    # pass
    def __init__(self, img_dir, label_dir) -> None:
        super().__init__()

    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)

    def generate_patch(self):
        pass
    
    @staticmethod
    def merge_patch(dataset:"LargePatchDataset"):
        pass