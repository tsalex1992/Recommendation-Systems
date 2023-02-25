from torch.utils.data import Dataset
from typing import List
import numpy as np
from utils import prepare_training_data_for_next_item_pred_transformer


class NextItemPredDataset(Dataset):
    def __init__(
        self,
        user_ids: List,
        user_items_vectors: List[np.ndarray],
        user_rating_times_vectors: List[np.ndarray],
        max_seq_len: int,
    ):
        self.data = prepare_training_data_for_next_item_pred_transformer(
            user_ids, user_items_vectors, user_rating_times_vectors, max_seq_len
        )
        # Order the data by pred_index
        self.data = sorted(self.data, key=lambda x: x.pred_index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.data[index].user_id,
            self.data[index].items_ids,
            # Convert times to float
            self.data[index].times.float(),
            self.data[index].pred_index,
        )
