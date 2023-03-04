from torch.utils.data import IterableDataset


# class NextItemPredDataset(Dataset):
#     def __init__(
#         self,
#         user_ids: List,
#         user_items_vectors: List[np.ndarray],
#         user_rating_times_vectors: List[np.ndarray],
#         max_seq_len: int,
#     ):
#         self.user_ids = user_ids
#         self.user_items_vectors = user_items_vectors
#         self.user_rating_times_vectors = user_rating_times_vectors
#         self.max_seq_len = max_seq_len
#         self.eos_indices = [len(user_items) for user_items in user_items_vectors]

#     def __len__(self):
#         return (sum(self.eos_indices) - len(self.eos_indices)) * len(self.user_items_vectors)


#     def __getitem__(self, index):
#         item = prepare_training_data_for_next_item_pred_transformer(
#             self.user_ids, self.user_items_vectors, self.user_rating_times_vectors, self.max_seq_len
#         )
# return (
#     item.user_id,
#     item.items_ids,
#     # Convert times to float
#     item.times.float(),
#     item.pred_index,
# )
# Convert the above to iterator dataset
class NextItemPredDataset(IterableDataset):
    def __init__(self, my_iterator):
        super(NextItemPredDataset).__init__()
        self.iterator = my_iterator

    def __iter__(self):
        return self.iterator
