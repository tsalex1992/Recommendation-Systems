import torch
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


def load_item_embeddings(n_items, n_item_state):
    # Load the numpy matrix from the pickle file
    item_embeddings = pickle.load(open("item_embeddings.pkl", "rb"))
    # Convert the numpy matrix to a tensor
    item_embeddings = torch.from_numpy(item_embeddings)
    # Add the embeddings for SOS and EOS tokens
    # SOS will be all zeros and EOS will be all ones
    item_embeddings = torch.cat(
        [
            torch.zeros(1, n_item_state),
            torch.ones(1, n_item_state),
            item_embeddings,
        ],
        dim=0,
    )
    return item_embeddings


def convert_input_id_to_movie_id(input_id):
    # If input_id is equal to n_items, then it is the SOS token
    if input_id == 0:
        return "SOS"
    # If input_id is equal to n_items + 1, then it is the EOS token
    elif input_id == 1:
        return "EOS"
    # If input_id is greater than n_items + 1, then it is a movie id
    elif input_id > 1:
        return input_id - 2


def convert_movie_id_to_input_id(movie_id):
    return movie_id + 2


def get_movie_title_from_id(movies_df, movie_id):
    # Go to the index of the movie id and get the title
    return movies_df.iloc[movie_id]["title"]


@dataclass
class BaseTrainingDataForNextItemPred:
    user_ids: torch.Tensor
    items_ids: torch.Tensor
    times: torch.Tensor


@dataclass
class TrainingDataForNextItemPredInstance:
    user_id: torch.Tensor
    items_ids: torch.Tensor
    times: torch.Tensor
    pred_index: int
    true_item_id: torch.Tensor


# Gets user ratings vectors and outputs all the subsequent items for each user
def prepare_base_training_data_for_next_item_pred_transformer(
    user_ids: List,
    user_items_vectors: List[np.ndarray],
    user_rating_times_vectors: List[np.ndarray],
    max_seq_len: int,
) -> BaseTrainingDataForNextItemPred:
    """Converts user ratings vectors into a matrix of items for the transformer model.

    Args:
        user_ids (list): List of user ids
        user_items_vectors (list): List of user ratings vectors as numpy arrays
        user_rating_times_vectors (list): List of user rating times vectors as numpy arrays
        max_seq_len (int): The maximum sequence length for the transformer model

    Returns:
        items (torch.Tensor): A torch matrix of size (n_users, max_item_seq_len + 2)
        user_ids (torch.Tensor): A torch vector of size (n_users)
    """
    # Create a torch matrix of size (n_users, max_item_seq_len + 2)
    # The +2 is for the SOS and EOS tokens

    assert len(user_items_vectors) == len(user_rating_times_vectors)

    items = torch.ones((len(user_items_vectors), max_seq_len + 2), dtype=torch.long)

    times = torch.zeros((len(user_rating_times_vectors), max_seq_len + 2), dtype=torch.long)
    # Convert the first column to the SOS token
    items[:, 0] = 0
    # Convert the last column to the EOS token
    items[:, -1] = 1

    # Iterate over each user and add their items to the matrix
    for i, user_items in enumerate(user_items_vectors):
        # Add the user items to the matrix
        items[i, 1 : len(user_items) + 1] = torch.tensor(user_items)

        # Add the relevant times to the matrix
        times[i, 1 : len(user_rating_times_vectors[i]) + 1] = torch.tensor(
            user_rating_times_vectors[i]
        )

    # Convert user ids into a tensor
    user_ids = torch.tensor(user_ids)
    return BaseTrainingDataForNextItemPred(user_ids, items, times)


# Get BaseTrainingDataForNextItemPred and outputs all seb sequences for each user
def prepare_training_data_for_next_item_pred_transformer(
    user_ids: List,
    user_items_vectors: List[np.ndarray],
    user_rating_times_vectors: List[np.ndarray],
    max_seq_len: int,
) -> List[TrainingDataForNextItemPredInstance]:

    base_training_data = prepare_base_training_data_for_next_item_pred_transformer(
        user_ids, user_items_vectors, user_rating_times_vectors, max_seq_len
    )
    eos_token = 1
    eos_indices = [len(user_items) for user_items in user_items_vectors]
    res = []

    # For each user get all the subsequences until the EOS token
    # Fill each sequence with the EOS token until the max_seq_len

    for i, user_items in enumerate(base_training_data.items_ids):
        # Get the index of the EOS token
        eos_index = eos_indices[i]
        user_id = base_training_data.user_ids[i]
        for j in range(0, eos_index - 1):
            # Get the subsequence
            subsequence = user_items[0 : j + 1]
            # Fill the rest of the sequence with the EOS token
            subsequence = torch.cat(
                [
                    subsequence,
                    torch.ones(max_seq_len + 2 - len(subsequence), dtype=torch.long),
                ]
            )
            # Get the subsequence times
            subsequence_times = base_training_data.times[i][0 : j + 1]
            # Fill the rest of the sequence with the EOS token
            subsequence_times = torch.cat(
                [
                    subsequence_times,
                    torch.zeros(max_seq_len + 2 - len(subsequence_times), dtype=torch.long),
                ]
            )
            # Create a TrainingDataForNextItemPredInstance
            training_data = TrainingDataForNextItemPredInstance(
                user_id, subsequence, subsequence_times, j, user_items[j + 1]
            )
            # Yield the TrainingDataForNextItemPredInstance
            res.append(training_data)
    return res
