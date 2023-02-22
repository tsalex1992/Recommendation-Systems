from dataclasses import dataclass
from torch.utils.data import Dataset
import torch
import numpy as np


class CB2CFMultiModalEncoderDataset(Dataset):
    def __init__(
        self,
        genres: np.ndarray,
        actors: np.ndarray,
        directors: np.ndarray,
        unix_release_time: np.ndarray,
        description: np.ndarray,
        language: np.ndarray,
        movie_ids: np.ndarray,
        embedding: np.ndarray,
    ):
        # Convert the inputs to pytorch tensors
        self.genres = torch.from_numpy(genres).float()
        self.actors = torch.from_numpy(actors).float()
        self.directors = torch.from_numpy(directors).float()
        self.unix_release_time = torch.from_numpy(unix_release_time).float().unsqueeze(1)
        self.description = description
        self.language = torch.from_numpy(language).float()
        self.movie_ids = torch.from_numpy(movie_ids)
        self.embeddings = torch.from_numpy(embedding).float()

    def __len__(self):
        return len(self.movie_ids)

    def __getitem__(self, index):
        return {
            "genres": self.genres[index],
            "actors": self.actors[index],
            "directors": self.directors[index],
            "unix_release_time": self.unix_release_time[index],
            "description": self.description[index],
            "language": self.language[index],
            "movie_ids": self.movie_ids[index],
            "embeddings": self.embeddings[index],
        }
