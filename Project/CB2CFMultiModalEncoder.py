import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class CB2CFMultiModalEncoder(nn.Module):
    def __init__(
        self,
        number_of_genres,
        number_of_actors,
        number_of_directors,
        number_of_languages,
        item_embedding_dim=64,
        dropout=0.2,
    ):
        super(CB2CFMultiModalEncoder, self).__init__()
        text_embedding_dim = 384
        self.text_encoding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.text_encoding_model.max_seq_length = 400
        self.genres_embedding = nn.Sequential(
            nn.Linear(number_of_genres, item_embedding_dim),
            nn.BatchNorm1d(item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.actors_embedding = nn.Sequential(
            nn.Linear(number_of_actors, item_embedding_dim),
            nn.BatchNorm1d(item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.directors_embedding = nn.Sequential(
            nn.Linear(number_of_directors, item_embedding_dim),
            nn.BatchNorm1d(item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.time_layer = nn.Sequential(
            nn.Linear(1, item_embedding_dim),
            nn.BatchNorm1d(item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.language_embedding = nn.Sequential(
            nn.Linear(number_of_languages, item_embedding_dim),
            nn.BatchNorm1d(item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.description_layer = nn.Sequential(
            nn.Linear(text_embedding_dim, text_embedding_dim),
            nn.BatchNorm1d(text_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.combiner = nn.Sequential(
            nn.LayerNorm(5 * item_embedding_dim + text_embedding_dim),
            nn.Linear(5 * item_embedding_dim + text_embedding_dim, 3 * item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(3 * item_embedding_dim, item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(item_embedding_dim, item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.freeze_parameters()

    def freeze_parameters(self):
        for param in self.text_encoding_model.parameters():
            param.requires_grad = False

    def forward(self, genres, actors, directors, unix_release_time, description, language):
        genres = self.genres_embedding(genres)
        actors = self.actors_embedding(actors)
        directors = self.directors_embedding(directors)
        unix_release_time = self.time_layer(unix_release_time)
        language = self.language_embedding(language)
        description = self.text_encoding_model.encode(description, convert_to_tensor=True)
        description = self.description_layer(description)
        movie_embedding = torch.cat(
            (genres, description, actors, directors, unix_release_time, language), dim=1
        )
        movie_embedding = self.combiner(movie_embedding)
        return movie_embedding
