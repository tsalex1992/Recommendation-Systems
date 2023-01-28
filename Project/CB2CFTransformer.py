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
    ):
        super(CB2CFMultiModalEncoder, self).__init__()
        text_embedding_dim = 384
        self.text_encoding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.genres_embedding = nn.Embedding(number_of_genres, item_embedding_dim)
        self.actors_embedding = nn.Embedding(number_of_actors, item_embedding_dim)
        self.directors_embedding = nn.Embedding(number_of_directors, item_embedding_dim)
        self.year_layer = nn.Sequential(
            nn.Linear(1, item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.language_embedding = nn.Embedding(number_of_languages, item_embedding_dim)
        self.description_layer = nn.Sequential(
            nn.Linear(text_embedding_dim, item_embedding_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(7 * item_embedding_dim, item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.freeze_parameters()

    def freeze_parameters(self):
        for param in self.text_encoding_model.parameters():
            param.requires_grad = False

    def forward(self, genres, actors, directors, year, description, language):
        genres = self.genres_embedding(genres)
        actors = self.actors_embedding(actors)
        directors = self.directors_embedding(directors)
        year = self.year_layer(year)
        language = self.language_embedding(language)
        description = self.text_encoding_model.encode(description)
        description = self.description_layer(description)
        movie_embedding = torch.cat((genres, actors, directors, year, description, language), dim=1)
        movie_embedding = self.fc_layer(movie_embedding)
        return movie_embedding
