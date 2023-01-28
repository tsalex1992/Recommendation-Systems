import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms


class CB2CFMultiModalEncoder(nn.Module):
    def __init__(self, item_embedding_dim=64):
        super(CB2CFMultiModalEncoder, self).__init__()
        text_embedding_dim = 384
        self.text_encoding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.image_encoding_model = EfficientNet.from_pretrained("efficientnet-b0")
        # make the last layer of the image encoding model identity
        self.genre_layer = nn.Sequential(
            nn.Linear(text_embedding_dim, item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.actors_layer = nn.Sequential(
            nn.Linear(text_embedding_dim, item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.year_layer = nn.Sequential(
            nn.Linear(text_embedding_dim, item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.language_layer = nn.Sequential(
            nn.Linear(text_embedding_dim, item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.description_layer = nn.Sequential(
            nn.Linear(text_embedding_dim, item_embedding_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        self.image_layer = nn.Sequential(
            # Convert batch_size, 1280, 7, 7 to batch_size, 1280
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(6 * item_embedding_dim, item_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.freeze_parameters()

    def freeze_parameters(self):
        for param in self.image_encoding_model.parameters():
            param.requires_grad = False

        for param in self.text_encoding_model.parameters():
            param.requires_grad = False

    def forward(self, genres, actors, year, description, language, image):
        # TODO: make the prefixes parameters of the model
        movie_desc_string = "Movie description: " + description
        movie_desc_embedding = self.text_encoding_model.encode(movie_desc_string)
        genres_desc_string = "Movie genres: " + " ".join(genres)
        genres_desc_embedding = self.text_encoding_model.encode(genres_desc_string)
        actors_desc_string = "Movie actors: " + " ".join(actors)
        actors_desc_embedding = self.text_encoding_model.encode(actors_desc_string)
        year_embedding = "Movie year: " + str(year)
        year_embedding = self.year_layer(year_embedding)
        language_embedding = "Movie language: " + language
        language_embedding = self.language_layer(language_embedding)
        movie_desc_embedding = self.description_layer(movie_desc_embedding)
        genres_desc_embedding = self.genre_layer(genres_desc_embedding)
        actors_desc_embedding = self.actors_layer(actors_desc_embedding)
        normalized_image = self.transform_image(image)
        image_embedding = self.image_encoding_model(normalized_image)
        image_embedding = self.image_layer(image_embedding)
        # concatenate the embeddings
        movie_embedding = torch.cat(
            (
                movie_desc_embedding,
                genres_desc_embedding,
                actors_desc_embedding,
                year_embedding,
                language_embedding,
                image_embedding,
            ),
            dim=1,
        )
        movie_embedding = self.fc_layer(movie_embedding)
        return movie_embedding

    def transform_image(self, image):
        tfms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img = tfms(Image.open(image)).unsqueeze(0)
        return img
