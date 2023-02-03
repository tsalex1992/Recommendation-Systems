import pandas as pd

df = pd.read_csv("data/movies_metadata.csv")
credits_df = pd.read_csv("data/credits.csv")

# drop unnecessary columns
df = df.drop(
    columns=[
        "adult",
        "belongs_to_collection",
        "homepage",
        "poster_path",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "status",
        "tagline",
        "video",
        "budget",
        "revenue",
        "vote_average",
        "popularity",
    ]
)
# Remove rows which have NaN values
df = df.dropna()
credits_df = credits_df.dropna()
# print row where id is 1997-08-20
# change the id data type to int
df["id"] = df["id"].astype(int)
# merge on the id column
df = df.merge(credits_df, on="id")

# print how many items we have
print("Number of items: ", len(df))
