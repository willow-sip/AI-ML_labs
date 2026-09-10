import pandas as pd
import numpy as np

np.random.seed(42)

# import kagglehub
# path = kagglehub.dataset_download("prasertk/spotify-global-2019-moststreamed-tracks")
# print("Path to dataset files:", path)

df = pd.read_csv("spotify_global_2019_most_streamed_tracks_audio_features.csv") # point 1

print("First 5 records:\n", df.head()) # point 2

#=================================================================

print("Skipped values for each column (before):\n", df.isnull().sum()) # point 3
print("\nAs seen, there're no missed values, thus we will create some.")

# creating NaN values in random places
mask_danceability = np.random.rand(len(df)) < 0.05
mask_mode = np.random.rand(len(df)) < 0.05

df.loc[mask_danceability, 'danceability'] = np.nan
df.loc[mask_mode, 'mode'] = np.nan

print("\n(NEW)Skipped values for each column (before):\n", df.isnull().sum())