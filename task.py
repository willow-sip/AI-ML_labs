import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

#=================================================================

median_val = df['danceability'].median() # point 4 - filling with mediane
df['danceability'] = df['danceability'].fillna(median_val)
print(f"Filled 'danceability' skips with mediane: {median_val}")

mode_val = df['mode'].mode()[0] # point 4 - filling with mode
df['mode'] = df['mode'].fillna(mode_val)
print(f"Filled 'mode' skips with mode: {mode_val}")

print("\nSkipped values for each column (after):\n", df.isnull().sum().sum())


#=================================================================

features_for_normalization = [ # point 5 - Z-normalization
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

cols_to_scale = [col for col in features_for_normalization if col in df.columns]

if cols_to_scale:
    scaler = StandardScaler()
    
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    print("Normalization done for columns:", cols_to_scale)
    print("\nExample of 5 normalized rows:\n")
    print(df[cols_to_scale].head())
else:
    print("No columns for normalization found in dataset")