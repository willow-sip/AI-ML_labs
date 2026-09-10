import pandas as pd

# import kagglehub
# path = kagglehub.dataset_download("prasertk/spotify-global-2019-moststreamed-tracks")
# print("Path to dataset files:", path)

df = pd.read_csv("spotify_global_2019_most_streamed_tracks_audio_features.csv") # point 1

print("First 5 records:\n", df.head()) # point 2