# Import necessary modules 
import requests 
import json 
 
# Ask user for their Spotify token 
spotify_token = input("Please enter your Spotify token: ") 
 
# Set the necessary headers for the API request 
headers = { 
  "Accept": "application/json", 
  "Content-Type": "application/json", 
  "Authorization": f"Bearer {spotify_token}" 
} 
 
# Send a GET request to the Spotify API to retrieve the last 5 songs listened 
response = requests.get("https://api.spotify.com/v1/me/player/recently-played?limit=5", headers=headers) 
 
# Check the status code of the response to make sure it was successful 
if response.status_code != 200: 
  print("There was an error retrieving the songs from Spotify.") 
 
# If the request was successful, parse the response data 
else: 
  data = json.loads(response.text) 
 
  # Loop through the list of songs and print their names 
  tracks = []
  for track in data["items"]: 
    tracks.append(track['track']['id'])
    print(f"Track: {track['track']['name']}")

print(tracks)

valence_values = []
arousal_values = []
for i in range(5):
    # Replace "SONG_ID" with the ID of the song you want to analyze
    song_id = tracks[i]

    # Make a GET request to the /audio-features endpoint of the Spotify Web API
    response = requests.get(
        f"https://api.spotify.com/v1/audio-features/{song_id}",
        headers={"Authorization": f"Bearer {spotify_token}"},
    )

    # Parse the response as JSON
    audio_features = response.json()

    # Print the valence and arousal values for the song
    valence_values.append(audio_features["valence"])
    arousal_values.append(audio_features["energy"])
    print("Valence:", audio_features["valence"])
    print("Arousal:", audio_features["energy"])


#Show results
import numpy as np
import matplotlib.pyplot as plt

X = ['Song 1', 'Song 2', 'Song 3', 'Song 4', 'Song 5']
  
X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, valence_values, 0.4, label = 'Valence')
plt.bar(X_axis + 0.2, arousal_values, 0.4, label = 'Arousal')

plt.show()