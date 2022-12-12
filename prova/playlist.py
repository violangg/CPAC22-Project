# %% Import what we need

import os
import json
import time
import requests
import urllib
import numpy as np
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import your_code

audio_feature_url="https://api.spotify.com/v1/audio-features"
add_item_playlist_url="https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
create_playlist_url="https://api.spotify.com/v1/users/{user_id}/playlists"
# %% Get the token
# 1) go to https://developer.spotify.com/console/post-playlists/
# 2) press "get token"
# 3) remember to include playlist-modify-public 
# 4) login
# 5) agree 
# 6) execute this cell and give the script the token (see above)
if "token" not in locals(): # if you have not inserted the token 
    token=input("Please, give me your token\n")
header={"Authorization": "Bearer %s"%token}


# %% Get the list of songs

assert os.path.exists("list_of_songs.json"), "Please put here a list of songs"
with open("list_of_songs.json",'r') as fp:
    ids=json.load(fp)["ids"]

# %% Get the audio features
audio_features=[]
for id_ in ids:
    params={"ids":id_}
    req=requests.get(url=audio_feature_url, params=params, headers=header)
    assert req.status_code==200, req.content
    audio_features_song=req.json()["audio_features"][0]
    audio_features.append(audio_features_song)
    time.sleep(1) # wait 1 second between the questions
# %% Now let's create some way to organize them!

shuffled_songs=your_code.sort_songs(audio_features)

# %% Create the playlist
# Go to https://open.spotify.com/ , top right corner, press "Account"
# look at your username or user_id
name_playlist=input("What's the name of the playlist you want to create?\n")
user_id=input("What's your username?\n")

params={"name":name_playlist, "description": "made during cpac!"}


# %% Create the playlist
req=requests.post(url=create_playlist_url.format(user_id=user_id), 
                    json=params, headers=header)
assert req.status_code==201, req.content
playlist_info=req.json()
print("Playlist created with url %s"%playlist_info["external_urls"]["spotify"])
# %% Populating the playlist
# Doc at https://developer.spotify.com/documentation/web-api/reference/playlists/add-tracks-to-playlist/
uris=[]
for song in shuffled_songs:
    uris.append(song["uri"])
params={"uris":uris, }
req=requests.post(url=add_item_playlist_url.format(playlist_id=playlist_info["id"]), json=params, headers=header)
assert req.status_code==201, req.content
playlist_info_songs=req.json()
# %%
