from django.shortcuts import render
from MuPo.spotify import get_authorization_url
from django.shortcuts import redirect
from django_cpac.secrets import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI
import requests
from spotipy.oauth2 import SpotifyOAuth
import json

def index(request):
    auth_url= get_authorization_url()
    return render(request, 'index.html', {'auth_url': auth_url})

def postauth(request):
    sp_oauth = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=['user-read-recently-played'])
    code = request.GET.get('code')
    token = sp_oauth.get_access_token(code)
    print(f"This is your access token: {token}")
    access_token = token['access_token']
    tracks = get_recently_played(access_token)

    return render(request, 'postauth.html', {'tracks': tracks})



def get_access_token(request):
    token_info = request.session.get('token_info', {})
    access_token = token_info.get('access_token', None)
    return access_token

def get_recently_played(access_token):
    url = "https://api.spotify.com/v1/me/player/recently-played?limit=5"
    headers = { 
        "Accept": "application/json", 
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {access_token}" 
        }
    response = requests.get(url, headers=headers) 

    if response.status_code != 200: 
        print("There was an error retrieving the songs from Spotify.")     
    else: 
        data = json.loads(response.text) 
    
    tracks = []
    for track in data["items"]: 
        tracks.append(track['track']['id'])
        print(f"Track: {track['track']['name']}")
    print(tracks)

    # return response.json()
    return tracks


