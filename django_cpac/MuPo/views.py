from django.shortcuts import render
from MuPo.spotify import get_authorization_url
from django.shortcuts import redirect
from django_cpac.secrets import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI
import requests
from spotipy.oauth2 import SpotifyOAuth
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


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

    x1, y1 = map_features(tracks, access_token)
    # change_mood(x1, y1)

    request.session['x1'] = x1
    request.session['y1'] = y1
    print(f'Coordinates stored: {x1}, {y1}')

    return render(request, 'postauth.html', {'tracks': tracks})

def upload(request):
    process_image(request)
    return render(request, 'upload.html')

def process_image(request):
    if request.method == 'POST':
        image = request.FILES['image']
        if image is not None:
            print('Non ci siamo')

        x1 = request.session.get('x1')
        y1 = request.session.get('y1')
        print(f'Coordinates retrieved: {x1}, {y1}')
        moody_img = change_mood(image, x1, y1)
        print('Got moody image')

        return render(request, 'result.html', {'image_url': image.url})

    return render(request, 'upload.html')


def map_features(tracks, token): 
    valence_values = []
    arousal_values = []

    for track in tracks:
        song_id = track['id']

        response = requests.get(
            f"https://api.spotify.com/v1/audio-features/{song_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        audio_features = response.json()
        print(audio_features)
        valence_values.append(audio_features["valence"])
        arousal_values.append(audio_features["energy"])

    coordinates = np.zeros((5,2))

    for i in range(5):
        x = valence_values[i] * 2 - 1
        y = arousal_values[i] * 2 - 1
        coordinates[i, 0] = x
        coordinates[i, 1] = y

    x0 = coordinates[0, 0]
    y0 = coordinates[0, 1]
    x2 = coordinates[1, 0]
    y2 = coordinates[1, 1]
    x3 = coordinates[2, 0]
    y3 = coordinates[2, 1]
    x4 = coordinates[3, 0]
    y4 = coordinates[3, 1]
    x5 = coordinates[4, 0]
    y5 = coordinates[4, 1]

    x1 = np.mean([x0, x2, x3, x4, x5])
    y1 = np.mean([y0, y2, y3, y4, y5])
    print('Coordinates correctly computed')

    return x1, y1

def change_mood(uploaded_img, x1, y1):
    # img = cv2.imread('/Users/violanegroni/Documents/GitHub/CPAC22-Project/django_cpac/roy.jpeg')
    img = uploaded_img

    # Quadrant I: x > 0, y > 0
    if x1 > 0 and y1 > 0:
        colormap = cv2.COLORMAP_SPRING # yellow/pink, euphoria
    # Quadrant II: x < 0, y > 0
    elif x1 < 0 and y1 > 0:
        colormap = cv2.COLORMAP_HOT #red/yellow, anger
    # Quadrant III: x < 0, y < 0
    elif x1 < 0 and y1 < 0:
        colormap = cv2.COLORMAP_OCEAN # blue, sadness  
    # Quadrant IV: x > 0, y < 0
    elif x1 > 0 and y1 < 0:
        colormap = cv2.COLORMAP_SUMMER #green, relax

    img = cv2.applyColorMap(img, colormap)
    cv2.imwrite('image_ready.png', img)

    print('Image processed and saved.')

    return img


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
    for item in data["items"]:
        track_info = {
            'id': item['track']['id'],
            'title': item['track']['name'],
            'artist': item['track']['artists'][0]['name'],
            'album_cover': item['track']['album']['images'][0]['url'],
            'preview_url': item['track']['preview_url']
        }
        tracks.append(track_info)
        print(f"Track: {track_info['title']}")
        print(track_info)

    return tracks