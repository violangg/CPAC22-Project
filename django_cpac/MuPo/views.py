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
from .models import Coordinates
import os
from django.conf import settings
import torch
from NeuralNetworks import TransformerNetwork, load_image, itot, ttoi, transfer_color

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

    map_features(request, tracks, access_token)

    return render(request, 'postauth.html', {'tracks': tracks})

def upload(request):
    if request.method == 'POST':
        processed_image_data = process_image(request)

        if processed_image_data is not None: 
            save_processed_image(processed_image_data)           
            return redirect('result')
    
    return render(request, 'upload.html')

def result(request):
    image_path = 'processed_image.png'
    image_url = settings.MEDIA_URL + image_path

    return render(request, 'result.html', {'image_url': image_url})   




def save_processed_image(img):
    media_root = settings.MEDIA_ROOT
    image_path = os.path.join(media_root, 'processed_image.png')
    cv2.imwrite(image_path, img)
    #cv2.imwrite('processed_image.png', img)

def map_features(request, tracks, token): 
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
    print(f'Coordinates correctly computed: {x1}, {y1}')

    coord = Coordinates(x=x1, y=y1)
    coord.save()

def process_image(request):
    # img = cv2.imread('/Users/violanegroni/Documents/GitHub/CPAC22-Project/django_cpac/roy.jpeg')

    coord = Coordinates.objects.latest('id')
    x1 = coord.x
    y1 = coord.y 
    print(f'Coordinates retrieved: {x1}, {y1}')


    if request.method == 'POST':
        print('POST request received')
        uploaded_file = request.FILES['image']
        if uploaded_file is None:
            print('...but no img retrieved')
    
        img = uploaded_file.read()
        img = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)


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

        # img = cv2.applyColorMap(img, colormap)
        # cv2.imwrite('processed_image.png', img)

        # print('Image processed and saved.')

        # _, img_png = cv2.imencode('.png', img)
        # image_data = img_png.tobytes()
        # return image_data

        img = cv2.applyColorMap(img, colormap)

        img = stylize(img)

        return img

    else:
        print('Not a POST request')
        return render(request, 'upload.html')


def stylize():
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    STYLE_TRANSFORM_PATH = "udnie_aggressive.pth"
    PRESERVE_COLOR = True

    net = TransformerNetwork()
    net.load_state_dict(torch.load(STYLE_TRANSFORM_PATH, map_location=torch.device('cpu')))
    net = net.to(device)

    with torch.no_grad():
        while(1):
            torch.cuda.empty_cache()
            print("Stylize Image~ Press Ctrl+C and Enter to close the program")
            content_image_path = input("Enter the image path: ")
            content_image = load_image(content_image_path)
            content_tensor = itot(content_image).to(device)
            generated_tensor = net(content_tensor)
            generated_image = ttoi(generated_tensor.detach())
            if (PRESERVE_COLOR):
                generated_image = transfer_color(content_image, generated_image)
    
    return generated_image



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