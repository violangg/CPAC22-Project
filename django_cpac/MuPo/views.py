from django.shortcuts import render
from MuPo.spotify import get_authorization_url
from django.shortcuts import redirect
from spotipy.oauth2 import SpotifyOAuth
from django_cpac.secrets import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI
import requests

def index(request):
    auth_url = get_authorization_url()
    return render(request, 'index.html', {'auth_url': auth_url})

def spotify_auth(request):
    sp_oauth = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=['user-read-recently-played'])
    auth_url = sp_oauth.get_authorize_url(show_dialog=True)
    return redirect(auth_url)

def spotify_callback(request):
    code = request.GET.get('code')
    sp_oauth = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=['user-read-recently-played'])
    token_info = sp_oauth.get_access_token(code)
    request.session['token_info'] = token_info
    return redirect('postauth')

def postauth(request):
    return render(request, 'postauth.html')

def get_access_token(request):
    token_info = request.session.get('token_info', {})
    access_token = token_info.get('access_token', None)
    return access_token

def get_recently_played(access_token):
    url = "https://api.spotify.com/v1/me/player/recently-played?limit=5"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    return response.json()


