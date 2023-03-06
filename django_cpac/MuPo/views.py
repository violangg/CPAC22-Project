from django.shortcuts import render
from MuPo.spotify import get_authorization_url
import requests

from django.shortcuts import redirect
from spotipy.oauth2 import SpotifyOAuth
from django_cpac.secrets import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

def index(request):
    auth_url = get_authorization_url()
    return render(request, 'index.html', {'auth_url': auth_url})

def spotify_auth(request):
    sp_oauth = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=['user-library-read'])
    auth_url = sp_oauth.get_authorize_url(show_dialog=True)
    return redirect(auth_url)

def spotify_callback(request):
    code = request.GET.get('code')
    sp_oauth = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=['user-library-read'])
    token_info = sp_oauth.get_access_token(code)
    request.session['token_info'] = token_info
    return redirect('postauth')

def exchange_code(request):
    AUTH_TOKEN_URL = "https://accounts.spotify.com/api/token"
    code = request.GET.get('code')
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIPY_REDIRECT_URI,
        "client_id": SPOTIPY_CLIENT_ID,
        "client_secret": SPOTIPY_CLIENT_SECRET
    }
    response = requests.post(AUTH_TOKEN_URL, data=data)
    response_data = response.json()
    access_token = response_data['access_token']
    refresh_token = response_data['refresh_token']
    expires_in = response_data['expires_in']
    # Save access token and refresh token in database for the user
    return redirect('postauth')

def postauth(request):
    return render(request, 'postauth.html')

