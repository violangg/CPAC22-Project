import base64
import requests
from django_cpac.secrets import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

def get_authorization_url():
    AUTH_URL = "https://accounts.spotify.com/authorize"
    params = {
        "client_id": SPOTIPY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIPY_REDIRECT_URI,
        "scope": "user-read-recently-played"
    }
    auth_url = AUTH_URL + "?" + "&".join([f"{k}={v}" for k, v in params.items()])
    return auth_url
