from django.urls import path
from .views import index, spotify_auth, spotify_callback, postauth

urlpatterns = [
    path('', index, name='index'),
    path('spotify_auth', spotify_auth, name='spotify_auth'),
    path('spotify_callback', spotify_callback, name='spotify_callback'),
    path('postauth', postauth, name='postauth'),  
]

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('spotify_auth', views.spotify_auth, name='spotify_auth'),
    path('spotify_callback', views.spotify_callback, name='spotify_callback'),
    path('postauth/', views.postauth, name='postauth'),
]
