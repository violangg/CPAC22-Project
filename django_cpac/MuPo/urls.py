from django.urls import path
from .views import index, postauth, upload

urlpatterns = [
    path('', index, name='index'),
    # path('spotify_auth', spotify_auth, name='spotify_auth'),
    # path('spotify_callback', spotify_callback, name='spotify_callback'),
    path('postauth/', postauth, name='postauth'),
    path('upload/', upload, name='upload'),

]
