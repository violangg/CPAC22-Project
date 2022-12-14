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
  for track in data["items"]: 
    print(f"Track: {track['track']['name']}")