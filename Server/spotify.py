import spotipy
import sys
from dotenv import load_dotenv
import os

load_dotenv()

client_id = os.getenv('SPOTIFY_CLIENT_ID')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

client_credentials_manager = spotipy.oauth2.SpotifyClientCredentials(client_id, client_secret)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def getIdByArtist(artist_name):
    results = spotify.search(q="artist:" + artist_name, type="artist")
    items = results["artists"]["items"]
    artist = items[0]
    artist_id = artist["id"]
    return artist_id


def getTopSongs(artist_name):
    num = 3  # 調べたい楽曲数(最大10)
    try:
        artist = ""
        search_id = getIdByArtist(artist_name)
        artist_top_tracks = spotify.artist_top_tracks(search_id, country="JP")["tracks"]
        artist += artist_name + ","
        if(artist_top_tracks == None):
            return ""
        for i in range(num):
            artist += artist_top_tracks[i]["name"] + ","
        return artist
    except IndexError:
        print("IndexError has occurred!")
    except AttributeError:
        print("AttributeError has occurred!")


if __name__ == '__main__':
    res = ""
    data = sys.stdin.readline().rstrip("\n").split(",")
    for artist in data:
        res += getTopSongs(artist)
    print(res.rstrip(","))
