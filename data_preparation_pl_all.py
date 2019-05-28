import html
import pickle
import re
import spotipy
import spotipy.util as sp_util
import numpy as np
import scipy.stats
import os

from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulStoneSoup, BeautifulSoup
from tqdm import tqdm

# import mpd_handling.descriptions
from util.util_data import *


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def compute_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return scipy.stats.entropy(counts, base=base)


mpd_data_path = '/media/irene/dataset/mpd.v1/data'
tr_info_path = '/media/irene/dataset/mpd.v1/data_tr'

track_features_to_investigate = ['popularity',
                                 'acousticness',
                                 'danceability',
                                 'energy',
                                 'instrumentalness',
                                 'liveness',
                                 'loudness',
                                 'speechiness',
                                 'tempo',
                                 'valence',
                                 'album_release_year']


all_pl_ids = []
pl_X = dict()
pl_Y_title = dict()

pl_id_to_key_dict = dict()
pl_key_to_id_dict = dict()

pl_X_list = []
pl_Y_title_list = []

_p_key = 0

filenames = os.listdir(mpd_data_path)
for filename in tqdm(sorted(filenames)):
    if filename.startswith("mpd.slice.") and filename.endswith(".json"):
        fullpath = os.sep.join((mpd_data_path, filename))
        f = open(fullpath, encoding="latin-1")
        js = f.read()
        f.close()
        mpd_slice = json.loads(js)
        for idx in range(len(mpd_slice['playlists'])):
            _pl = mpd_slice['playlists'][idx]
            pl_id = _pl['pid']
            all_pl_ids.append(pl_id)

            _pl_feature = []

            _curr_track_list = _pl['tracks']
            _curr_pl_feature_list_dict = dict()

            for _tfidx in range(len(track_features_to_investigate)):
                _curr_pl_feature_list_dict[track_features_to_investigate[_tfidx]] = list()

            for _tr in _curr_track_list:
                _curr_track_id = _tr['track_uri'].split(':')[-1]
                _curr_track_path = os.path.join(tr_info_path, _curr_track_id + '.json')
                _curr_track_info = load_json_to_dict(_curr_track_path)

                # popularity
                if _curr_track_info['info'] is not None:
                    _curr_pl_feature_list_dict[track_features_to_investigate[0]].append(
                        _curr_track_info['info'][track_features_to_investigate[0]])
                else:
                    print('info err : ' + _curr_track_path)
                    continue

                # others
                if _curr_track_info['audio_feature'] is not None:
                    for _tfidx in range(1, len(track_features_to_investigate) - 1):
                        _curr_pl_feature_list_dict[track_features_to_investigate[_tfidx]].append(
                            _curr_track_info['audio_feature'][track_features_to_investigate[_tfidx]])
                else:
                    print('audio_feature err : ' + _curr_track_path)
                    continue

                # release year
                if _curr_track_info['audio_feature'] is not None:
                    _curr_pl_feature_list_dict[track_features_to_investigate[-1]].append(
                        int(_curr_track_info['info']['album']['release_date'].split('-')[0]))
                else:
                    print('audio_feature err : ' + _curr_track_path)
                    continue

            ## (1) add track level characteristics


            try:
                # Compute averages
                _avgs = []
                for _tfidx in range(len(track_features_to_investigate)):
                    _tmp_arr = np.array(_curr_pl_feature_list_dict[track_features_to_investigate[_tfidx]])
                    _avgs.append(np.average(_tmp_arr))

                # Compute variance
                _variances = []
                for _tfidx in range(len(track_features_to_investigate)):
                    _tmp_arr = np.array(_curr_pl_feature_list_dict[track_features_to_investigate[_tfidx]])
                    _variances.append(np.std(_tmp_arr))

                _pl_feature.extend(_avgs)
                _pl_feature.extend(_variances)

                # Compute entropy of album_release_year
                _entropies = []
                for _tfidx in [-1]:
                    _tmp_arr = np.array(_curr_pl_feature_list_dict[track_features_to_investigate[_tfidx]])
                    _entropies.append(compute_entropy(_tmp_arr))

                # ..and ADD more intuitive characteristics..!

                ## (2) add playlist level characteristics

                _pl_feature.extend([_pl['num_albums'],
                                    _pl['num_artists'],
                                    _pl['num_tracks'],
                                    _pl['num_followers']])
                # 1 if _pl['collaborative'] else 0])

                pl_id_to_key_dict[_pl['pid']] = _p_key
                pl_key_to_id_dict[_p_key] = _pl['pid']

                pl_X[_pl['pid']] = _pl_feature
                pl_Y_title[_pl['pid']] = normalize_name(_pl['name'])

                pl_X_list.append(_pl_feature)
                pl_Y_title_list.append(normalize_name(_pl['name']))

                _p_key += 1
            except:
                print('error on average/variance computation : ' + str(idx) + ' in ' + filename)
                pass



save_dict_as_json(pl_X, 'data_pl_all/pl_X_dict.json')
# save_dict_as_json(pl_Y_desc, 'data_pl_all/pl_Y_desc_dict.json')
save_dict_as_json(pl_Y_title, 'data_pl_all/pl_Y_title_dict.json')

save_dict_as_json(pl_id_to_key_dict, 'data_pl_all/pl_id_to_key_dict.json')
save_dict_as_json(pl_key_to_id_dict, 'data_pl_all/pl_key_to_id_dict.json')

save_list_as_txt(pl_X_list, 'data_pl_all/pl_X_list.txt')
# save_list_as_txt(pl_Y_desc_list, 'data_pl_all/pl_Y_desc_list.txt')
save_list_as_txt(pl_Y_title_list, 'data_pl_all/pl_Y_title_list.txt')

pl_X_arr = np.array([pl_X_list])
np.save('data_pl_all/pl_X_arr.npy', pl_X_arr)

'''
Now come up with some playlist feature here :

    1. General
        - num_albums
        - num_artists
        - num_tracks
        - num_followers

    2. Audio characteristics
        - tempo 
        - mode
        - valence
        - dancibility
        - acousticness
        - energy
        - speechiness
        - loudness
        - instrumentalness
'''

'''
_pl['collaborative']            # : 'false',
_pl['description']              # : 'chilllll out',
normalize_name(_pl['name'])     # : 'relax',
_pl['num_albums']               # : 112,
_pl['num_artists']              # : 97,
_pl['num_tracks']               # : 124,
_pl['pid']                      # : 94,

'''

'''
_curr_track_info_list[idx]['info']
       {'album': {'album_type': 'compilation',
                  'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4F7Q5NV6h5TSwCainz8S5A'},
                              'href': 'https://api.spotify.com/v1/artists/4F7Q5NV6h5TSwCainz8S5A',
                              'id': '4F7Q5NV6h5TSwCainz8S5A',
                              'name': 'Duke Ellington',
                              'type': 'artist',
                              'uri': 'spotify:artist:4F7Q5NV6h5TSwCainz8S5A'}],
                  'available_markets': [],
                  'external_urls': {'spotify': 'https://open.spotify.com/album/66XVEGxwEK4mbvsYgNvWXH'},
                  'href': 'https://api.spotify.com/v1/albums/66XVEGxwEK4mbvsYgNvWXH',
                  'id': '66XVEGxwEK4mbvsYgNvWXH',
                  'images': [{'height': 640,
                            'url': 'https://i.scdn.co/image/4db6cece42c0d5f24d5fc09314af21eb0c0d57ca',
                            'width': 640},
                            {'height': 300,
                             'url': 'https://i.scdn.co/image/2108e69a3f63f80a5a6265d9a40f6aef60d2f497',
                             'width': 300},
                            {'height': 64,
                             'url': 'https://i.scdn.co/image/1b4b45dee22e6b37c38b6997329617a29a742111',
                             'width': 64}],
                  'name': 'Duke Ellington and his Famous Orchestra',
                  'release_date': '2013-05-01',
                  'release_date_precision': 'day',
                  'total_tracks': 23,
                  'type': 'album',
                  'uri': 'spotify:album:66XVEGxwEK4mbvsYgNvWXH'},
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4F7Q5NV6h5TSwCainz8S5A'},
                    'href': 'https://api.spotify.com/v1/artists/4F7Q5NV6h5TSwCainz8S5A',
                    'id': '4F7Q5NV6h5TSwCainz8S5A',
                    'name': 'Duke Ellington',
                    'type': 'artist',
                    'uri': 'spotify:artist:4F7Q5NV6h5TSwCainz8S5A'}],
        'available_markets': [],
        'disc_number': 1,
        'duration_ms': 166786,
        'explicit': False,
        'external_ids': {'isrc': 'ITN731004684'},
        'external_urls': {'spotify': 'https://open.spotify.com/track/4l7hHzGzb4X2QosbGA8Wu7'},
        'href': 'https://api.spotify.com/v1/tracks/4l7hHzGzb4X2QosbGA8Wu7',
        'id': '4l7hHzGzb4X2QosbGA8Wu7',
        'is_local': False,
        'name': 'Caravan',
        'popularity': 0,
        'preview_url': None,
        'track_number': 16,
        'type': 'track',
        'uri': 'spotify:track:4l7hHzGzb4X2QosbGA8Wu7'}


_curr_track_info_list[idx]['audio_feature']
e.g.
   {'acousticness': 0.727,
    'analysis_url': 'https://api.spotify.com/v1/audio-analysis/75JFxkI2RXiU7L9VXzMkle',
    'danceability': 0.56,
    'duration_ms': 309600,
    'energy': 0.442,
    'id': '75JFxkI2RXiU7L9VXzMkle',
    'instrumentalness': 1.71e-05,
    'key': 5,
    'liveness': 0.11,
    'loudness': -7.224,
    'mode': 1,
    'speechiness': 0.0243,
    'tempo': 146.448,
    'time_signature': 4,
    'track_href': 'https://api.spotify.com/v1/tracks/75JFxkI2RXiU7L9VXzMkle',
    'type': 'audio_features',
    'uri': 'spotify:track:75JFxkI2RXiU7L9VXzMkle',
    'valence': 0.212}


'''

