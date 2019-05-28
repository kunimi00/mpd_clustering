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

# from mpd_handling.descriptions import *
from util.util_data import *


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def compute_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return scipy.stats.entropy(counts, base=base)

# tr_info_path = '/media/irene/dataset/mpd.v1/data_tr'
all_pl_with_dzrmeta = load_json_to_dict('data_pl_dzrmeta/pl_with_tracks_on_dzrmeta_50.json')
spotify_id_to_metadata_dict = load_json_to_dict('data_pl_dzrmeta/spotify_id_to_metadata_dict.json')
dzr_metadata_sets_dict = load_json_to_dict('data_pl_dzrmeta/dzr_metadata_sets_dict.json')

# Measuring semantic cohesion
# (1) frequency counts of metadata co-occurrence
# (2) entropy of genre distribution /

track_features_to_investigate = ['category',
                                 'decade',
                                 'genre',
                                 'instrument',
                                 'lang',
                                 'location',
                                 'location:continent',
                                 'mood',
                                 'mood:activity',
                                 'mood:emotion',
                                 'mood:situation',
                                 'singer_type'
                                 ]

pl_X = dict()
pl_Y_desc = dict()
pl_Y_title = dict()

pl_id_to_key_dict = dict()
pl_key_to_id_dict = dict()

pl_X_list = []
pl_Y_desc_list = []
pl_Y_title_list = []

for _idx in tqdm(range(len(all_pl_with_dzrmeta))):
    _pl = all_pl_with_dzrmeta[_idx]
    _pl_feature = []

    _curr_track_list = _pl['tracks']
    _curr_pl_dict_for_feature_counts= dict()
    _curr_pl_dict_for_feature_list = dict()

    for tr_feature in track_features_to_investigate:
        _curr_pl_dict_for_feature_counts[tr_feature] = np.zeros(len(dzr_metadata_sets_dict[tr_feature]))
        _curr_pl_dict_for_feature_list[tr_feature] = []

    # 1) get feature counts.
    for _tr in _curr_track_list:
        _curr_track_id = _tr['track_uri'].split(':')[-1]
        try:
            _curr_track_info = spotify_id_to_metadata_dict[_curr_track_id]

            for tr_feature in track_features_to_investigate:
                if len(_curr_track_info[tr_feature]) > 0:
                    for _f_item in _curr_track_info[tr_feature]:
                        _curr_f_idx = dzr_metadata_sets_dict[tr_feature].index(_f_item)
                        _curr_pl_dict_for_feature_counts[tr_feature][_curr_f_idx] += 1

                        _curr_pl_dict_for_feature_list[tr_feature].append(_f_item)

                else:
                    continue
        except:
            pass

    # 2) get feature cohesion measure for each playlist
    # Entorpy
    _entropies = [0 for _ in range(len(track_features_to_investigate))]
    for _tidx in range(len(track_features_to_investigate)):
        tr_feature = track_features_to_investigate[_tidx]

        if len(_curr_pl_dict_for_feature_list[tr_feature]) > 0:
            _entropies[_tidx] = compute_entropy(_curr_pl_dict_for_feature_list[tr_feature])

    _pl_feature.extend(_entropies)

    ## (2) add playlist level characteristics
    _pl_feature.extend([_pl['num_albums'],
                        _pl['num_artists'],
                        _pl['num_tracks'],
                        _pl['num_followers']])

    pl_id_to_key_dict[_pl['pid']] = _idx
    pl_key_to_id_dict[_idx] = _pl['pid']

    # print(_idx, _pl_feature)
    pl_X[_pl['pid']] = _pl_feature
    pl_Y_title[_pl['pid']] = normalize_name(_pl['name'])

    pl_X_list.append(_pl_feature)
    pl_Y_title_list.append(normalize_name(_pl['name']))

save_dict_as_json(pl_X, 'data_pl_dzrmeta/pl_X_dict.json')
save_dict_as_json(pl_Y_title, 'data_pl_dzrmeta/pl_Y_title_dict.json')

save_dict_as_json(pl_id_to_key_dict, 'data_pl_dzrmeta/pl_id_to_key_dict.json')
save_dict_as_json(pl_key_to_id_dict, 'data_pl_dzrmeta/pl_key_to_id_dict.json')

save_list_as_txt(pl_X_list, 'data_pl_dzrmeta/pl_X_list.txt')
save_list_as_txt(pl_Y_title_list, 'data_pl_dzrmeta/pl_Y_title_list.txt')

pl_X_arr = np.array([pl_X_list])
np.save('data_pl_dzrmeta/pl_X_arr.npy', pl_X_arr)




'''
{'audience': 88113,
 'audio_content': 85757,
 'belief': 115972,
 'category': 1745150, *
 'decade': 3598600, *
 'formation': 171,
 'genre': 3272354, *
 'influence': 15815,
 'instrument': 2053880, *
 'lang': 3399127, *
 'location': 781254, *
 'location:city': 116899,
 'location:continent': 752589, *
 'location:region': 423608,
 'mood': 189547,
 'mood:activity': 507979, *
 'mood:celebration': 119215,
 'mood:day': 12768,
 'mood:day_moment': 71506,
 'mood:emotion': 1001879, *
 'mood:moment': 29188,
 'mood:season': 31285,
 'mood:situation': 322101, *
 'mood:weather': 2813,
 'movie:genre': 6698,
 'musical_form': 9024,
 'period': 19322,
 'record_type': 125267,
 'role': 39231,
 'singer_type': 212325} * 

'''