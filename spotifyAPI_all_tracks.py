'''

Get Spotify feature information for all unique tracks in MPD.

'''

from tqdm import tqdm
import html
import pickle
import re
import spotipy
import spotipy.util as sp_util
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulStoneSoup, BeautifulSoup
import os
import numpy as np
from util.util_data import *
import requests
from time import sleep
import argparse

def save_obj_curr_folder(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_curr_folder(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--startnum', type=int, default=0, help='')
    parser.add_argument('--endnum', type=int, default=0, help='')
    args = parser.parse_args()

    client_credentials_manager = SpotifyClientCredentials(client_id='7b64a7f89d0247589b529a5ded859da4', client_secret='0699b27eabc74cc3ba74077edd221f2b')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    unique_track_ids_from_pl = load_obj_curr_folder('./data_pl_all/unique_track_ids_from_pl_all.p')
    tr_info_dir = '/media/irene/dataset/mpd.v1/data_tr/'

    batch_size = 50    # Max number of tracks for a single API request.
    num_batches = len(unique_track_ids_from_pl) // batch_size

    print(num_batches)

    for _bidx in range(args.startnum, args.endnum):
        if _bidx == num_batches:
            _curr_tracks = unique_track_ids_from_pl[_bidx * batch_size:]
        else:
            _curr_tracks = unique_track_ids_from_pl[_bidx * batch_size:(_bidx+1) * batch_size]

        print(str(_bidx) + '/' + str(num_batches) + ' : ' + str(len(_curr_tracks)))



        while True:
            try:
                _curr_tracks_info_list = sp.tracks(tracks=_curr_tracks)['tracks']
                _curr_tracks_audio_feature_list = sp.audio_features(tracks=_curr_tracks)

                ## Checking Null responses --> Decided to just ignore them.
                # while(True):
                #     _curr_tracks_info_list = sp.tracks(tracks=_curr_tracks)['tracks']
                #     _curr_tracks_audio_feature_list = sp.audio_features(tracks=_curr_tracks)
                #     if None in _curr_tracks_info_list or None in _curr_tracks_audio_feature_list:
                #         if None in _curr_tracks_info_list:
                #             print('Null found in info \n', _curr_tracks_info_list.index(None))
                #             print(_curr_tracks[_curr_tracks_info_list.index(None)])
                #         else:
                #             print('Null found in audio \n', _curr_tracks_audio_feature_list.index(None))
                #             print(_curr_tracks[_curr_tracks_audio_feature_list.index(None)])
                #         sleep(10)
                #         continue
                #     else:
                #         break

            except requests.exceptions.ConnectionError:
                print("Let me sleep for 5 seconds")
                sleep(5)
                print("Was a nice sleep, now let me continue...")
                _curr_tracks_info_list = sp.tracks(tracks=_curr_tracks)['tracks']
                _curr_tracks_audio_feature_list = sp.audio_features(tracks=_curr_tracks)
                continue

            break

        assert len(_curr_tracks_info_list) == len(_curr_tracks_audio_feature_list)

        for _tidx in range(len(_curr_tracks_info_list)):
            _curr_track_info = dict()
            _curr_track_info['info'] = _curr_tracks_info_list[_tidx]
            _curr_track_info['audio_feature'] = _curr_tracks_audio_feature_list[_tidx]

            _curr_track_id = unique_track_ids_from_pl[_bidx * batch_size + _tidx]
            save_dict_as_json(_curr_track_info, os.path.join(tr_info_dir, str(_curr_track_id) + '.json'))


if __name__ == '__main__':
    main()
