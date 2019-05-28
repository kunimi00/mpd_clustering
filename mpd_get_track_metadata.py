from tqdm import tqdm
import html
import pickle
import re
import spotipy
import spotipy.util as sp_util
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulStoneSoup, BeautifulSoup
from mpd_handling.descriptions import *
import numpy as np
import os


def save_obj_curr_folder(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_curr_folder(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

mpd_data_path = '/media/irene/dataset/mpd.v1/data'

# all_pl_with_description = pickle.load(open('mpd_pl_with_description.p', 'rb'))

filenames = os.listdir(mpd_data_path)

pl_with_description = []
num_pl_with_both = 0
unique_title_clean_set = set()
unique_title_org_set = set()
unique_title_norm_set = set()




msd_id_to_spotify_id_dict = load_obj_curr_folder('data_lastfm/msd_id_to_spotify_id_dict.p')
spotify_id_to_msd_id_dict = load_obj_curr_folder('data_lastfm/spotify_id_to_msd_id_dict.p')

client_credentials_manager = SpotifyClientCredentials(client_id='7b64a7f89d0247589b529a5ded859da4', client_secret='0699b27eabc74cc3ba74077edd221f2b')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


MSD_id_to_7D_id = pickle.load(open('/media/iu/MSD/MSD_mel_split/MSD_split/MSD_id_to_7D_id.pkl', 'rb'))
sevenD_id_to_path = pickle.load(open('/media/iu/MSD/MSD_mel_split/MSD_split/7D_id_to_path.pkl', 'rb'))

lastfm_full_track_to_tags_txt_dict = load_obj_curr_folder('data_lastfm/lastfm_full_track_to_tags_txt_dict.p')
msd_to_spotify_file_path = './msd_to_spotify.tsv'
mel_base_dir = '/media/iu/MSD/mel128/'

msd50_track_key_to_tag_key_binary_matrix = load_obj_curr_folder('data_lastfm/msd50_track_key_to_tag_key_binary_matrix.p')
msd50_track_key_to_tag_bin_dict = load_obj_curr_folder('data_lastfm/msd50_track_key_to_tag_bin_dict.p')
msd50_track_id_to_key_dict = load_obj_curr_folder('data_lastfm/msd50_track_id_to_key_dict.p')
msd50_track_key_to_id_dict = load_obj_curr_folder('data_lastfm/msd50_track_key_to_id_dict.p')
msd50_tag_ids_in_order = load_obj_curr_folder('data_lastfm/msd50_tag_ids_in_order.p')
msd50_tag_key_to_id_dict = load_obj_curr_folder('data_lastfm/msd50_tag_key_to_id_dict.p')
msd50_track_id_to_file_path_dict = load_obj_curr_folder('data_lastfm/msd50_track_id_to_file_path_dict.p')
lastfm_full_track_to_tags_txt_dict = load_obj_curr_folder('data_lastfm/lastfm_full_track_to_tags_txt_dict.p')

err_matching_lastfm_tid_set= set()

all_tid_set = set()

for filename in tqdm(sorted(filenames)):
    if filename.startswith("mpd.slice.") and filename.endswith(".json"):
        curr_pl_info_dir = os.sep.join(('/media/irene/dataset/mpd.v1/pl_info', filename.replace(".json", "")))
        curr_json_dir = os.sep.join((mpd_data_path, filename))

        f = open(curr_json_dir, encoding="latin-1")
        js = f.read()
        f.close()
        mpd_slice = json.loads(js)

        for idx in range(len(mpd_slice['playlists'])):
            _curr_playlist = dict()
            _curr_playlist['pid'] = mpd_slice['playlists'][idx]['pid']
            name = mpd_slice['playlists'][idx]['name']
            _curr_playlist['name'] = normalize_name(name)
            _curr_playlist['num_albums'] = mpd_slice['playlists'][idx]['num_albums']
            _curr_playlist['num_artists'] = mpd_slice['playlists'][idx]['num_artists']
            _curr_playlist['num_tracks'] = mpd_slice['playlists'][idx]['num_tracks']
            _curr_playlist['is_collaborative'] = mpd_slice['playlists'][idx]['collaborative']

            if 'description' in list(mpd_slice['playlists'][idx].keys()):
                description = mpd_slice['playlists'][idx]['description']
                _curr_playlist['description'] = BeautifulSoup(description).string
            else:
                _curr_playlist['description'] = ''

            curr_tracks = mpd_slice['playlists'][idx]['tracks']

            curr_track_info_list = []
            for _track in curr_tracks:
                _curr_track_info = dict()
                _curr_track_info['tid'] = _track['track_uri'].split(':')[-1]
                _curr_track_id = _curr_track_info['tid']
                _curr_track_info['info'] = sp.track(_curr_track_id)
                # _curr_track_info['audio_feature'] = sp.audio_features(tracks=[_curr_track_id])
                # _curr_track_info['audio_analysis'] = sp.audio_analysis(_curr_track_id)
                all_tid_set.add(_curr_track_id)

                _MSD_id = spotify_id_to_msd_id_dict[_curr_track_id]

                try:
                    _MSD_id = spotify_id_to_msd_id_dict[_curr_track_id]

                    _curr_track_info['audio_path'] = sevenD_id_to_path[MSD_id_to_7D_id[_MSD_id]]

                    _curr_track_info['50_tag_bin'] = msd50_track_key_to_tag_bin_dict[msd50_track_id_to_key_dict[_MSD_id]]

                    _lastfm_50_tag_bin = _curr_track_info['50_tag_bin']
                    _curr_track_info['50_tag_list'] = [msd50_tag_ids_in_order[_i] for _i in
                                                       np.argwhere(_lastfm_50_tag_bin == 1).squeeze()]
                    _curr_track_info['full_tag_list'] = lastfm_full_track_to_tags_txt_dict[
                        spotify_id_to_msd_id_dict[_curr_track_id]]

                    print("   OK --> " + _curr_track_id + ':' + _track['track_name'])


                except Exception as e:
                    # print(e)
                    print(_curr_track_id + ':' + _track['track_name'] + " : can't be matched with Last.fm data.")
                    # print('')
                    err_matching_lastfm_tid_set.add(_curr_track_id)
                    _curr_track_info['audio_path'] = ''
                    _curr_track_info['50_tag_bin'] = np.zeros((50,))
                    _curr_track_info['50_tag_list'] = []
                    _curr_track_info['full_tag_list'] = []
                # print("   OK --> " + _curr_track_id + ':' + _track['track_name'])
                curr_track_info_list.append(_curr_track_info)

            _curr_playlist['tracks_info'] = curr_track_info_list
            # save_obj_curr_folder(_curr_playlist, os.path.join(curr_pl_info_dir, str(_curr_playlist['pid']) + '.p'))
            print('Saved :' + os.path.join(curr_pl_info_dir, str(_curr_playlist['pid']) + '.p'))


# save_obj_curr_folder(err_matching_lastfm_tid_set, './err_matching_lastfm_tid_set.p')
# save_obj_curr_folder(all_tid_set, './all_tid_set.p')

# print('err_matching_lastfm_tid_list : ', str(len(err_matching_lastfm_tid_set)))
print('all_tid_set : ', str(len(all_tid_set)))