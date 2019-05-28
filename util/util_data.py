'''
 Load / Save data
'''

import numpy as np
import pandas as pd
import json


## TXT format
def save_list_as_txt(obj, path):
    with open(path, 'w') as f:
        for item in obj:
            f.write("%s\n" % item)

def load_txt_to_list(path):
    with open(path, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    return lines


## CSV format
def save_list_as_csv(obj, path, _column_names):
    df = pd.DataFrame(obj, columns=_column_names)
    df.to_csv(path, index=False)
    
def load_csv_to_list(path, _delim=','):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=_delim)
        csv_list = list(csv_reader)
    return csv_list

def load_csv_to_df(path, out_format='tuple', _delim=','):
    curr_df = pd.read_csv(path, delimiter=_delim)
    if out_format == 'tuple':
        return [tuple(x) for x in curr_df.values]
    elif out_format == 'dict':
        return curr_df.to_dict().values()

    
## JSON format    
def save_dict_as_json(obj, path):
    with open(path, 'w') as outfile:  
        json.dump(obj, outfile)

def load_json_to_dict(path):
    f = open(path, encoding="latin-1")
    js = f.read()
    f.close()
    return json.loads(js)