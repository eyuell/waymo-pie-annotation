"""
Copyright 2021, Eyuell H Gebremedhin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
Inspiration from:
        https://github.com/Sreeni1204/Waymo_Kitti_converter
        
"""
"""
As OBD step
"""

import os
import time
from datetime import timedelta
from util_obd import extract_OBD


def get_main_path():
    #Path where the input and output folders are available
    return '/home/eyuell/Desktop/ForD/WAYMO/'


def file_names_and_path(source_folder):

    dir_list = os.listdir(source_folder)
    files = list()

    for directory in dir_list:

        path = os.path.join(source_folder, directory)

        if os.path.isdir(path):
            files = files + file_names_and_path(path)
        else:
            files.append(path)

    tf_files = [f for f in files if f.endswith('.tfrecord')]

    return tf_files


if __name__=="__main__":

    main_path = get_main_path()
    source_folder = main_path + 'input/segment/'
    dest_folder = main_path + 'output/'
    annt_dir = dest_folder + 'annot_files/annotations_vehicle/'

    path = File_names_and_path(source_folder)

    print('Extraction process started:')
    start = time.time()

    pp = 0
    m = 0
    for filename in path:
        pp += 1
        print("Working on segment", pp, "of", len(path),"\r")
        tf_name = filename.split('/')[-1]
        vid_name = tf_name.split('_')[0].split('-')[1]
        m = extract_OBD(m, filename, vid_name, annt_dir, pp)

    # Concluding 
    end = time.time()
    elapsed = end - start
    t_del = timedelta(seconds=elapsed)
    print('\nNumber of OBD annotations extracted:', m)
    print('Extraction process complete in Duration= {}\n'.format(t_del))
