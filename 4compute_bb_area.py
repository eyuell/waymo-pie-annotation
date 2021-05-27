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

import os
import xml.etree.ElementTree as ET


# Method taken from https://github.com/Sreeni1204/Waymo_Kitti_converter
def File_names_and_path(source_folder):
    
    dir_list = os.listdir(source_folder)
    files = list()

    for directory in dir_list:

        path = os.path.join(source_folder, directory)

        if os.path.isdir(path):
            files = files + File_names_and_path(path)
        else:
            files.append(path)

    annot_files = [f for f in files if f.endswith('_annt.xml')]

    return annot_files


def get_ped_bb_area(path_to_file):
    tree = ET.parse(path_to_file)
    ped_annt = 'ped_annotations'
    
    peds = 0
    bb_area = 0
    
    tracks = tree.findall('./track')
    for t in tracks:
        boxes = t.findall('./box')
        obj_label = t.get('label')
        obj_id = boxes[0].find('./attribute[@name=\"id\"]').text

        if obj_label == 'pedestrian':
            for b in boxes:
                peds += 1
                bb_area +=  (float(b.get('xbr')) - float(b.get('xtl'))) * (float(b.get('ybr')) - float(b.get('ytl')))
                y = True
    
    return peds, bb_area


if __name__ == '__main__':
    #location of annotation files where the bounding boxes could be computed
    source_folder = 'C:/Users/eyuel/Desktop/WinShare/PIEPredict/PIE/annotations/'
    
    path = File_names_and_path(source_folder)

    pp = 0
    m = 0
    ped_instances = 0
    total_bb_area = 0.0

    for filename in path:
        pp += 1
        seg_id = filename.split('_annt.xml')[0].split('_')[-1]
        ped_num, bb_area = get_ped_bb_area(filename)
        
        ped_instances += ped_num
        total_bb_area += bb_area

        # To show for each segment
        print(pp, seg_id, ped_num, bb_area)

    # Summarized
    if pp:
        print("=========================================")
        print("# of videos:", pp)
        print("Pedestrian instances:", ped_instances)
        print("Total Bounding Box Area:", total_bb_area)
        print("=========================================")
