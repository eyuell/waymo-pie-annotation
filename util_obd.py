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

import numpy as np
import tensorflow as tf
import itertools
import math
import os
import json
from pyproj import Proj, transform

tf.compat.v1.enable_eager_execution()
from waymo_open_dataset import dataset_pb2 as open_dataset
import util_adapter

pre_pose = []
pre_stamp = 0
pre_velo = []

def get_pose_elements(img_pose):
    poses = []
    raw_aray = str(img_pose).split('transform: ')

    for element in raw_aray:
        if element:
            poses.append(float(element[:len(element) -1]))

    return poses

# Reclaimer: this data is only for intention and trajectory 
# prediction which does not need location information
def localize(long_x, lat_y, loc):
    c_area = loc.split('_')[1]

    # Changing to global location
    if c_area == 'sf': # y: 37.757608, x: -122.444944
        #return (long_x - 195.275), (lat_y + 94,191)
        return (-122 - abs(long_x - round(long_x, 0))), (37 + abs(lat_y - round(lat_y, 0)))
    elif c_area == 'phx': # y: 33.505137, x: -112.079407
        #return (long_x - 176.529), (lat_y - 49.798)
        return (-112 - abs(long_x - round(long_x, 0))), (33 + abs(lat_y - round(lat_y, 0)))
    else:
        #37.397096, -122.081237 Mountain View ?
        #34.150407, -118.396521 Los Angeles ?
        #42.333728, -83.049712 Detroit ?
        #47.614101, -122.316848 Seattle ?
        #print("other than Sf and Pnx: ", loc) Needs Change of X and Y constants accordingly
        return (long_x - 185), (lat_y + 40)


def get_OBD(pose_els, cur_tim, loc):
    global pre_pose
    global pre_stamp
    global pre_velo

    if len(pre_pose):
        dist = np.array([pose_els[3] - pre_pose[3], pose_els[7] - pre_pose[7], pose_els[11] - pre_pose[11] ])
        t_inter = (cur_tim - pre_stamp) / 1000000 # seconds
        velo = np.array(dist) / t_inter
        accn = np.array((velo - pre_velo)) / t_inter
        velo_mag_kmh =  np.sqrt(velo.dot(velo)) * 3.6
    else:
        velo = 0
        velo_mag_kmh = 0
        accn = [0, 0, 0]

    roll = np.arctan2(pose_els[9], pose_els[10]) * 180 / np.pi # tanInverse(r32,r33)
    pitch = np.arctan2(pose_els[5], pose_els[0]) * 180 / np.pi # tanInverse(r21,r11)

    if np.cos(pitch):
        yaw = np.arctan2(-1 * pose_els[8], pose_els[0]/np.cos(pitch)) * 180 / np.pi
    else:
        yaw = np.arctan2(-1 * pose_els[5], pose_els[0]/np.sin(pitch)) * 180 / np.pi

    heading_angle = 360 - yaw

    inProj = Proj('epsg:3857')
    outProj = Proj('epsg:4326')
    x1, y1 = pose_els[3] * 1000, pose_els[7] * 1000
    long_x, lat_y = transform(inProj, outProj, x1, y1)

    long_x, lat_y = localize(long_x, lat_y, loc)

    pre_pose = np.array(pose_els)
    pre_stamp = cur_tim
    pre_velo = velo

    return [ velo_mag_kmh, accn, heading_angle, lat_y, long_x, roll, pitch, yaw ]

def generate_obd_line(frame_num, obd_contents):

    fr_num = '{0:05}'.format(frame_num)

    x = str()
    x = x + '<frame id="' + fr_num + '" GPS_speed="' + str(round(obd_contents[0],2)) + '" OBD_speed="'   #/>
    x = x + str(round(obd_contents[0], 2)) + '" accX="' + str(round(obd_contents[1][0],3)) + '" accY="' + str(round(obd_contents[1][1],3))
    x = x + '" accZ="' + str(round(obd_contents[1][2],3)) + '" heading_angle="' + str(round(obd_contents[2],3)) + '" latitude="'
    x = x + str(round(obd_contents[3],6)) + '" longitude="' + str(round(obd_contents[4],6)) + '" roll="' + str(round(obd_contents[5],3)) 
    x = x + '" pitch="' + str(round(obd_contents[6],3)) + '" yaw="' + str(round(obd_contents[7],3)) + '" />'

    return x    #<frame id="0" GPS_speed="23.27" OBD_speed="21.63" accX="-0.03" accY="0.94" accZ="0.16"
                # heading_angle="342.99" latitude="43.6556265" longitude="-79.402457" roll="6.2" pitch="0.257" yaw="6.28" />

def stream_out(output_file, content):
    with open(output_file, 'wt') as fid:
        fid.write("%s" % (content))
        fid.close()

def refresh_path(n_path):
    if not os.path.exists(n_path):
        os.makedirs(n_path)

def extract_OBD(i, filename, vid_name, annt_dir, vv):
    obd_frames = []

    i_str = '{0:04}'.format(vv)
    FileName = filename
    dataset = tf.data.TFRecordDataset(FileName, compression_type='')

    frame_num = 0
    for data in dataset:

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        pose_elements = get_pose_elements(frame.pose)
        obds = get_OBD(pose_elements, frame.timestamp_micros, frame.context.stats.location)
        xml_frm = generate_obd_line(frame_num, obds)
        obd_frames.append(xml_frm)

        i += 1
        frame_num += 1

    serial_n = i_str
    fold_name = 'video_' + serial_n
    f_name = 'video_' + vid_name
    obd_path = annt_dir + fold_name + '/'
    refresh_path(obd_path)

    output_file = obd_path + f_name + '_obd.xml'
    str_f = str(obd_frames)
    str_f = str_f.replace(',', '')
    str_f = str_f.replace('[', '<?xml version="1.0" encoding="UTF-8"?><vehicle_info>')
    str_f = str_f.replace(']', '</vehicle_info>')
    str_f = str_f.replace("'", '')
    stream_out(output_file, str_f)

    return i