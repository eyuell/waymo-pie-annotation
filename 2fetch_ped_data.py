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
As Step 2
Read labels and images to return:
    annotations
    annotation_attributes
    pictures with pedestrians ids
"""

# Check on the availability of a font for pedestrian ID for frames


from os.path import join, abspath, isfile, isdir, exists
from os import listdir, makedirs
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
from collections import OrderedDict
from util_annt import get_header
import time
from datetime import timedelta
import pathlib

#Path where the input and output folders are available
main_path = '/home/eyuell/Desktop/ForD/WAYMO/output/'

dest_dir = main_path + 'annot_files/'
labels_dir = main_path + 'camera/labels/'
img_dir = main_path + 'camera/images/'
img_des = main_path + 'compiled_imgs/'

font_path = str(pathlib.Path().absolute()) + '/coolvetica.ttf'

# capture existing folder names
content_names = listdir(main_path)

# for collecting the attributes
attr_control = {}
compiled_attr = []
compiled_annt = {}
compiled_obd = []


def display_peds(peds, file_name, img_dir, img_des, vid_name):
    global font_path
    file_num = '{0:05}.png'.format(vid_name)

    image1 = Image.open(img_dir + file_num)
    font = ImageFont.truetype(font_path, 20)
    for k, v in peds.items():
        width, height = font.getsize(k)
        image2 = Image.new('RGBA', (width, height), (0, 0, 128, 92))
        draw2 = ImageDraw.Draw(image2)
        draw2.text((0,0), text=str(k), font=font, fill=(0, 255, 128))

        image2 = image2.rotate(90, expand=1)

        px, py , h = v
        h = int(h)
        px = int(px) - 10
        py = int(py) - h -20

        sx, sy = image2.size
        image1.paste(image2, (px, py, px + sx, py + sy), image2)

    image1.save(img_des + file_num)


def generate_xml_line(frame_num, xbr, xtl, ybr, ytl, ped_id, key_frame):
    x = str()

    if key_frame:
        k_frame = '1'
    else:
        k_frame = '0'

    fr_num = '{0:05}'.format(frame_num)

    x = x + '<box frame="' + fr_num + '" keyframe="' + k_frame + '" occluded="" outside="" '
    x = x + 'xtl="' + str(xtl) + '" ytl="' + str(ytl) + '" xbr="' + str(xbr) + '" ybr="' + str(ybr)
    x = x + '"> <attribute name="id">' + str(ped_id) + '</attribute> <attribute name="action">""</attribute> '
    x = x + '<attribute name="gesture">""</attribute> <attribute name="look">""</attribute>'
    x = x + '<attribute name="cross">""</attribute> <attribute name="occlusion">""</attribute> </box> '

    return x


def generate_obd_line(frame_num):
    x = str()
    x = x + '<frame id="' + str(frame_num) + '" <OBD_speed=""/>'

    return x

def generate_attr_line(ped_id):
    x = str()

    if isinstance(ped_id, int):
        ped_id = '"' + str(ped_id) + '"'

    x = x + '<pedestrian id="' + ped_id + '" age="adult" crossing="" crossing_point="" '
    x = x + 'critical_point="" exp_start_point="" intention_prob="" gender="" ' # designated=""
    x = x + 'intersection="" motion_direction="" num_lanes="" signalized="n/a" traffic_direction="TW" />'

    return x


def stream_out(output_file, content):
    with open(output_file, 'wt') as fid:
        fid.write("%s" % (content))
        fid.close()


def get_sorted_labels(label_names):
    frames = {}
    for label in label_names: # video_0001.xml
        ft = label.index('_')
        lt = label.index('.')
        vid_name = label[:ft]
        frame_num = label[ft+1:lt]
        frame_num = int(frame_num)
        frames[frame_num] = [label, vid_name]

    return OrderedDict(sorted(frames.items()))
    """
    for label in label_names: # video_0001_annt.xml
        spl_lab = label.split('_')
        if spl_lab:
            vid_name = spl_lab[0]
            frame_num = int(spl_lab[1])
            frames[frame_num] = [label, vid_name]

            return OrderedDict(sorted(frames.items()))
        else:
            return None
    """

def get_vid_names():
    if isdir(labels_dir):
        fold_names = listdir(labels_dir)
        return list(filter(lambda x: str(x).__contains__('video_') , fold_names))
    else:
        return []


def refresh_path(n_path):
    if not exists(n_path):
        makedirs(n_path)


def handel_attribute(vid, vid_name, compiled_attr):
    att_path = dest_dir + 'annotations_attributes/' + vid + '/'
    refresh_path(att_path)

    attr_file = att_path + 'video_' + vid_name + '_attributes.xml' #<ped_attributes>
    str_f = str(compiled_attr)
    str_f = str_f.replace('[', '<ped_attributes>')
    str_f = str_f.replace(']', '</ped_attributes>')
    str_f = str_f.replace(',', '')
    str_f = str_f.replace("'", '')
    stream_out(attr_file, str_f)


def handel_annot(compiled_annt, vid, vid_name, n_frames=198):
    updated_annot = []
    for k, v in compiled_annt.items():
        updated_annot.append(v)

    ann_path = dest_dir + 'annotations/' + vid + '/'
    refresh_path(ann_path)

    annt_file = ann_path + 'video_' + vid_name + '_annt.xml'
    str_f = str(updated_annot)
    str_f = str_f.replace(',', '')
    str_f = str_f.replace('[[', get_header(vid_name, vid, n_frames)) # check number of frames
    str_f = str_f.replace(']]', '</track></annotations>')
    str_f = str_f.replace(']', '</track>')
    str_f = str_f.replace('[', '<track label="pedestrian">')
    str_f = str_f.replace("'", '')
    stream_out(annt_file, str_f)

def generate_ped_data(cont, frame_num, frame_key):
    global compiled_attr, attr_control

    #ped_id = str(cont[-1]) # Full length id
    ped_id = str(cont[-1].split('-')[-1]) # Short length id
    len(ped_id)
    ped_id = ped_id[:len(ped_id) -2 ]
    ped_short = ped_id[len(ped_id)-4:]

    w = float(cont[3])
    h = float(cont[4])
    x = float(cont[1])
    y = float(cont[2]) - (h * 0.5)

    xbr = round(x + (w * 0.5), 2)
    xtl = round(x - (w * 0.5), 2)
    ybr = round(y + (h * 0.5), 2)
    ytl = round(y - (h * 0.5), 2)

    #frame_num = int(frame_num)

    xml_line = generate_xml_line(frame_num, xbr, xtl, ybr, ytl, ped_id, frame_key) 

    if ped_id in compiled_annt:
        compiled_annt[ped_id].append(xml_line)
    else:
        compiled_annt[ped_id] = [xml_line]

    yy = float(cont[2]) - (h * 0.5)

    if yy < 50:
        yy = 50 + yy

    peds[ped_short] = (x, yy, h)

    if not bool(attr_control.get(ped_id)):
        compiled_attr.append(generate_attr_line(ped_id))
        attr_control[ped_id] = True

    return peds

if __name__=="__main__":
    start = time.time()
    vid_list = get_vid_names()
    nn = 1
    for vid in vid_list:
        compiled_annt.clear()
        print("Working on segment", nn, "of", len(vid_list),"\r")
        nn += 1
        label_names = []
        cont = []
        peds = {}
        lab_path = labels_dir + vid + '/'

        label_names = listdir(lab_path)
        n_frames = len(label_names)

        if(n_frames):

            sorted_labels = get_sorted_labels(label_names)

            frame_code = -1
            for frame_num, labels in sorted_labels.items():
                #compiled_obd.append(generate_obd_line(frame_num))

                label = labels[0]
                vid_name = labels[1]
 
                label_file = join(lab_path, label)
                with open(label_file) as f:
                    lines = f.readlines()
                    for obj in lines:
                        cont = obj.split(',')
                        if cont[0] == "TYPE_PEDESTRIAN":
                            frame_key = False

                            if frame_code < 0:
                                frame_code = frame_num

                            if (frame_num - frame_code)%5 == 0:
                                frame_key = True

                            peds = generate_ped_data(cont, frame_num, frame_key)

                    f.close()
                if peds: # Generate images only if there is pedestrian in it
                    file_name = label[0:label.index('.')] + '.png'
                    img_path = img_dir + vid + '/'
                    refresh_path(img_path)

                    des_path = img_des + vid + '/'
                    refresh_path(des_path)

                    display_peds(peds, file_name, img_path, des_path, frame_num)

            handel_attribute(vid, vid_name, compiled_attr)
            compiled_attr = []

            handel_annot(compiled_annt, vid, vid_name, n_frames)
            compiled_annt.clear()

            print("Success on segment", nn - 1,'\n')

        else:
            print("\nThere is no content in labels directory\n")

    # Concluding
    end = time.time()
    elapsed = end - start
    t_del = timedelta(seconds=elapsed)
    print('Done! Duration= {}\n'.format(t_del))
