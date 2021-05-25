"""
Taken and Customized from:
            https://github.com/KushalBKusram/WaymoDataToolkit
"""
import os
import cv2
import glob
import pickle
import threading
import numpy as np
from urllib.parse import urlparse
import sys
import shutil
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from google.protobuf.json_format import MessageToDict

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from pprint import pprint

class ToolKit:

    def __init__(self, training_dir=None, testing_dir=None, validation_dir=None, save_dir=None):

        self.segment = None
        self.seg = None
        self.last_frame = None
        self.seg_number = 1
        self.training_dir = training_dir
        self.testing_dir = testing_dir
        self.validation_dir = validation_dir

        self.save_dir = save_dir

        self.camera_dir = self.save_dir + "/camera"
        self.camera_images_dir = self.camera_dir + "/images"
        self.camera_labels_dir = self.camera_dir + "/labels"
        if not os.path.exists(self.camera_dir):
            os.makedirs(self.camera_dir)
        if not os.path.exists(self.camera_images_dir):
            os.makedirs(self.camera_images_dir)
        if not os.path.exists(self.camera_labels_dir):
            os.makedirs(self.camera_labels_dir)

        #self.camera_list = ["UNKNOWN", "FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
        self.camera_list = ["FRONT"]    #Customized
        self.img_vid_path = ''
        self.lbl_vid_path = ''

    def assign_segment(self, segment, seg_num):
        self.segment = segment
        self.seg = segment.split('_')[0][8:]
        self.seg_number = seg_num
        self.dataset = tf.data.TFRecordDataset("{}/{}".format(self.training_dir, self.segment), compression_type='')

        vid_path1 = "{}/{}".format(self.camera_images_dir, self.get_vid_num())
        self.prepare_path(vid_path1)
        self.img_vid_path = vid_path1

        vid_path2 = "{}/{}".format(self.camera_labels_dir, self.get_vid_num())
        self.prepare_path(vid_path2)
        self.lbl_vid_path = vid_path2

    def list_training_segments(self):
        seg_list = []
        for file in os.listdir(self.training_dir):
            if file.endswith(".tfrecord"):
                seg_list.append(file)
        return seg_list

    def list_testing_segments(self):
        pass

    def list_validation_segments(self):
        pass

    def get_vid_num(self):
        num_part = "%04.f" % self.seg_number
        return 'video_' + num_part

    #########################################################################
    # Extract Camera Images and Labels
    #########################################################################

    # Extract Camera Image
    def extract_image(self, ndx, frame):
        for index, data in enumerate(frame.images): #"%05.f.png") % frame_num)
            if index == 0 and data.name == 1: #Customized (0 & 1) for FRONT
                decodedImage = tf.io.decode_jpeg(data.image, channels=3, dct_method='INTEGER_ACCURATE')
                decodedImage = cv2.cvtColor(decodedImage.numpy(), cv2.COLOR_RGB2BGR)
                #cv2.imwrite("{}/{}_{}.png".format(self.camera_images_dir, ndx, self.camera_list[data.name]), decodedImage)
                #cv2.imwrite("{}/{}_{}.png".format(self.camera_images_dir, self.seg, ndx), decodedImage)
                file_name = "%05.f.png" % ndx
                cv2.imwrite("{}/{}".format(self.img_vid_path, file_name), decodedImage)

    # Extract Camera Label
    def extract_labels(self, ndx, frame):
        for index, data in enumerate(frame.camera_labels):
            if index == 0 and data.name == 1:   #Customized
                camera = MessageToDict(data)
                camera_name = camera["name"]
                label_file = open("{}/{}_{}.txt".format(self.lbl_vid_path, self.seg, ndx), "w")
                try:
                    labels = camera["labels"]
                    for label in labels:
                        x = label["box"]["centerX"]
                        y = label["box"]["centerY"]
                        width = label["box"]["width"]
                        length = label["box"]["length"]
                        #x = x - 0.5 * length
                        #y = y - 0.5 * width
                        obj_type = label["type"]
                        obj_id = label["id"]
                        label_file.write("{},{},{},{},{},{}\n".format(obj_type, x, y, length, width, obj_id))
                        if self.last_frame != self.seg and obj_type == 'TYPE_PEDESTRIAN':
                            self.last_frame = self.seg
                            #print(self.seg)
                            open("{}/camera/ped_frames_file.txt".format(self.save_dir), 'a+').write("{}     \n".format(self.seg)).close() 

                except:
                    pass
                label_file.close()

    # Implemented Extraction as Threads
    def threaded_camera_image_extraction(self, datasetAsList, range_value):

        frame = open_dataset.Frame()

        for frameIdx in range_value:
            frame.ParseFromString(datasetAsList[frameIdx])
            self.extract_image(frameIdx, frame)
            self.extract_labels(frameIdx, frame)

    # Function to call to extract images
    def extract_camera_images(self):

        open("{}/camera/last_file.txt".format(self.save_dir), 'w+').write(self.segment) 

        # Convert tfrecord to a list
        datasetAsList = list(self.dataset.as_numpy_iterator())
        totalFrames = len(datasetAsList)

        threads = []

        for i in self.batch(range(totalFrames), 30):
            t = threading.Thread(target=self.threaded_camera_image_extraction, args=[datasetAsList, i])
            t.start()
            threads.append(t)

        for thread in threads:
            thread.join()

    #########################################################################
    # Consolidate Object Count per Camera and frontal_velocity, weather, time and location
    #########################################################################
    def consolidate(self):

        if not os.path.isdir("{}/consolidation".format(self.save_dir)):
            os.makedirs("{}/consolidation".format(self.save_dir))

        # Convert tfrecord to a list
        datasetAsList = list(self.dataset.as_numpy_iterator())
        totalFrames = len(datasetAsList)

        frame = open_dataset.Frame()

        stat_file = open("{}/consolidation/{}.csv".format(self.save_dir, self.segment[:-9]), "w")

        for frameIdx in range(totalFrames):

            frame.ParseFromString(datasetAsList[frameIdx])

            front_list = []
            front_left_list = []
            front_right_list = []
            side_left_list = []
            side_right_list = []

            for index, data in enumerate(frame.camera_labels):
                type_unknown = 0
                type_vehicle = 0
                type_ped = 0
                type_sign = 0
                type_cyclist = 0
                camera = MessageToDict(data)
                camera_name = camera["name"]
                try:
                    labels = camera["labels"]
                except:
                    labels = None
                if labels is not None:
                    for label in labels:
                        if label["type"] == "TYPE_UNKNOWN":
                            type_unknown += 1
                        elif label["type"] == "TYPE_VEHICLE":
                            type_vehicle += 1
                        elif label["type"] == "TYPE_PEDESTRIAN":
                            type_ped += 1
                        elif label["type"] == "TYPE_SIGN":
                            type_sign += 1
                        elif label["type"] == 'TYPE_CYCLIST':
                            type_cyclist += 1
                    if camera_name == "FRONT":
                        front_list = [type_unknown, type_vehicle, type_ped, type_sign, type_cyclist]
                    elif camera_name == "FRONT_LEFT":
                        front_left_list = [type_unknown, type_vehicle, type_ped, type_sign, type_cyclist]
                    elif camera_name == "FRONT_RIGHT":
                        front_right_list = [type_unknown, type_vehicle, type_ped, type_sign, type_cyclist]
                    elif camera_name == "SIDE_LEFT":
                        side_left_list = [type_unknown, type_vehicle, type_ped, type_sign, type_cyclist]
                    elif camera_name == "SIDE_RIGHT":
                        side_right_list = [type_unknown, type_vehicle, type_ped, type_sign, type_cyclist]
                else:
                    if camera_name == "FRONT":
                        front_list = [0, 0, 0, 0, 0]
                    elif camera_name == "FRONT_LEFT":
                        front_left_list = [0, 0, 0, 0, 0]
                    elif camera_name == "FRONT_RIGHT":
                        front_right_list = [0, 0, 0, 0, 0]
                    elif camera_name == "SIDE_LEFT":
                        side_left_list = [0, 0, 0, 0, 0]
                    elif camera_name == "SIDE_RIGHT":
                        side_right_list = [0, 0, 0, 0, 0]
            obj_list = front_list + front_left_list + front_right_list + side_left_list + side_right_list
            # determine the velocity
            velocity = MessageToDict(frame.images[0])
            stat_file.write("{},{},{},{},{}\n".format(','.join([str(obj_count) for obj_count in obj_list]), ','.join([str(vel) for vel in velocity["velocity"].values()]), frame.context.stats.weather, frame.context.stats.time_of_day, frame.context.stats.location))

    #########################################################################
    # Util Functions
    #########################################################################

    def delete_files(self, files):
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def refresh_dir(self, dir):
        if os.path.isdir(dir):
            sys.stdout.write('Preparing Files . . .  \r')
            try:
                shutil.rmtree(dir)
                os.makedirs(dir, exist_ok=True)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

    def prepare_path(self, f_path):
        if not os.path.exists(f_path):
            os.makedirs(f_path)
