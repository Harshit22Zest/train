import signal
import sys
import os
import time

import MySQLdb
import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import constants
import zipfile
from concurrent.futures import ThreadPoolExecutor
from azure.storage.blob import ContainerClient

from datetime import datetime
import json
import shutil
import subprocess
import darknet
import requests
from subprocess import Popen
from utility import Utility


class AzureStorage:

    def __init__(self, connect):
        self.connection_string = connect.config_json["azure_storage_connection_string"]
        self.container_name = connect.config_json["container_name_video"]
        self.blob_name = connect.config_json["blob_name"]

    def upload(self, file_to_upload, _upload_location, container_name=None):
        try:
            if not container_name:
                container_name = self.container_name
            container_client = ContainerClient.from_connection_string(self.connection_string, container_name)
            print("uploading in progress ...")
            blob_client = container_client.get_blob_client(_upload_location)
            with open(file_to_upload, 'rb') as data:
                blob_client.upload_blob(data)
                print("Uploaded {} to container {} ...".format(file_to_upload, container_name))
                # os.remove(file_to_upload)
                # print("Successfully removed file {}".format(file_to_upload))
            return 1
        except Exception as e_:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("exception in uploading the file  " + str(e_) + ' ' + str(exc_tb.tb_lineno))
            return 0

    def remove_blob(self, _remove_location):
        try:
            container_client = ContainerClient.from_connection_string(self.connection_string, self.container_name)
            print("deleting in progress ...")
            blob_client = container_client.get_blob_client(_remove_location)
            blob_client.delete_blob()
            return 1
        except Exception as e_:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("exception in removing the file  " + str(e_) + ' ' + str(exc_tb.tb_lineno))
            return 0

    def download(self, file_to_download, download_location, container_name=None):
        try:
            if not container_name:
                container_name = self.container_name
            container_client = ContainerClient.from_connection_string(self.connection_string, container_name)
            print("downloading in progress ...")
            blob_client = container_client.get_blob_client(file_to_download)
            with open(download_location, "wb+") as data:
                blob_data = blob_client.download_blob()
                blob_data.readinto(data)
            print("download completed at {}".format(file_to_download))
        except Exception as e_:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("exception in downloading the file  " + str(e_) + ' ' + str(exc_tb.tb_lineno))


class Compressor:
    @classmethod
    def zip(cls, image_directory):
        def zipdir(path, ziph):
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file),
                               os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

        zipf = zipfile.ZipFile(image_directory + '.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir(image_directory, zipf)
        zipf.close()
        return image_directory + '.zip'

    @classmethod
    def unzip(cls, zip_directory):
        zipf = zipfile.ZipFile(zip_directory)
        zipf.extractall(os.path.split(zip_directory)[0])


connect = Utility(configuration=constants.platform_utility_configuration, rabbitmq_consumer=constants.rabbitmq_consumer, rabbitmq_publisher=constants.train_api_publisher)
executor = ThreadPoolExecutor(max_workers=constants.max_workers)
azure_storage = AzureStorage(connect=connect)


def get_annotations(blob_annotation_end_point, blob_image_end_point):
    print("get annotation is caled")
    try:
        local_annotation_zip_end_point = os.path.join(connect.config_json["gpu_video_location"], blob_annotation_end_point)
        print("local annotation zip end point {}, exists = {}".format(local_annotation_zip_end_point, os.path.exists(local_annotation_zip_end_point)))
        local_image_zip_end_point = os.path.join(connect.config_json["gpu_video_location"], blob_image_end_point)
        print("local image zip endpoint = {} exists = {}".format(local_image_zip_end_point, os.path.exists(local_image_zip_end_point)))
        local_annotation_unzip_location = os.path.splitext(str(local_annotation_zip_end_point))[0]
        print("local annotation unzip location {} exists = {}".format(local_annotation_unzip_location, os.path.exists(local_annotation_unzip_location)))
        local_image_unzip_location = os.path.splitext(str(local_image_zip_end_point))[0]
        print("local image unzip location {} exists = {} ".format(local_image_unzip_location, os.path.exists(local_image_unzip_location)))
        if os.path.exists(local_annotation_unzip_location) and os.path.exists(local_image_unzip_location):
            print("going to if condition exists in both")
            return local_annotation_unzip_location, local_image_unzip_location
        else:
            print("going to else condition not exists in both")
            if not os.path.exists(os.path.dirname(local_annotation_unzip_location)):
                print("creating the directory ={}".format(local_annotation_unzip_location))
                os.makedirs(os.path.dirname(local_annotation_unzip_location))
                print("created")

            if not os.path.exists(os.path.dirname(local_image_unzip_location)):
                print("creating the directory = {}".format(local_image_unzip_location))
                os.makedirs(os.path.dirname(local_image_unzip_location))
                print("created")

            print("downloading the blob_annotation_end_point = {} from = {} ".format(blob_annotation_end_point, local_annotation_zip_end_point))
            azure_storage.download(blob_annotation_end_point, local_annotation_zip_end_point)
            print('downloaded')
            print("downloading the blob image end point = {} from = {} ".format(blob_image_end_point, local_image_zip_end_point))
            azure_storage.download(blob_image_end_point, local_image_zip_end_point)
            print('downloaded')

            print('unzipping the  {}'.format(local_annotation_zip_end_point))
            Compressor.unzip(local_annotation_zip_end_point)
            print('unzipped')
            print('unzipping the  {}'.format(local_image_zip_end_point))
            Compressor.unzip(local_image_zip_end_point)
            print('unzipped')

        return local_annotation_unzip_location, local_image_unzip_location
    except Exception as e_:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Exception occurred in train : " + str(e_) + ' ' + str(exc_tb.tb_lineno))


def config_generator(config, model_name, obj_file, model_type, total_classes):
    print('config generator is called -> config = {} model_name = {} obj_file = {} model_type = {}'.format(config, model_name, obj_file, model_type))
    try:
        data = {"height": str(config.get('model_height')),
                "width": str(config.get('model_width')),
                "batch": str(config.get('batch')),
                "subdivision": str(config.get('subdivision')),
                "iterations": str(config.get('iterations')),
                "model": model_type,  # "yolov4-tiny",
                "model_name": model_name,
                "data_path": obj_file,  # os.path.join(train_test_artifacts_end_point, "obj.data")
                "sample": os.path.join(os.path.split(obj_file)[0], "sample.cfg"),
                "new_model_path": os.path.split(obj_file)[0],  # train_test_artifacts_end_point
                "darknet": os.path.join(connect.config_json["working_directory_train"], "darknet")}

        print('data i get {}'.format(data))
        sample = data["sample"]
        if not os.path.exists(sample):
            print("downloading  in {} from {}".format(sample, os.path.join(connect.config_json.get("sample_cfg"), os.path.split(sample)[1])))
            azure_storage.download(os.path.join(connect.config_json.get("sample_cfg"), os.path.split(sample)[1]), sample, container_name=connect.config_json["model_container"])

        model_cfg = os.path.join(os.path.split(obj_file)[0], data["model"] + data["model_name"] + ".cfg")
        print('the cfg will be stored at {}'.format(model_cfg))
        print('starting the anchor command bash train.sh {} {} 6 {} {}'.format(data["darknet"], data["data_path"], data["width"], data["height"]))
        anchor_command = Popen(['bash', 'train.sh', data["darknet"], data["data_path"], '6', data["width"], data["height"]], stdout=subprocess.PIPE)
        print('starting 5 sec timer')
        time.sleep(5)
        print('time end')
        class_num = int(total_classes)
        print('stopping the anchor command')
        os.kill(anchor_command.pid, signal.SIGTERM)
        print('stopped')
        if os.path.exists(data["darknet"] + "/anchors.txt"):
            print('anchors txt file is there')
            with open(data["darknet"] + "/anchors.txt", 'r+') as f:
                anchors = f.readline()
            print(anchors)
            print('starting creating and writing on the cfg file')
            with open(model_cfg, 'w+') as out:
                print('starting reading the sample file')
                with open(data["sample"], 'r+') as sample:
                    for j, i in enumerate(sample.readlines()):
                        print(i, j)
                        if i.split("=")[0] == "batch":
                            out.write("batch=" + data["batch"] + "\n")
                        elif i.split("=")[0] == "subdivision":
                            out.write("subdivision=" + data["subdivision"] + "\n")
                        elif i.split("=")[0] == "width":
                            out.write("width=" + data["width"] + "\n")
                        elif i.split("=")[0] == "height":
                            out.write("height=" + data["height"] + "\n")
                        elif i.split("=")[0] == "max_batches":
                            out.write("max_batches=" + data["iterations"] + "\n")
                        elif i.split("=")[0] == "steps":
                            out.write("steps=" + str(int((int(data["iterations"]) / 10) * 8)) + "," + str(int((int(data["iterations"]) / 10) * 8)) + "\n")
                        elif i.split("=")[0] == "filters" and (i.split("=")[1]).strip() == '30':
                            print(i)
                            out.write("filters=" + str(int(class_num + 5) * 3) + "\n")
                        elif i.split("=")[0] == "anchors":
                            out.write("anchors=" + str(anchors) + "\n")
                        elif i.split("=")[0] == "classes":
                            out.write("classes=" + str(class_num) + "\n")
                        else:
                            out.write(i)
                print('closing the sample file')
                sample.close()
            print('created  cfg file')

            out.close()
            if os.path.exists(model_cfg):
                print('cfg file is there {}'.format(model_cfg))
                return model_cfg
            else:
                print('cfg file is not there = {}'.format(model_cfg))
                sys.exit()
        else:
            print('anchors txt file is not there')
            sys.exit()

    # with open
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Exception in generating the config file : " + str(e_) + ' ' + str(exc_tb.tb_lineno))


def MAP(data_file, name_file, cfg_file, weight_file, model_name, model_id):
    print('MAP is called')
    try:
        print('model_name = {} ,obj_data_file = {},  model_cfg ={}, obj_name_file = {}, obj_weights_file = {} ,model_id={}'.format(model_name, data_file, cfg_file, name_file, weight_file, model_id))
        data = {
            "obj_path": data_file,  # "/home/cvuser/HPCL_files/20_class_files/obj_20_class.data",
            "cfg_path": cfg_file,  # "/home/cvuser/HPCL_files/20_class_files/yolov4-tiny_20_class.cfg",
            "weight_path": weight_file,  # "/home/cvuser/HPCL_files/20_class_files/backup/yolov4-tiny_20_class_best.weights",
            "darknet": os.path.join(connect.config_json["working_directory_train"], "darknet")}
        # result = subprocess.run(['bash','calc_map.sh',data["obj_path"],data["cfg_path"],data["weight_path"]], stdout=subprocess.PIPE)
        # result.stdout
        print("writing the command = bash calc_map.sh {} {} {} {}".format(data["darknet"], data["obj_path"], data["cfg_path"], data["weight_path"]))
        map_command = Popen(['bash', 'calc_map.sh', data["darknet"], data["obj_path"], data["cfg_path"], data["weight_path"]], stdout=subprocess.PIPE)
        print('command run success')
        out, err = map_command.communicate()
        print('err = {}'.format(err))
        print("out = {}".format(out))
        print("after decoding")
        out = ((out.decode()).split("mAP@0.50")[1]).split("%")[0][-6:-1]
        print("out = {}".format(out))
        sql = '''update  m_model set Map = "{}" where id = "{}"'''.format(str(out), model_id)
        rows_effected = connect.update_database(sql=sql)
    except Exception as e_:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Exception in MAP : " + str(e_) + ' ' + str(exc_tb.tb_lineno))


def train(body):
    start = time.time()
    start_datetime = datetime.now()
    print('train function called, data got  = {}'.format(body))
    try:
        end_points = body.get("end_points")
        tags_mapper = {}
        next_class_index, next_image_count = 0, 0
        train_test_end_point = os.path.join(connect.config_json["gpu_video_location"], "{}_training_{}".format(body.get("model_name"), str(int(datetime.timestamp(datetime.now().replace(microsecond=0))))))
        print('train_test_end_point = {}, exists = {}'.format(train_test_end_point, os.path.exists(train_test_end_point)))
        train_test_artifacts_end_point = os.path.join(train_test_end_point, 'artifacts')
        print('train_test_artifacts_end_point = {} exists = {}'.format(train_test_artifacts_end_point, os.path.exists(train_test_artifacts_end_point)))
        if not os.path.exists(train_test_artifacts_end_point):
            print('creating the directory = {}'.format(train_test_artifacts_end_point))
            os.makedirs(train_test_artifacts_end_point)
        if not os.path.exists(train_test_end_point):
            print('creating the directory = {}'.format(train_test_end_point))
            os.makedirs(train_test_end_point)

        test_file_path = os.path.join(train_test_artifacts_end_point, 'test.txt')
        print('opening the files in write mode ')
        print('test file path = {} exists = {}'.format(test_file_path, os.path.exists(test_file_path)))
        train_file_path = os.path.join(train_test_artifacts_end_point, 'train.txt')
        print('train file path = {} exists = {}'.format(train_file_path, os.path.exists(train_file_path)))

        for end_point in end_points:
            print('endpoint = {}'.format(end_point))
            local_annotation_end_point, local_image_end_point = get_annotations(blob_annotation_end_point=end_point.get("blob_annotation_end_point"), blob_image_end_point=end_point.get("blob_image_end_point"))
            print('local_annotation_end_point = {},it contains ={}'.format(local_annotation_end_point, os.listdir(local_annotation_end_point)))
            print('local_image_end_point = {}, it contains ={}'.format(local_image_end_point, os.listdir(local_image_end_point)))

            for directory in os.listdir(local_annotation_end_point):
                annotation_duration_end_point = os.path.join(local_annotation_end_point, directory)
                image_duration_end_point = os.path.join(local_image_end_point, directory)
                index_mapper = {}
                print('annotation_duration_end_point = {},it contains ={}'.format(annotation_duration_end_point, os.listdir(annotation_duration_end_point)))
                print('image_duration_end_point = {}, it contains ={}'.format(image_duration_end_point, os.listdir(image_duration_end_point)))
                print('open the {} file in read mode '.format(os.path.join(annotation_duration_end_point, "classes.txt")))
                with open(os.path.join(annotation_duration_end_point, "classes.txt")) as classes_file:
                    tags_list = classes_file.readlines()
                    print('classes are ={}'.format(tags_list))
                    for i in range(len(tags_list)):
                        tags_list[i] = (tags_list[i].replace("\n", " ")).strip()

                    for tag_name in tags_list:
                        tag_name = (tag_name.replace("\n", " ")).strip()
                        if tag_name not in tags_mapper:
                            tags_mapper[tag_name] = next_class_index
                            next_class_index = next_class_index + 1
                    for tag_index, tag_name in enumerate(tags_list):
                        index_mapper[tag_index] = tags_mapper[tag_name]
                    # next_class_index = next_class_index + len(tags_list)

                print('tag_mapper = {},  index_mapper + {}, next_class_index = {}'.format(tags_mapper, index_mapper, next_class_index))
                for _yolo_file in os.listdir(annotation_duration_end_point):
                    yolo_file = os.path.join(annotation_duration_end_point, _yolo_file)
                    if "classes.txt" not in yolo_file:
                        print('yolofile = {}'.format(yolo_file))
                        image_source_file = os.path.join(image_duration_end_point, os.path.split(yolo_file)[1].replace(".txt", ".jpg"))
                        print('image source  file = {}'.format(image_source_file))
                        image_destination_file = os.path.join(train_test_end_point, '{}_{}'.format(next_image_count, os.path.split(image_source_file)[1]))
                        print('image dest file = {}'.format(image_destination_file))
                        print("copying the file from = {} to = {}".format(image_source_file, image_destination_file))
                        os.popen('cp {} {}'.format(image_source_file.replace(" ", "\\ "), image_destination_file))
                        time.sleep(2)
                        print('copied')
                        print("opening the file {} in write mode and file = {} in read mode".format(os.path.join(train_test_end_point, "{}_{}".format(next_image_count, os.path.split(yolo_file)[1])), yolo_file))
                        with open(os.path.join(train_test_end_point, "{}_{}".format(next_image_count, os.path.split(yolo_file)[1])), 'w') as annotating_file, open(yolo_file) as annotated_file:
                            yolo_string = annotated_file.read()
                            print('yolo string = {}'.format(yolo_string))
                            for index in sorted(index_mapper, reverse=True):
                                if str(index) + ' ' in yolo_string:
                                    print('yolo string = {}'.format(yolo_string))
                                    yolo_string = yolo_string.replace(str(index) + ' ', str(index_mapper[index]) + ' ')
                            print('writing on {} the {}'.format(os.path.join(train_test_end_point, "{}_{}".format(next_image_count, os.path.split(yolo_file)[1])), yolo_string))
                            annotating_file.write(yolo_string)
                            next_image_count = next_image_count + 1
                            print('next image count = {}'.format(next_image_count))

        print('writing on file ={}'.format(os.path.join(train_test_end_point, "classes.txt")))
        with open(os.path.join(train_test_end_point, "classes.txt"), 'w') as classes_file:
            classes_string = '\n'.join(sorted(tags_mapper, key=tags_mapper.get))
            print('classes string = {}'.format(classes_string))
            classes_file.write(classes_string)

        test_file_path = os.path.join(train_test_artifacts_end_point, 'test.txt')
        print('opening the files in write mode ')
        print('test file path = {} exists = {}'.format(test_file_path, os.path.exists(test_file_path)))
        train_file_path = os.path.join(train_test_artifacts_end_point, 'train.txt')
        print('train file path = {} exists = {}'.format(train_file_path, os.path.exists(train_file_path)))

        with open(test_file_path, 'w') as file_test, open(train_file_path, 'w') as file_train:
            for i, j in enumerate(os.listdir(train_test_end_point)):
                print(i, j)
                if j[-4:] != '.jpg' or 'classes' in j or 'artifacts' in j:
                    continue
                if i % 5 == 0:
                    print(i, j, "train.txt")
                    file_test.write(os.path.join(train_test_end_point, j))
                    file_test.write('\n')
                else:
                    print(i, j, "test.txt")
                    file_train.write(os.path.join(train_test_end_point, j))
                    file_train.write('\n')
        print('opening the files in write mode ={} '.format(os.path.join(train_test_artifacts_end_point, "obj.names")))

        with open(os.path.join(train_test_artifacts_end_point, "obj.names"), 'w') as obj_names:
            classes_string = '\n'.join(sorted(tags_mapper, key=tags_mapper.get))
            print('writing the class string ={}'.format(classes_string))
            obj_names.write(classes_string)

        print('backup  = {} exists = {}'.format(os.path.join(train_test_artifacts_end_point, "backup"), os.path.exists(os.path.join(train_test_artifacts_end_point, "backup"))))
        if not os.path.exists(os.path.join(train_test_artifacts_end_point, "backup")):
            print('creating the directory = {}'.format(os.path.join(train_test_artifacts_end_point, "backup")))
            os.makedirs(os.path.join(train_test_artifacts_end_point, "backup"))

        print('opening the files in write mode ={} '.format(os.path.join(train_test_artifacts_end_point, "obj.data")))
        with open(os.path.join(train_test_artifacts_end_point, "obj.data"), 'w') as obj_data:
            obj_data.write("class = {}\n".format(len(tags_mapper)))
            obj_data.write("train = {}\n".format(train_file_path))
            obj_data.write("valid = {}\n".format(test_file_path))
            obj_data.write("names = {}\n".format(os.path.join(train_test_artifacts_end_point, "obj.names")))
            obj_data.write("backup = {}\n".format(os.path.join(train_test_artifacts_end_point, "backup")))

        obj_names_file = os.path.join(train_test_artifacts_end_point, "obj.names")
        obj_data_file = os.path.join(train_test_artifacts_end_point, "obj.data")
        # azure_storage.download(body.get("config_end_point"), os.path.join(train_test_artifacts_end_point, os.path.split(body.get("config_end_point"))[1]), container_name=connect.config_json.get("model_container"))

        if not os.path.exists(os.path.join(train_test_artifacts_end_point, os.path.split(body.get("conv_end_point"))[1])):
            azure_storage.download(body.get("conv_end_point"), os.path.join(train_test_artifacts_end_point, os.path.split(body.get("conv_end_point"))[1]), container_name=connect.config_json["model_container"])

        model_cfg = config_generator(body.get('config'), body.get('model_name'), obj_data_file, body.get("model_type"), body.get('total_classes'))
        # add a condition if conv endpoint file is not in server then download it otherwise not
        with open(os.path.join(train_test_artifacts_end_point, "input.txt"), "w+") as input_data:
            # input_data.write("image_end_point = {}, exits = {} \n".format(local_image_end_point, os.path.exists(local_image_end_point)))
            # input_data.write("annotation_end_point = {} , exits = {} \n".format(local_annotation_end_point, os.path.exists(local_annotation_end_point)))
            input_data.write("names file endpoint = {} , exits = {}\n".format(os.path.join(train_test_artifacts_end_point, "obj.names"), os.path.exists(os.path.join(train_test_artifacts_end_point, "obj.names"))))
            input_data.write("data file endpoint = {}, exits = {}\n".format(os.path.join(train_test_artifacts_end_point, "obj.data"), os.path.exists(os.path.join(train_test_artifacts_end_point, "obj.data"))))
            input_data.write("conv file endpoint = {}, exits = {} \n".format(os.path.join(train_test_artifacts_end_point, os.path.split(body.get("conv_end_point"))[1]), os.path.exists(os.path.join(train_test_artifacts_end_point, os.path.split(body.get("conv_end_point"))[1]))))
            # input_data.write("sample cfg file endpoint = {}, exits = {}\n".format(os.path.join(train_test_artifacts_end_point,"sample.cfg"),  os.path.exists(os.path.join(train_test_artifacts_end_point,"sample.cfg"))))
            input_data.write("new created cfg file endpoint = {}, exits = {} \n".format(model_cfg, os.path.exists(model_cfg)))

        try:
            print("Training started")
            os.chdir(os.path.join(connect.config_json["working_directory_train"], "darknet"))
            print(" ".join(["./darknet", "detector", "train", os.path.join(train_test_artifacts_end_point, "obj.data"), os.path.join(train_test_artifacts_end_point, os.path.split(model_cfg)[1]), os.path.join(train_test_artifacts_end_point, os.path.split(body.get("conv_end_point"))[1]), "-map", "-dont_show"]))
            subprocess.run(["./darknet", "detector", "train", os.path.join(train_test_artifacts_end_point, "obj.data"), os.path.join(train_test_artifacts_end_point, os.path.split(model_cfg)[1]), os.path.join(train_test_artifacts_end_point, os.path.split(body.get("conv_end_point"))[1]), "-map", "-dont_show"])
            print("Training completed")
        except Exception as e_:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Exception in training : " + str(e_) + ' ' + str(exc_tb.tb_lineno))

        print('uploading the file from ={} to = {}'.format(os.path.join(train_test_artifacts_end_point, "obj.data"), os.path.join(connect.config_json["deployment_models"], body.get("model_name") + "/obj.data")))
        azure_storage.upload(os.path.join(train_test_artifacts_end_point, "obj.data"), os.path.join(connect.config_json["deployment_models"], body.get("model_name") + "/obj.data"), container_name=connect.config_json["model_container"])
        print('uploaded')

        print('uploading the file from ={} to = {}'.format(os.path.join(train_test_artifacts_end_point, "obj.names"), os.path.join(connect.config_json["deployment_models"], body.get("model_name") + "/obj.names")))
        azure_storage.upload(os.path.join(train_test_artifacts_end_point, "obj.names"), os.path.join(connect.config_json["deployment_models"], body.get("model_name") + "/obj.names"), container_name=connect.config_json["model_container"])
        print('uploaded')

        print('uploading the file from ={} to = {}'.format(os.path.join(train_test_artifacts_end_point, "input.txt"), os.path.join(connect.config_json["deployment_models"], body.get("model_name") + "/input.txt")))
        azure_storage.upload(os.path.join(train_test_artifacts_end_point, "input.txt"), os.path.join(connect.config_json["deployment_models"], body.get("model_name") + "/input.txt"), container_name=connect.config_json["model_container"])
        print('uploaded')

        local_config_path = model_cfg
        print('uploading the file from ={} to = {}'.format(local_config_path, os.path.join(os.path.join(connect.config_json["deployment_models"], body.get("model_name")), os.path.split(local_config_path)[1])))
        azure_storage.upload(local_config_path, os.path.join(os.path.join(connect.config_json["deployment_models"], body.get("model_name")), os.path.split(local_config_path)[1]), container_name=connect.config_json["model_container"])
        print('uploaded')

        print('backup = {}, it contains = {}'.format(os.path.join(train_test_artifacts_end_point, "backup"), os.listdir(os.path.join(train_test_artifacts_end_point, "backup"))))
        model_file = [f for f in os.listdir(os.path.join(train_test_artifacts_end_point, "backup")) if os.path.isfile(os.path.join(os.path.join(train_test_artifacts_end_point, "backup"), f))]
        weight_file = os.path.join(connect.config_json["deployment_models"], os.path.join(body.get("model_name"), model_file[0]))
        print('uploading the file from ={} to = {}'.format(os.path.join(os.path.join(train_test_artifacts_end_point, "backup"), model_file[0]), os.path.join(connect.config_json["deployment_models"], os.path.join(body.get("model_name"), model_file[0]))))
        azure_storage.upload(os.path.join(os.path.join(train_test_artifacts_end_point, "backup"), model_file[0]), os.path.join(connect.config_json["deployment_models"], os.path.join(body.get("model_name"), model_file[0])), container_name=connect.config_json["model_container"])
        print('uploaded')

        # local_config_path = os.path.join(train_test_artifacts_end_point, os.path.split(body.get("config_end_point"))[1])

        # clean_data(directories=[train_test_artifacts_end_point, local_annotation_end_point, local_image_end_point])

        blob_locations = [os.path.join(connect.config_json["deployment_models"], body.get("model_name") + "/obj.data"), os.path.join(connect.config_json["deployment_models"], body.get("model_name") + "/obj.names"), os.path.join(connect.config_json["deployment_models"], os.path.join(body.get("model_name"), model_file[0])), os.path.join(os.path.join(connect.config_json["deployment_models"], body.get("model_name")), os.path.split(local_config_path)[1])]
        # sql = '''update m_model set end_point = "{}" where name = "{}"'''.format(str(blob_locations), body.get("model_name"))
        # print('updating/ inserting  the table model endpoint = {} where name  = {}/ inserting the name and endpoint '.format(str(blob_locations), body.get("model_name")))
        # rows_effected = connect.update_database(sql=sql)
        #
        # if not rows_effected:
        sql = '''insert into m_model (name, end_point, model_type_id, location_id) values ("{}", "{}","{}","{}")'''.format(body.get("model_name"), str(blob_locations), body.get("model_type_id"), body.get("location_id"))
        model_id = connect.update_database_and_return_id(sql=sql)
        print('query done')
        print('calling the link = {}'.format(os.path.join(connect.config_json.get("app_server_host") + "download_model")))
        # requests.post(os.path.join(connect.config_json.get("app_server_host") + "download_model"), headers={"content-type": "text", "entity-location": json.dumps(connect.config_json.get("entity-location"))})
        print('called')
        end_datetime = datetime.now()
        end = time.time()
        task_result = {"user_id": str(body.get('user_id')), "name": "train done successfully for model name  {}".format(body.get('model_name')), "status": "Success", "time_taken": str(end - start), "start_datetime": str(start_datetime), "end_datetime": str(end_datetime)}
        task_result = json.dumps(task_result)
        check_task_status(task_result)

        print('calculating access precision - Map')

        MAP(data_file=obj_data_file, name_file=obj_names_file, cfg_file=model_cfg, weight_file=weight_file, model_name=body.get("model_name"), model_id=model_id)

    except Exception as e_:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Exception occurred in train : " + str(e_) + ' ' + str(exc_tb.tb_lineno))
        end_datetime = datetime.now()
        end = time.time()
        task_result = {"user_id": str(body.get('user_id')), "name": "Exception while training  at {}".format(str(exc_tb.tb_lineno)), "status": "Failed", "time_taken": str(end - start), "start_datetime": str(start_datetime), "end_datetime": str(end_datetime)}
        task_result = json.dumps(task_result)
        check_task_status(task_result)


def check_task_status(status):
    print('checking task status')
    try:
        status = json.loads(status)
        user_id = status.get('user_id')
        if user_id is None:
            user_id = "#"
        print("user-id", user_id)
        print("%s", status)
        status = json.dumps(status)
        if 'Success' in status:
            # connect.master_redis.r.delete(key)
            connect.publish(event=status, routing_key="task.{}".format(user_id))
            print("published task success")
        elif 'Failed' in status:
            # connect.master_redis.r.delete(key)
            connect.publish(event=status, routing_key="task.{}".format(user_id))
            print("published task failed")
        time.sleep(30)
    except Exception as _e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("check_task_status thread : exception in check_task_status " + str(_e) + ' ' + str(exc_tb.tb_lineno))
        time.sleep(60)


def clean_data(directories):
    print('calling the clean data function with dir ={}'.format(directories))
    for directory in directories:
        shutil.rmtree(directory)
        print("removed {}".format(directory))


# gpu location is the dir where the models folders are stored , working directory train is the dir where the darknet folder exists, deployments in the blob storage where the models that we are training are stored there
# two files that is required , one is the train.sh which should be with the train python script and one is calc_map.sh that should be in the darknet folder(train.sh - to start training , calc_map.sh -> to run the calculation of MAP)

def initialize_directoy():
    for directory in [connect.config_json.get("gpu_video_location"), connect.config_json.get("working_directory_train")]:
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)


if __name__ == '__main__':
    initialize_directoy()
    print("azure_storage_connection_string = {}".format(connect.config_json.get("azure_storage_connection_string")))

    print("container_name_video = {}".format(connect.config_json.get("container_name_video")))

    print("blob_name = {}".format(connect.config_json.get("blob_name")))

    print("gpu_video_location = {}".format(connect.config_json.get("gpu_video_location")))

    print("model_container = {}".format(connect.config_json.get("model_container")))

    print("working_directory_train = {}".format(connect.config_json.get("working_directory_train")))

    print("deployment_models = {}".format(connect.config_json.get("deployment_models")))

    print("app_server_host = {}".format(connect.config_json.get("app_server_host")))

    print("darknet directory  exists = {} on location = {}".format(os.path.exists(os.path.join(connect.config_json.get("working_directory_train"), "darknet")), os.path.join(connect.config_json.get("working_directory_train"), "darknet")))

    try:
        class_mediaduration = {}
        mediaduration_mediaid = {}
        class_mapper = {}
        tags = {}
        model_type = {}
        entity_mapper = {}
        location_mapper = {}
        table_data_4 = connect.query_database(sql='SELECT id, name FROM m_entity')
        table_data_5 = connect.query_database(sql='SELECT id, name FROM m_location')

        for row in table_data_4[0]:
            entity_mapper[str(row[0])] = str(row[1])
            entity_mapper[str(row[1])] = str(row[0])

        for row in table_data_5[0]:
            location_mapper[str(row[0])] = str(row[1])
            location_mapper[str(row[1])] = str(row[0])

        entity = list(connect.config_json.get('entity-location').keys())[0]
        location = connect.config_json['entity-location'][entity]['locations'][0]

        table_data = connect.query_database(sql='SELECT id, name FROM m_tags where entity_id ={}'.format(entity_mapper.get(entity)))
        table_data_1 = connect.query_database(sql='SELECT id, name, conv_end_point FROM m_model_type')
        table_data_2 = connect.query_database(sql='SELECT tag_id, media_duration_id FROM media_tags_mapping')
        table_data_3 = connect.query_database(sql='SELECT id, media_status_id FROM t_media_durations')

        i = 0
        for row in table_data[0]:
            class_mapper[str(row[0])] = str(row[1])
            class_mapper[str(row[1])] = str(row[0])
            tags[str(i)] = str(row[1])
            i = i + 1

        i = 0
        for row in table_data_1[0]:
            model_type[str(i)] = {"id": str(row[0]), "model_type": str(row[1]), "conv_end_point": str(row[2])}
            i = i + 1

        for row in table_data_2[0]:
            if str(row[0]) not in class_mediaduration:
                class_mediaduration[str(row[0])] = []

            class_mediaduration[str(row[0])].append(str(row[1]))

        for row in table_data_3[0]:
            mediaduration_mediaid[str(row[0])] = str(row[1])

        ## taking inputs from the user #####################################################33

        for key, value in tags.items():
            print(str(key) + ". " + str(value))
        tag_nos = input("Select the tags like Eg:  1,3,5,6 for testing please enter 0,1  \n")
        for key, value in model_type.items():
            print(str(key) + ". " + str(value["model_type"]))
        model_nos = input("Select any anyone model type like Eg: 3 for testing please enter 0 \n")
        model_name = input("enter the name of model \n")
        model_height = input("enter the model height \n")
        model_width = input("enter the model width \n")
        model_batch = input("enter the model batch \n")
        model_subdivisions = input("enter the model subdivisions \n")
        model_iterations = input("enter the model iterations \n")

        ## getting all the annotation and image endpoints of the selected classes#########

        tag_names = []

        for tag in tag_nos.split(','):
            tag_names.append(tags[tag.strip()])
        tags_ids = [str(class_mapper.get(str(tag_name))) for tag_name in tag_names]
        media_duration_lists = [class_mediaduration.get(str(tag_id)) for tag_id in tags_ids]
        media_duration_ids = []

        for media_duration_id_list in media_duration_lists:
            media_duration_ids.extend(media_duration_id_list)

        media_ids = []
        for media_duration_id in media_duration_ids:
            if str(mediaduration_mediaid.get(str(media_duration_id))) not in media_ids:
                media_ids.append(str(mediaduration_mediaid.get(str(media_duration_id))))

        blob_end_points = []
        media_id_s = connect.query_database('select image_end_point, annotation_end_point from t_media where id in ({}) and state_id in ("6")'.format(','.join("'" + media_id + "'" for media_id in media_ids)))
        for media_id in media_id_s[0]:
            blob_end_points.append({'blob_image_end_point': media_id[0], 'blob_annotation_end_point': media_id[1]})

        model = model_type[model_nos]

        body = {"total_classes": len(tag_names), "model_name": model_name, "config": {"model_height": model_height, "model_width": model_width, "batch": model_batch, "subdivision": model_subdivisions, "iterations": model_iterations},
                "conv_end_point": model["conv_end_point"], "model_type": model["model_type"], "end_points": blob_end_points, "model_type_id": model["id"], "location_id": location_mapper.get(location)}

        train(body)



    except Exception as e_:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Exception occurred in main : " + str(e_) + ' ' + str(exc_tb.tb_lineno))
