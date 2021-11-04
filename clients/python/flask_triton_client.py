#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import cv2
import os

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from labels import COCOLabels, WillLabels

# get terminal size
columns, rows = os.get_terminal_size(0)
def print_title(text):
        print('-'*columns)
        print(text, '\n')

class Client:

    def __init__(self, url, model_name='yolov4', label_name='COCO', conf=0.9, nms=0.1, info=False, client_timeout=False):
        
        ##############################################
        # initialize
        self.model = model_name
        self.label = self.get_label(label_name)

        self.confidence = conf
        self.nms = nms

        self.info = info
        self.client_timeout = client_timeout

        ##############################################
        # Create Triton Client
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=url,verbose=False)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

        ##############################################
        # Check Health and Model Information
        self.check_triton_status()
        if self.info==True : self.get_model_info()

    def get_label(self, lable_name):
        ''' 
        取得標籤物件 
        '''
        if lable_name=='COCO':
            return COCOLabels
        elif lable_name=='WILL':
            return WillLabels

    def check_triton_status(self):
        '''
        檢查 Triton 的狀態
        '''
        if not self.triton_client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)
        if not self.triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)
        if not self.triton_client.is_model_ready(self.model):
            print("FAILED : is_model_ready")
            sys.exit(1)   

    def get_model_info(self):
        '''
        取得 Triton Server 的模型資訊
        '''
        try:
            metadata = self.triton_client.get_model_metadata(self.model)
            print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED : get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration
        try:
            config = self.triton_client.get_model_config(self.model)
            if not (config.config.name == self.model):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)            

    def image_infer(self, image, width, height, output, input_type='FP32'):
        ##############################################
        # init
        if image is None:
            print("FAILED: no input image")
            sys.exit(1)
        
        image_draw = image.copy()
        inputs, outputs = list(), list()
        inputs.append(grpcclient.InferInput('input', [1, 3, width, height], input_type))
        outputs.append(grpcclient.InferRequestedOutput('detections'))

        # Data Preprocess
        print("Creating buffer from image file...")
        image_buffer = preprocess(image, [width, height])   # image_buffer is for inference
        image_buffer = np.expand_dims(image_buffer, axis=0)
        inputs[0].set_data_from_numpy(image_buffer)    

        ##############################################
        # Send Inference request and get response
        print("Invoking inference...")
        results = self.triton_client.infer(model_name=self.model,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=self.client_timeout)

        if self.info:
            statistics = self.triton_client.get_inference_statistics(model_name=self.model)
            if len(statistics.model_stats) != 1:
                print("FAILED: get_inference_statistics")
                sys.exit(1)
            print(statistics)
        
        print("Inference Done !!!")

        ##############################################
        # Parse Output
        result = results.as_numpy('detections')
        print(f"Received result buffer of size {result.shape}")
        print(f"Naive buffer sum: {np.sum(result)}")

        detected_objects = postprocess(result, image.shape[1], image.shape[0], [width, height], self.confidence, self.nms)
        print(f"Detected objects: {len(detected_objects)}")

        ##############################################
        # Rendering Results : bounding box, text
        for box in detected_objects:
            print(f"{self.label(box.classID).name}: {box.confidence}")
            image_draw = render_box(image_draw, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
            size = get_text_size(image_draw, f"{self.label(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
            image_draw = render_filled_box(image_draw, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), color=(220, 220, 220))
            image_draw = render_text(image_draw, f"{self.label(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)

        ##############################################
        # Save files or Show results
        if output:
            cv2.imwrite(output, image_draw)
            print(f"Saved result to {output}")
        else:
            cv2.imshow('image', image_draw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def video_infer(self, cap=''):
        print('Not yet !!!!')

    def infer(self, mode, input, width, height, output):
        '''
        進行推論 可以選擇模式 : [ 'image', 'video' ]
        '''
        print_title(f'Running in {mode} mode')

        if mode == 'image':
            image = cv2.imread(str(input))
            self.image_infer(image, width, height, output)
        elif mode == 'video':
            # cap = cv2.VideoCapture(input)
            self.video_infer()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('mode',
                        choices=['dummy', 'image', 'video'],
                        default='dummy',
                        help='Run mode. \'dummy\' will send an emtpy buffer to the server to test if inference works. \'image\' will process an image. \'video\' will process a video.')
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input file to load from in image or video mode')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=False,
                        default='yolov4',
                        help='Inference model name, default yolov4')
    parser.add_argument('--width',
                        type=int,
                        required=False,
                        default=608,
                        help='Inference model input width, default 608')
    parser.add_argument('--height',
                        type=int,
                        required=False,
                        default=608,
                        help='Inference model input height, default 608')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL, default localhost:8001')
    parser.add_argument('-o',
                        '--out',
                        type=str,
                        required=False,
                        default='',
                        help='Write output into file instead of displaying it')
    parser.add_argument('-c',
                        '--confidence',
                        type=float,
                        required=False,
                        default=0.8,
                        help='Confidence threshold for detected objects, default 0.8')
    parser.add_argument('-n',
                        '--nms',
                        type=float,
                        required=False,
                        default=0.5,
                        help='Non-maximum suppression threshold for filtering raw boxes, default 0.5')
    parser.add_argument('-f',
                        '--fps',
                        type=float,
                        required=False,
                        default=24.0,
                        help='Video output fps, default 24.0 FPS')
    parser.add_argument('-i',
                        '--model-info',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Print model status, configuration and statistics')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose client output')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds, default no timeout')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-r',
                        '--root-certificates',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded root certificates, default none')
    parser.add_argument('-p',
                        '--private-key',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded private key, default is none')
    parser.add_argument('-x',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain default is none')
    parser.add_argument('-l',
                        '--label',
                        type=str,
                        required=False,
                        default='COCO',
                        help='Taget Label')
    
    FLAGS = parser.parse_args()

    '''
    重新改寫之後 把 Client 寫成物件
    可以直接調用 infer
    這個將方便後續 Flask 的調用
    '''
    client = Client(url=FLAGS.url,
                    model_name=FLAGS.model, 
                    label_name=FLAGS.label, 
                    conf=FLAGS.confidence ,
                    nms=FLAGS.nms ,
                    info=FLAGS.model_info ,
                    client_timeout=FLAGS.client_timeout)

    client.infer(FLAGS.mode, FLAGS.input, FLAGS.width, FLAGS.height, FLAGS.out)