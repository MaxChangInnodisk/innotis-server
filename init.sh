#!/bin/bash

COL=$(stty size | cut -d" " -f2)
RED='\033[0;31m'
BRED='\033[7;31m'
Green='\033[0;32m'
BGreen='\033[7;32m'
GRAY_GR='\033[4;32m'
RSET='\033[0m' # No Color

pwd=$(realpath $(dirname ${0} ))
echo $pwd

function title() {
    SYM=$2
    printf "\n"
    if [[ -z $2 ]];then printf "$SYM%.0s" $(seq 1 $COL);fi
    echo -e "${BGreen}${1}${RSET}"
}

function sub_title() {
    printf "\n"
    echo -e "${GRAY_GR}${1}${RSET}"
}

function cp2deploy() {
    SRC=$1
    DST=$2
    DST_DIR=$(dirname ${DST})
    if [[ ! -d "${DST_DIR}" ]];then
        mkdir -p ${DST_DIR}
    fi
    echo "Copy ${SRC} to ${DST}"
    cp ${SRC} ${DST}
}

if [[ $1 = "build" ]];then

    ROOT=/innotis-server/triton-deploy
    BUILD_ROOT=/innotis-server/build
    
    title "Build Tools For Convert"
    cd ${BUILD_ROOT}
    cmake ..
    make
    
    title "Convert YOLOv4 Model From Serial to TensorRT Engine"
    
    sub_title "Converting yolov4.wts ..."
    ./main
    sub_title "Converting yolov4will.wts ..."
    ./main -n yolov4will

    title "Prepare Triton-Deploy"
    sub_title "Model Repository: <innotis-server>/triton-deploy/models/<task_name>/<version>/model.plan"
    YOLO_SRC=${BUILD_ROOT}/yolov4.engine
    YOLO_WILL_SRC=${BUILD_ROOT}/yolov4custom.engine

    YOLO_DST=${ROOT}/models/yolov4/1/model.plan
    YOLO_WILL_DST=${ROOT}/models/yolov4_will/1/model.plan

    cp2deploy ${YOLO_SRC} ${YOLO_DST}
    cp2deploy ${YOLO_WILL_SRC} ${YOLO_WILL_DST}
    
    sub_title "Dynamic Library: <innotis-server>/triton-deploy/plugin/<*.so>"
    LIB_SRC=${BUILD_ROOT}/liblayerplugin.so
    LIB_DST=${ROOT}/plugins/

    cp2deploy ${LIB_SRC} ${LIB_DST}

    echo "Change Owner of All Files"
    chown -R 1000:1000 /innotis-server
    
else

    # Download Google Drive Large File via wget
    # wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt
    title "Initialize"
    if [[ ! -d "${pwd}/triton-deploy/" ]];then
        sub_title "Prepare File Structure for Triton Deploy"
        # https://drive.google.com/file/d/1ts-oIAiKiPcCiORl7-mz22kxu_5QeuTU/view?usp=sharing
        # FILEID = 1ts-oIAiKiPcCiORl7-mz22kxu_5QeuTU
        # FILENAME = triton-deploy.tar.gz
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ts-oIAiKiPcCiORl7-mz22kxu_5QeuTU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ts-oIAiKiPcCiORl7-mz22kxu_5QeuTU" -O triton-deploy.tar.gz && rm -rf /tmp/cookies.txt
        tar zxvf triton-deploy.tar.gz && rm triton-deploy.tar.gz
    fi

    if [[ ! -d "${pwd}/models/" ]];then
        sub_title "Download Custom YOLOv4 Model"
        # https://drive.google.com/file/d/19EkdUS5oYrbcRSKH5cxVTZpcVbylGpFe/view?usp=sharing
        # FILEID = 19EkdUS5oYrbcRSKH5cxVTZpcVbylGpFe
        # FILENAME = triton-models.tar.gz
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19EkdUS5oYrbcRSKH5cxVTZpcVbylGpFe' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19EkdUS5oYrbcRSKH5cxVTZpcVbylGpFe" -O triton-models.tar.gz && rm -rf /tmp/cookies.txt
        tar zxvf triton-models.tar.gz && rm triton-models.tar.gz
    fi

    sub_title "Copy Model from ./models to ./build"
    if [[ -d "${pwd}/build/" ]];then rm -rf ./build ; fi
    mkdir ./build
    cp ./models/*/*.wts ./build

    title "Run TensorRT's Docker Container"
    docker run --gpus all -it --rm -v $(pwd):/innotis-server nvcr.io/nvidia/tensorrt:21.03-py3 /innotis-server/init.sh build

    sub_title "Store IP Information to 'server_ip.txt'"
    hostname -I | cut -d ' ' -f 1 > server_ip.txt
    IP=$(cat ./server_ip.txt)
    echo -e "\nServer IP : ${BRED}${IP}${RSET}"

    title "All Done"
    exit 1
fi