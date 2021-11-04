# InnoTIS

InnoTIS 是 Innodisk and Aetina 用來提供 Aetina Server 運行AI模型的效果，我們結合了 NVIDIA Triton Inference Server 的技術讓使用者可以透過gRPC的方式傳送資料到我們的 Aetina Server 進行 AI 推論進而取得辨識結果。

---
## 特色

* 客製化版本，目前僅提供三個模型可以使用。
   1. **DENSENET_ONNX** ( 原廠範例 )
   2. **YOLOV4** ( COCO Dataset )
   3. **YOLOV4_WILL** ( Can Detect Does People Wear A Mask )
* 針對客製化模型修改程式碼 `*.cpp` `*.h`，如須修改請查看 [文章](https://max-c.notion.site/Custom-Model-with-YOLOv4-277f3185e53c4f25be5d46cb117cb12a)。 
* 採用 Triton 之中提供的 `gRPC` API，故 HTTP 的埠號不會有反應。
---
## 如何使用？

1. **啟動 innotis-server 並記下 ip 位置**
   1. Download repository
       ```bash
       $ git clone https://github.com/MaxChangInnodisk/innotis-server.git
       $ cd innotis-server
       ```
   2. Run `init.sh`

       ```bash
       $ ./init.sh
       ```
      *  請記下 IP 位置 (innotis-client 啟用後用使用到)：
           
           ![image](figures/ip.png)
     
   3. Run `run.sh`
      
       ```bash
       $ ./run.sh
       ```
       * 確保 GRPC and HTTP service 已經被開啟：

           ![image](figures/service_started.png)
2. **啟動 innotis-client ( 請使用第二個Terminal )**

    Github: [innotis-client](https://github.com/MaxChangInnodisk/innotis-client)

   * DockerHub: pull image & run container from docker hub
       ```bash
       $ docker run -t -p 5000:5000 -t maxchanginnodisk/innotis
       ```
   * Dockerfile: you can also build from docker file
     * Please visit [innotis-client](https://github.com/MaxChangInnodisk/innotis-client) to get more information.
   * Miniconda: virtual environment might be a great idea for developer
     * Please visit [innotis-client](https://github.com/MaxChangInnodisk/innotis-client) to get more information.

3. **開啟瀏覽器 輸入 localhost:5000**

    * 注意：Server IP 記得要修改。

4. **盡情遊玩**

---
## 參考

* [YOLOv4 on Triton Inference Server with TensorRT](https://github.com/isarsoft/yolov4-triton-tensorrt)
* [TensorRTx](https://github.com/wang-xinyu/tensorrtx)

