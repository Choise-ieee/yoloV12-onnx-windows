According to the reference:https://github.com/sunsmarterjie/yolov12/issues/122#issue-3149851112 and https://github.com/mohamedsamirx/YOLOv12-ONNX-CPP

Step-1 windows compile
1. make sure VS2019 installed sucesss
2. make sure anaconda installed sucess
3. create conda environment:conda create -n yolov12 python=3.11
4. install PyTorch + CUDA 12.9(12.1)：pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu121
5. prepare the packet for next Flash-Attention:pip install ninja packaging wheel gitpython
6. Flash-Attention setup:
   git config --system core.longpaths true
   git clone --recurse-submodules -b v2.7.4 https://github.com/Dao-AILab/flash-attention.git
   cd flash-attention
   python setup.py develop
   python setup.py bdist_wheel
7. Then the Flash-Attention will be installed as .whl file,which can be seen from pip-list
8. then pip install the following:
   timm==1.0.14
   albumentations==2.0.4
   onnx==1.14.0
   onnxruntime==1.15.1
   pycocotools==2.0.7
   PyYAML==6.0.1
   scipy==1.13.0
   onnxslim==0.1.31
   onnxruntime-gpu==1.18.0
   gradio==4.44.1
   opencv-python==4.9.0.80
   psutil==5.9.8
   py-cpuinfo==9.0.0
   huggingface-hub==0.23.2
   safetensors==0.4.3
   numpy==1.26.4
9. The yolov12-windows have installed sucess: you can check by command: yolo predict model=./yolov12x.pt source=./bus.jpg show=True
10. to transcode from pt to onnx by the command:
    from ultralytics import YOLO
    model = YOLO('yolov12{n/s/m/l/x}.pt')
    model.export(format="onnx", half=True)  # half=ture means fp16, otherwise is FP32
11. then in VS2019, you pick up the cpp files,in onnx environment, I use the onnxruunming-time 18.1

![image](https://github.com/user-attachments/assets/5a56884f-8479-41db-97a3-785715649a73)

Note:
I sucess achieve in VS2019, cuda 12.1 and onnxrunningtime-1.18 in windows10 and windows server2012.
1.C++ inference in step 11, If you use cudnn9.x please use onnxruunming-time 18.1.1, and if you use cudnn8.x use onnx 18.1.0
![96a2ab3cb7ff4a5f9279acbad4d3162](https://github.com/user-attachments/assets/f2a78ac6-99e0-4a47-910e-ec724618c049)
2.If "out of memory" displayed in Flash-Attention setup, you can config more virtual memory in windows system
![image](https://github.com/user-attachments/assets/b7d2f1e7-8956-4d90-baa9-b6422a98554d)
3. Ampere GPUs and newer is needed for Flash-Attention in step 6！！！！But the inference in vs2019 is no need the Ampere GPU and newer
![image](https://github.com/user-attachments/assets/4ef66554-e80e-4891-8f7c-4e7c34d18179)
4. the Flash-Attention setup cost to much time, I use i9-14900k and ultra9-295K, use 2-hours long.


