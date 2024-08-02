# train_yolov5.py
import os
import torch
from pathlib import Path
import multiprocessing

def main():
    # Clone YOLOv5 repository if not already present
    if not Path('yolov5').exists():
        os.system('git clone https://github.com/ultralytics/yolov5')

    # Install requirements
    os.system('pip install -qr yolov5/requirements.txt')

    from yolov5 import train, val

    # Set up directories
    dataset_dir = '../data/raw'

    # Train the model
    train.run(data=f'{dataset_dir}/data.yaml',  # path to data.yaml file
              epochs=50,  # number of training epochs
              imgsz=640,  # image size
              batch_size=8,  # batch size
              project='yolov5_crater_detection',  # project name
              name='exp',  # experiment name
              exist_ok=True)  # overwrite existing experiment

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
