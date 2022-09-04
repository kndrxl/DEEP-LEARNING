# Based on the code from YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Modified on 29/06/2022
"""
Yolo inference: this contains code implementing a detection pipe for Yolo.
"""

import torch
import os
import cv2
import glob
from pathlib import Path
from logging import Logger
from utilities.visualizer import Visualizer, colors

logger = Logger('utils.yolo_inference')

class YoloInference:
    def __init__(
        self, 
        save_dir=None,
        imgsz=(1280, 1280), 
        augment=False, 
        draw_line_thickness=3,
        save_img=True,
        save_crop=False,
        save_txt=True,
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,
        agnostic_nms=False,  # class-agnostic NMS
        classes=None
    ):
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.imgsz = imgsz
        self.augment = augment
        self.draw_line_thickness=draw_line_thickness
        self.save_img = save_img
        self.save_crop = save_crop
        self.save_txt = save_txt
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.classes = classes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_path = Path(__file__).resolve().parents[0]
        self.model_configs =  glob.glob(os.path.join(Path(__file__).resolve().parents[1], 'models/configs/*.yaml'))
        self.valid_model_names = []
        self.model = None
        self.names = None
        self.stride = None
        self.colors = colors

        for model_config in self.model_configs:
            self.valid_model_names.append(f'{Path(model_config).stem}')
    
    def deinitialize(self):
        self.model = None
        self.stride = None
        self.colors = None

    def load_model(self, model_name):
        self.model = torch.hub.load('ultralytics/yolov5', model_name)  # or yolov5n - yolov5x6, custom
        self.stride = self.model.stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.model.float()


    @torch.no_grad()
    def frame_inference(self, im, save_path, save_name):
        if save_path is not None:
            self.save_dir = Path(save_path)

        if self.model is None:
            logger.error('A model has not been loaded! Please call `load_model` first.')
            raise Exception('Model Not Loaded Exception')
        else:
            # create folder for detected frames and labels
            if not os.path.exists(f"{self.save_dir}/frames/detected/"):
                os.makedirs(f"{self.save_dir}/frames/detected/")
                os.makedirs(f"{self.save_dir}/labels/")
            
            img_save_path = str(self.save_dir / 'frames' / 'detected' / Path(save_name))  # im.jpg
            txt_save_path = str(self.save_dir / 'labels' / Path(save_name).stem)

            if isinstance(im, str):
                logger.info(f"Loading image: {im}")
                im = cv2.imread(im) 

            results = self.model(im)
    
            with open(f'{txt_save_path}.txt', 'w') as f:
                f.write('')
                
            imc = im.copy() 
            visualizer = Visualizer(
                imc, 
                line_width=self.draw_line_thickness, 
                example=str(self.names)
            )

            # Write results
            rpd = results.pandas()
            for pd_frame in rpd.xyxy:
                for ind in pd_frame.index:
                    label = f'{pd_frame["name"][ind]} {pd_frame["confidence"][ind]}'
                    box = (
                        pd_frame['xmin'][ind],  
                        pd_frame['ymin'][ind],
                        pd_frame['xmax'][ind],
                        pd_frame['ymax'][ind]
                    )

                    # print(f'{label}  {box}')
                    with open(f'{txt_save_path}.txt', 'a') as f:
                        f.write(f'{label}  {box}\n')

                    visualizer.box_label(box, label, color=self.colors(pd_frame['class'][ind], True))    

            cv2.imwrite(img_save_path, imc)

            return results
