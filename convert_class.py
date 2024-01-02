import cv2
import os
import torch
import pandas as pd
import json
from PIL import Image
import xml.etree.ElementTree as ET
import sys
import argparse

class yolo_converter():
    def __init__(self, model_path, convert_type, input_dir, output_dir):
        self.convert_type = convert_type
        self.input_dir = input_dir
        self.output_dir = output_dir 

        if self.output_dir == "":
            self.output_dir = self.input_dir

        self.model = torch.hub.load('yolov5', 'custom', path=model_path, source='local', autoshape=True, force_reload=True)


    def convert_to_txt(self):

        # dir_path = "./images"
        # output_dir = "./output_txt"  # Output directory for storing Darknet format results

        # input_dir = sys.argv[1]
        # output_dir = sys.argv[2]


        os.makedirs(args.output_dir, exist_ok=True)

        img_files = [f for f in os.listdir(args.input_dir) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

        length = len(img_files)

        for i, f in enumerate(img_files):
            img_path = os.path.join(args.input_dir, f)
            img = cv2.imread(img_path)

            result = self.model(img, size=640)  # Inference on a single image

            output_file = os.path.join(args.output_dir, os.path.splitext(f)[0] + ".txt")  
            with open(output_file, "w") as f_out:
                count=0
                for *box, class_id, _ in result.xyxy[0]:
                    class_id = int(class_id) 
                    height = (box[3] - box[1]) / img.shape[0]
                    x_center = (box[0] + box[2]) / 2 / img.shape[1]
                    y_center = (box[1] + box[3]) / 2 / img.shape[0]
                    width = (box[2] - box[0]) / img.shape[1]
                    pred = result.pandas().xyxy[0]
                    class_label=pred["class"][count]
                    count+=1
                    f_out.write(f"{class_label} {x_center} {y_center} {width} {height}\n")

            print(f"Processed image {i+1}/{length}")

        print("Darknet format results saved.")


    def convert_to_json(self):

        # dir_path = "./images"
        # output_dir = "./output_json" 

        # input_dir = sys.argv[1]
        # output_dir = sys.argv[2]

        os.makedirs(args.output_dir, exist_ok=True)

        img_files = [f for f in os.listdir(args.input_dir) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

        length = len(img_files)
        for i, f in enumerate(img_files):
            img_path = os.path.join(args.input_dir, f)
            img = cv2.imread(img_path)
            result = self.model(img, size=640)
            output_file = os.path.join(args.output_dir, os.path.splitext(f)[0] + ".json")
            output_data = []
            count=0
            for *box, class_id, _ in result.xyxy[0]:
                class_id = int(class_id)
                height = (box[3] - box[1]) / img.shape[0]
                x_center = (box[0] + box[2]) / 2 / img.shape[1]
                y_center = (box[1] + box[3]) / 2 / img.shape[0]
                width = (box[2] - box[0]) / img.shape[1]
                pred = result.pandas().xyxy[0]
                class_name = pred["name"][count]
                count += 1
                box_data = {
                    "class_label": class_name,
                    "x_center": x_center, 
                    "y_center": y_center,
                    "width": width,
                    "height": height
                }
                output_data.append(box_data)
            
            with open(output_file, "w") as f_out:
                json.dump(output_data, f_out, indent=4, default=lambda x: x.tolist())
            
            print(f"Processed image {i+1}/{length}")

        print("JSON format results saved.")

    def convert_to_xml(self):

        # dir_path = "./images"
        # output_dir = "./output_xml" 

        # input_dir = sys.argv[1]
        # output_dir = sys.argv[2]

        os.makedirs(args.output_dir, exist_ok=True)

        img_files = [f for f in os.listdir(args.input_dir) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

        length = len(img_files)
        for i, f in enumerate(img_files):
            img_path = os.path.join(args.input_dir, f)
            img = cv2.imread(img_path)
            result = self.model(img, size=640)
            classes = []
            output_file = os.path.join(args.output_dir, os.path.splitext(f)[0] + ".xml")

            # Create the XML element tree structure
            root = ET.Element("annotation")
            folder = ET.SubElement(root, "folder")
            folder.text = "output"
            filename = ET.SubElement(root, "filename")
            filename.text = os.path.splitext(f)[0] + ".jpg"
            path = ET.SubElement(root, "path")
            path.text = os.path.join(args.output_dir, os.path.splitext(f)[0] + ".jpg")

            source = ET.SubElement(root, "source")
            database = ET.SubElement(source, "database")
            database.text = "Unknown"

            size = ET.SubElement(root, "size")
            width_elem = ET.SubElement(size, "width")
            width_elem.text = str(img.shape[1])
            height_elem = ET.SubElement(size, "height")
            height_elem.text = str(img.shape[0])
            depth = ET.SubElement(size, "depth")
            depth.text = str(img.shape[2])

            segmented = ET.SubElement(root, "segmented")
            segmented.text = "0"
            
            count=0
            for *box, class_id, _ in result.xyxy[0]:
                class_id = int(class_id)
                # class_name = result.names[int(class_id)]
                pred = result.pandas().xyxy[0]
                class_name=pred["name"][count]
                count+=1
                xmin = int((box[0] / img.shape[1]) * img.shape[1])
                ymin = int((box[1] / img.shape[0]) * img.shape[0])
                xmax = int((box[2] / img.shape[1]) * img.shape[1])
                ymax = int((box[3] / img.shape[0]) * img.shape[0])

                object_elem = ET.SubElement(root, "object")
                name = ET.SubElement(object_elem, "name")
                name.text = class_name
                pose = ET.SubElement(object_elem, "pose")
                pose.text = "Unspecified"
                truncated = ET.SubElement(object_elem, "truncated")
                truncated.text = "0"
                difficult = ET.SubElement(object_elem, "difficult")
                difficult.text = "0"
                bndbox = ET.SubElement(object_elem, "bndbox")
                xmin_elem = ET.SubElement(bndbox, "xmin")
                xmin_elem.text = str(xmin)
                ymin_elem = ET.SubElement(bndbox, "ymin")
                ymin_elem.text = str(ymin)
                xmax_elem = ET.SubElement(bndbox, "xmax")
                xmax_elem.text = str(xmax)
                ymax_elem = ET.SubElement(bndbox, "ymax")
                ymax_elem.text = str(ymax)

            tree = ET.ElementTree(root)
            tree.write(output_file)

            print(f"Processed image {i+1}/{length}")

        print("XML format results saved.")


if __name__ == "__main__":
    # # Parse command-line arguments
    # if len(sys.argv) < 5:
    #     print("Usage: python convert_class.py model_path convert_type input_dir output_dir")
    #     sys.exit(1)

    # model_path = sys.argv[1]
    # convert_type = sys.argv[2]
    # input_dir = sys.argv[3]
    # output_dir = sys.argv[4]

    # # Create an instance of the yolo_converter class
    # converter = yolo_converter(model_path, convert_type, input_dir, output_dir)

    parser = argparse.ArgumentParser(description="Yolo Converter")

    parser.add_argument("--model_path", type=str, help="Path to the YOLO model file")
    parser.add_argument("--convert_type", type=str, choices=["txt", "json", "xml"], help="Type of conversion: txt, json, or xml")
    parser.add_argument("--input_dir", type=str, help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, help="Output directory to store converted files")

    args = parser.parse_args()

    # Create an instance of the yolo_converter class
    converter = yolo_converter(args.model_path, args.convert_type, args.input_dir, args.output_dir)

    # Perform the conversion based on the specified convert_type
    if args.convert_type == "txt":
        converter.convert_to_txt()
    elif args.convert_type == "json":
        converter.convert_to_json()
    elif args.convert_type == "xml":
        converter.convert_to_xml()
    else:
        print("Invalid convert_type. Supported types: txt, json, xml")


# --model_path D:\Personal\yolo_v8\yolov8_models_infer\yolov8n.pt --convert_type txt --input_dir D:\Personal\yolo_v8\yolo_converter_class\images --output_dir D:/Personal/yolo_v8/yolo_converter_class/images_output