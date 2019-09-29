import cv2
import numpy as np


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device
        self.video = cv2.VideoCapture(2)
        # Load config stuff
        BASE_DIR = "C:/Users/e-on/Desktop/d/"
        classes = BASE_DIR + "objects.names"
        weights = BASE_DIR + "yolov3-tiny_50000.weights"
        config = BASE_DIR + "yolov3-tiny.cfg"
        
        # Load names classes
        with open(classes, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Generate color for each class randomly
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Define network from configuration file and load the weights from the given weights file
        self.net = cv2.dnn.readNet(weights, config)
        
    # Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
    def getOutputsNames(self,net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Darw a rectangle surrounding the object and its class name
    def draw_pred(self, img, class_id, classes, COLORS, x, y, x_plus_w, y_plus_h):

        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
        
    def stream(self):
        hasframe, image = self.video.read()
        # images=cv2.resize(images, (620, 480))

        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), [0, 0, 0], True, crop=False)
        Width = image.shape[1]
        Height = image.shape[0]
        self.net.setInput(blob)
    
        outs = self.net.forward(self.getOutputsNames(self.net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # print(len(outs))

        # In case of tiny YOLOv3 we have 2 output(outs) from 2 different scales [3 bounding box per each scale]
        # For normal normal YOLOv3 we have 3 output(outs) from 3 different scales [3 bounding box per each scale]

        # For tiny YOLOv3, the first output will be 507x6 = 13x13x18
        # 18=3*(4+1+1) 4 boundingbox offsets, 1 objectness prediction, and 1 class score.
        # and the second output will be = 2028x6=26x26x18 (18=3*6)

        for out in outs:
            # print(out.shape)
            for detection in out:

                # each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
                scores = detection[5:]  # classes scores starts from index 5
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        # apply  non-maximum suppression algorithm on the bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        x1, y1, w1, h1 = 0, 0, 0, 0;

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            x1 = x
            y1 = y
            w1 = w
            h1 = h
            self.draw_pred(image, class_ids[i], self.classes, self.COLORS, round(x), round(y), round(x + w), round(y + h))

        # Put efficiency information.
        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        lable2 = 'w: {w} , h: {h} '.format(w=w1,h=h1)

        cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(image, lable2, (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        ret, jpeg = cv2.imencode('.jpg', image)
        
        return jpeg.tobytes()
    
    def transform(self, x: int, y: int, origin: tuple, length_real: float, height_real: float, length_pixel: int, height_pixel: int) -> tuple:
        # transform x and y coordinate
        x -= origin[0]
        y -= origin[1]
        # calculate in units
        x_unit = x * (length_real/length_pixel)
        y_unit = y * (height_real/height_pixel)
        return x_unit, y_unit
    
    
    def get_prediction(self, origin: tuple, length_real: float, height_real: float, length_pixel: int, height_pixel: int):
        hasframe, image = self.video.read()

        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), [0, 0, 0], True, crop=False)
        Width = image.shape[1]
        Height = image.shape[0]
        self.net.setInput(blob)
    
        outs = self.net.forward(self.getOutputsNames(self.net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            # print(out.shape)
            for detection in out:

                # each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
                scores = detection[5:]  # classes scores starts from index 5
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        # apply  non-maximum suppression algorithm on the bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        x1, y1, w1, h1 = 0, 0, 0, 0;
        
        coordinates = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            x1 = 300
            y1 = 150
            x_transformed, y_transformed = self.transform(x1, y1, origin, length_real, height_real, length_pixel, height_pixel)
            w_transformed = w * (length_real/length_pixel)
            h_transformed = h * (height_real/height_pixel)
            coordinates.append((int(x_transformed), int(y_transformed), int(w_transformed), int(h_transformed)))
            
            
        
        return coordinates    
        