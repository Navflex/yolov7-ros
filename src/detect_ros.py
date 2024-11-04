#!/usr/bin/python3

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.ros import create_detection_msg
from visualizer import draw_detections

import os
from typing import Tuple, Union, List

import torch
import cv2
from torchvision.transforms import ToTensor
import numpy as np
import rospy

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Import TensorRT dependencies
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def parse_classes_file(path):
    classes = []
    with open(path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            classes.append(line)
    return classes


def rescale(ori_shape: Tuple[int, int], boxes: Union[torch.Tensor, np.ndarray],
            target_shape: Tuple[int, int]):
    """Rescale the output to the original image shape
    :param ori_shape: original width and height [width, height].
    :param boxes: original bounding boxes as a torch.Tensor or np.array or shape
        [num_boxes, >=4], where the first 4 entries of each element have to be
        [x1, y1, x2, y2].
    :param target_shape: target width and height [width, height].
    """
    xscale = target_shape[1] / ori_shape[1]
    yscale = target_shape[0] / ori_shape[0]

    boxes[:, [0, 2]] *= xscale
    boxes[:, [1, 3]] *= yscale

    return boxes


class YoloV7:
    def __init__(self, weights, conf_thresh: float = 0.5, iou_thresh: float = 0.45,
                 device: str = "cuda"):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = attempt_load(weights, map_location=device)
        self.model.eval()

    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.model(img)[0]
        detections = non_max_suppression(
            pred_results, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
        )
        if detections:
            detections = detections[0]
        return detections

class YoloTRT:
    """Class for running YOLOv7 inference using TensorRT."""
    def __init__(self, engine_path, conf_thresh=0.5, iou_thresh=0.4):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh  # IoU threshold for NMS
        self.stream = cuda.Stream()

        # Initialize TensorRT engine and context
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Set up memory allocation
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)
        self.input_size = trt.volume(self.input_shape) * self.engine.max_batch_size
        self.output_size = trt.volume(self.output_shape) * self.engine.max_batch_size
        self.dtype = trt.nptype(self.engine.get_binding_dtype(0))

        # Allocate host and device buffers
        self.host_input = cuda.pagelocked_empty(self.input_size, dtype=self.dtype)
        self.cuda_input = cuda.mem_alloc(self.host_input.nbytes)
        self.host_output = cuda.pagelocked_empty(self.output_size, dtype=self.dtype)
        self.cuda_output = cuda.mem_alloc(self.host_output.nbytes)

    def preprocess(self, img):
        """Preprocess the image for TensorRT inference.
        The image is resized to match the TensorRT input shape, converted to RGB, 
        normalized, and flattened into a contiguous array for TensorRT.
        """
        img_resized = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_flattened = img_transposed.flatten()
        np.copyto(self.host_input, img_flattened)
        return img

    def inference(self, img):
        """Run inference using TensorRT."""
        self.preprocess(img)
        cuda.memcpy_htod_async(self.cuda_input, self.host_input, self.stream)
        self.context.execute_async_v2(bindings=[int(self.cuda_input), int(self.cuda_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_output, self.cuda_output, self.stream)
        self.stream.synchronize()
        return self.postprocess(self.host_output, img.shape[:2])

    def postprocess(self, outputs, original_shape):
        """Postprocess the output from TensorRT and apply NMS."""
        h_orig, w_orig = original_shape
        num_detections = int(outputs[0])
        detections = []

        for i in range(num_detections):
            x1, y1, x2, y2, conf, class_id = outputs[1 + 6 * i: 1 + 6 * (i + 1)]
            if conf > self.conf_thresh:
                # Scale the bounding boxes back to the original image size
                x1 = int(x1 * (w_orig / self.input_shape[2]))
                y1 = int(y1 * (h_orig / self.input_shape[1]))
                x2 = int(x2 * (w_orig / self.input_shape[2]))
                y2 = int(y2 * (h_orig / self.input_shape[1]))
                detections.append([x1, y1, x2, y2, conf, class_id])

        # Apply Non-Maximum Suppression
        if detections:
            detections = torch.tensor(detections)  # Convert to a PyTorch tensor
            detections = non_max_suppression(
                detections.unsqueeze(0), self.conf_thresh, self.iou_thresh
            )[0]

        return detections.numpy() if detections is not None else np.array([])


class Yolov7Publisher:
    def __init__(self, img_topic: str, weights: str, conf_thresh: float = 0.5,
                 iou_thresh: float = 0.45, pub_topic: str = "yolov7_detections",
                 device: str = "cuda",
                 img_size: Union[Tuple[int, int], None] = (640, 640),
                 queue_size: int = 1, visualize: bool = False,
                 class_labels: Union[List, None] = None,
                 use_tensorrt: bool = False, trt_engine: str = None):
        """
        :param img_topic: name of the image topic to listen to
        :param weights: path/to/yolo_weights.pt
        :param conf_thresh: confidence threshold
        :param iou_thresh: intersection over union threshold
        :param pub_topic: name of the output topic (will be published under the
            namespace '/yolov7')
        :param device: device to do inference on (e.g., 'cuda' or 'cpu')
        :param queue_size: queue size for publishers
        :visualize: flag to enable publishing the detections visualized in the image
        :param img_size: (height, width) to which the img is resized before being
            fed into the yolo network. Final output coordinates will be rescaled to
            the original img size.
        :param class_labels: List of length num_classes, containing the class
            labels. The i-th element in this list corresponds to the i-th
            class id. Only for viszalization. If it is None, then no class
            labels are visualized.
        """
        self.img_size = img_size
        self.device = device
        self.class_labels = class_labels

        vis_topic = pub_topic + "visualization" if pub_topic.endswith("/") else \
            pub_topic + "/visualization"
        self.visualization_publisher = rospy.Publisher(
            vis_topic, Image, queue_size=queue_size
        ) if visualize else None

        self.bridge = CvBridge()

        self.tensorize = ToTensor()
        if self.use_tensorrt:
            if not trt_engine:
                raise ValueError("TensorRT engine path must be provided if use_tensorrt is True.")
            self.model = YoloTRT(trt_engine, conf_thresh)
        else:
            self.model = YoloV7(
                weights=weights, conf_thresh=conf_thresh, iou_thresh=iou_thresh,
                device=device
            )
        self.img_subscriber = rospy.Subscriber(
            img_topic, Image, self.process_img_msg
        )
        self.detection_publisher = rospy.Publisher(
            pub_topic, Detection2DArray, queue_size=queue_size
        )
        rospy.loginfo("YOLOv7 initialization complete. Ready to start inference")

    def process_img_msg(self, img_msg: Image):
        """ callback function for publisher """
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='passthrough'
        )

        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)

        h_orig, w_orig, c = np_img_orig.shape

        if self.use_tensorrt:
            # TensorRT inference
            detections = self.model.inference(np_img_orig)
        else:
            # automatically resize the image to the next smaller possible size
            w_scaled, h_scaled = self.img_size
            np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))

            # conversion to torch tensor (copied from original yolov7 repo)
            img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = torch.from_numpy(np.ascontiguousarray(img))
            img = img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.
            img = img.to(self.device)

            # inference & rescaling the output to original img size
            detections = self.model.inference(img)
            detections[:, :4] = rescale(
                [h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig])
            detections[:, :4] = detections[:, :4].round()

        # publishing
        detection_msg = create_detection_msg(img_msg, detections)
        self.detection_publisher.publish(detection_msg)

        # visualizing if required
        if self.visualization_publisher:
            bboxes = [[int(x1), int(y1), int(x2), int(y2)]
                      for x1, y1, x2, y2 in detections[:, :4].tolist()]
            classes = [int(c) for c in detections[:, 5].tolist()]
            rgb_img = cv2.cvtColor(np_img_orig, cv2.COLOR_BGR2RGB)
            vis_img = draw_detections(rgb_img, bboxes, classes,
                                      self.class_labels)
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding='rgb8')
            self.visualization_publisher.publish(vis_msg)


if __name__ == "__main__":
    rospy.init_node("yolov7_node")

    ns = rospy.get_name() + "/"

    weights_path = rospy.get_param(ns + "weights_path")
    classes_path = rospy.get_param(ns + "classes_path")
    img_topic = rospy.get_param(ns + "img_topic")
    out_topic = rospy.get_param(ns + "out_topic")
    conf_thresh = rospy.get_param(ns + "conf_thresh")
    iou_thresh = rospy.get_param(ns + "iou_thresh")
    queue_size = rospy.get_param(ns + "queue_size")
    img_size = rospy.get_param(ns + "img_size")
    visualize = rospy.get_param(ns + "visualize")
    device = rospy.get_param(ns + "device")
    trt_engine_path = rospy.get_param(ns + "trt_engine_path", None)  # Added parameter for TensorRT engine
    use_tensorrt = rospy.get_param(ns + "use_tensorrt", False)  # Added flag for TensorRT usage

    # some sanity checks
    if not os.path.isfile(weights_path):
        raise FileExistsError(f"Weights not found ({weights_path}).")
    
    if classes_path: 
        if not os.path.isfile(classes_path):
            raise FileExistsError(f"Classes file not found ({classes_path}).")
        classes = parse_classes_file(classes_path)
    else:
        rospy.loginfo("No class file provided. Class labels will not be visualized.")
        classes = None

    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.")


    publisher = Yolov7Publisher(
        img_topic=img_topic,
        pub_topic=out_topic,
        weights=weights_path,
        device=device,
        visualize=visualize,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        img_size=(img_size, img_size),
        queue_size=queue_size,
        class_labels=classes,
        trt_engine=trt_engine_path,  # Pass the TensorRT engine path
        use_tensorrt=use_tensorrt 
    )

    rospy.spin()
