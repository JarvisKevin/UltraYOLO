import numpy as np
import cv2
from typing import Any, Dict, List, Tuple, Union

class CompressByResize:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scale_fill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(
        self, max_range=2, prob=1.0
    ):

        self.max_range = max_range
        self.prob = prob

    def __call__(self, labels: Dict[str, Any] = None, image: np.ndarray = None) -> Union[Dict[str, Any], np.ndarray]:
        """
        Resize and pad an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        Args:
            labels (Dict[str, Any] | None): A dictionary containing image data and associated labels, or empty dict if None.
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.

        Returns:
            (Dict[str, Any] | nd.ndarray): If 'labels' is provided, returns an updated dictionary with the resized and padded image,
                updated labels, and additional metadata. If 'labels' is empty, returns the resized
                and padded image.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if np.random.random() > self.prob:
            return labels if labels is not None else image
        
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        img_h, img_w = img.shape[:2]
        r = np.random.uniform(low=1, high=self.max_range)
        img = cv2.resize(img, (int(img_w//r), int(img_h//r)))
        img = cv2.resize(img, (img_w, img_h))

        if len(labels):
            labels["img"] = img
            return labels
        else:
            return img