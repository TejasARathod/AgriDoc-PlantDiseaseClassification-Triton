import os
import numpy as np
import math
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.mask_alpha = 0.5
        self.conf_threshold = 0.3
        self.iou_threshold = 0.65
        self.num_classes = 1
        self.num_masks = 32
        self.input_height = 640
        self.input_width = 640
        self.class_names = ["Leaf"]
        self.colors = {cls: [0, 255, 0] for cls in self.class_names}
        self.output_dir = "cropped_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def execute(self, requests):
        responses = []
        for request in requests:
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            output0 = pb_utils.get_input_tensor_by_name(request, "output0").as_numpy()
            output1 = pb_utils.get_input_tensor_by_name(request, "output1").as_numpy()
            print(f"[DEBUG] output0 shape: {output0.shape}")
            print(f"[DEBUG] output1 shape: {output1.shape}")

            image_np = image_tensor.as_numpy()
            print(f"[DEBUG] Raw image tensor shape: {image_np.shape}")

            if image_np.ndim == 4:
                image = image_np[0]
            elif image_np.ndim == 3:
                image = image_np
            else:
                raise ValueError(f"Unexpected image tensor shape: {image_np.shape}")

            print(f"[DEBUG] After batch handling: {image.shape}")

            image = (image * 255).astype(np.uint8)
            print(f"[DEBUG] After scaling: {image.shape}")

            image = np.transpose(image, (1, 2, 0))
            print(f"[DEBUG] After transpose: {image.shape}")

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            print(f"[DEBUG] After BGR conversion: {image.shape}")

            self.img_height, self.img_width = image.shape[:2]
            boxes, scores, class_ids, mask_pred = self.process_box_output(output0)
            mask_maps = self.process_mask_output(mask_pred, output1, boxes)

            # If no valid boxes found, send dummy output so ensemble step doesn't deadlock
            if len(boxes) == 0:
                dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
                output_tensor = pb_utils.Tensor("resnet_input", dummy)
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
                continue

            for i, (box, cid, score, mask) in enumerate(zip(boxes, class_ids, scores, mask_maps)):
                x1, y1, x2, y2 = box.astype(int)
                cropped_img = np.zeros_like(image)
                roi = image[y1:y2, x1:x2]
                roi_mask = mask[y1:y2, x1:x2]
                cropped_img[y1:y2, x1:x2] = roi * roi_mask[:, :, np.newaxis]
                class_name = self.class_names[cid]
                class_dir = os.path.join(self.output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                crop_path = os.path.join(class_dir, f"crop_{i}.png")
                cv2.imwrite(crop_path, cropped_img)
                print(f"Saved crop: {crop_path}")

                # Resize and normalize for ResNet classifier
                resized_crop = cv2.resize(cropped_img, (224, 224))
                resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                normalized_crop = resized_crop.astype(np.float32) / 255.0
                transposed_crop = np.transpose(normalized_crop, (2, 0, 1))  # CHW

                # Add batch dimension to shape (1, 3, 224, 224)
                batched_crop = transposed_crop[np.newaxis, :, :, :]

                # Create tensor for next model
                output_tensor = pb_utils.Tensor("resnet_input", batched_crop)
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

            return responses


    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def nms(self, boxes, scores, iou_threshold):
        sorted_indices = np.argsort(scores)[::-1]
        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
            keep_indices = np.where(ious < iou_threshold)[0]
            sorted_indices = sorted_indices[keep_indices + 1]
        return keep_boxes

    def compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - intersection
        return intersection / union

    def rescale_boxes(self, boxes, input_shape, image_shape):
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        return boxes

    def process_box_output(self, box_output):
        predictions = np.squeeze(box_output).T  # (N, 37)
        objectness = predictions[:, 4]
        predictions = predictions[objectness > self.conf_threshold, :]
        scores = objectness[objectness > self.conf_threshold]
        if len(scores) == 0:
            return np.empty((0, 4)), np.array([]), np.array([]), np.empty((0, 32))

        box_preds = predictions[:, :5]
        mask_preds = predictions[:, 5:]
        class_ids = np.zeros_like(scores, dtype=int)
        boxes = self.extract_boxes(box_preds)
        indices = self.nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_preds[indices]

    def extract_boxes(self, box_predictions):
        boxes = box_predictions[:, :4]
        boxes = self.rescale_boxes(boxes, (self.input_height, self.input_width), (self.img_height, self.img_width))
        boxes = self.xywh2xyxy(boxes)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)
        return boxes

    def process_mask_output(self, mask_predictions, mask_output, boxes):
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)
        num_mask, mask_height, mask_width = mask_output.shape
        masks = self.sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))
        scale_boxes = self.rescale_boxes(boxes, (self.input_height, self.input_width), (mask_height, mask_width))

        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))

        for i in range(len(scale_boxes)):
            sx1, sy1, sx2, sy2 = map(int, map(np.floor, scale_boxes[i]))
            x1, y1, x2, y2 = map(int, map(np.floor, boxes[i]))
            scale_crop_mask = masks[i][sy1:sy2, sx1:sx2]
            crop_mask = cv2.resize(scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            crop_mask = cv2.blur(crop_mask, blur_size)
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

