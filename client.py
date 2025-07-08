import os
import cv2
import tritonclient.grpc as grpcclient
import numpy as np

CLASS_LABELS = ["Black Rot", "Healthy", "Scab"]

def prepare_input(image_path: str, input_size=(640, 640)) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image from: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)  # HWC to CHW
    return image[np.newaxis, :, :, :]  # Add batch dimension

def main():
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    image_dir = "test"
    for file_name in os.listdir(image_dir):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(image_dir, file_name)
        input_tensor = prepare_input(image_path)

        inputs = [grpcclient.InferInput("images", input_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_tensor)

        outputs = [grpcclient.InferRequestedOutput("final_output")]

        response = client.infer(
            model_name="ensemble",
            inputs=inputs,
            outputs=outputs
        )

        # Get prediction output
        prediction = response.as_numpy("final_output")  # shape: (3,)
        pred_class = int(np.argmax(prediction))
        pred_label = CLASS_LABELS[pred_class]
        print(f"Image: {file_name} → Predicted class: {pred_label} (index: {pred_class})")

if __name__ == "__main__":
    main()

