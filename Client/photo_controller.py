import cv2

from config import CLIP_AREA, CAMERA_INDEX

class PhotoController:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if self.cap.isOpened():
            print(f'Cap {CAMERA_INDEX} available.')
            self.is_available = True
        else:
            print(f'Unable to open camera {CAMERA_INDEX}.')
            self.is_available = False
        

    def get_photo(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Error: Unable to read frame from camera!")
            self.reset_camera()
            return None
        # Get frame dimensions
        height, width, _ = frame.shape
        # Define cropping area
        x_start = int(width * CLIP_AREA[0])
        y_start = int(height * CLIP_AREA[1])
        x_end = int(width * CLIP_AREA[2])
        y_end = int(height * CLIP_AREA[3])
        # Crop the frame
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        # Encode the cropped image in JPG format
        success, jpg_data = cv2.imencode('.jpg', cropped_frame)
        if not success:
            raise ValueError("Unable to encode image to JPG format")
        # Return the byte stream of the JPG data
        return jpg_data.tobytes()
    
    def reset_camera(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if self.cap.isOpened():
            print("Camera reinitialized successfully.")
        else:
            print("Error: Unable to reinitialize the camera.")

    def __del__(self):
        self.cap.release()
