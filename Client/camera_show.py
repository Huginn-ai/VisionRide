'''
This code implements a function to display all camera frames.
'''

import cv2
import numpy as np

from config import CLIP_AREA

def find_all_cameras():
    """Find all available cameras in the system"""
    camera_indexes = []
    for i in range(10):  # Check the first 10 possible camera indexes
        cap = cv2.VideoCapture(i-1)
        if cap.isOpened():
            camera_indexes.append(i-1)
            cap.release()
    return camera_indexes

def main():
    # Get all available cameras
    camera_indexes = find_all_cameras()
    
    if not camera_indexes:
        print("No camera devices found")
        return
    
    # Create a list of camera capture objects
    caps = [cv2.VideoCapture(idx) for idx in camera_indexes]
    
    try:
        while True:
            frames = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    # Get frame dimensions
                    height, width, _ = frame.shape
                    # Define cropping area
                    x_start = int(width * CLIP_AREA[0])
                    y_start = int(height * CLIP_AREA[1])
                    x_end = int(width * CLIP_AREA[2])
                    y_end = int(height * CLIP_AREA[3])
                    # Crop the frame
                    cropped_frame = frame[y_start:y_end, x_start:x_end]
                    # Add camera label
                    cv2.putText(cropped_frame, f"Camera {camera_indexes[i]}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
                    frames.append(cropped_frame)
                
            if frames:
                # Calculate grid layout
                n = len(frames)
                cols = int(np.ceil(np.sqrt(n)))
                rows = int(np.ceil(n / cols))
                
                # Resize all frames to the same dimensions
                cell_width = 640
                cell_height = 480
                resized_frames = [cv2.resize(frame, (cell_width, cell_height)) 
                                for frame in frames]
                
                # Create a blank canvas
                canvas = np.zeros((cell_height * rows, cell_width * cols, 3), 
                                dtype=np.uint8)
                
                # Place frames onto the canvas
                for idx, frame in enumerate(resized_frames):
                    i, j = divmod(idx, cols)
                    y1 = i * cell_height
                    y2 = y1 + cell_height
                    x1 = j * cell_width
                    x2 = x1 + cell_width
                    canvas[y1:y2, x1:x2] = frame
                
                # Display the merged frame
                cv2.imshow('All Cameras', canvas)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Release resources
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
