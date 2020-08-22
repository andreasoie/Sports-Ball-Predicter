import numpy as np
import cv2


class VideoCamera():
    
    WINDOW_TITLE = "THE BALL PREDICTOR"
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX 
    TEXT_SCALE = 1 #px
    LINE_THICKNESS = 2 #px
    PADDING = 30
    BOX_COLOR = (0, 255, 0) #BGR
    
    def __init__(self, SRC, WINDOW_WIDTH, WINDOW_HEIGHT):
        self.cap = cv2.VideoCapture(SRC)
        self.WINDOW_WIDTH = WINDOW_WIDTH
        self.WINDOW_HEIGHT = WINDOW_HEIGHT
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.WINDOW_HEIGHT)
        
    def update_frames(self):
        """ reads frame from source """
        _, frame = self.cap.read()
        self.set_main_frame(frame)
        self.set_pred_frame(frame)
    
    def set_main_frame(self, frame):
        """ uses all the pixels in the frame """
        self.main_frame = frame
        
    def get_main_frame(self):
        return self.main_frame
        
    def set_pred_frame(self, frame):
        """ creates a region of interest """
        self.pred_frame = self._resize_frame(frame)
        
    def get_pred_frame(self):
        return self.pred_frame
    
    def draw_prediction_box(self, frame):
        x_min, y_min, x_max, y_max = self._get_prediction_coords()
        self.set_main_frame(cv2.rectangle(frame, 
                                          (x_min, y_min), 
                                          (x_max, y_max), 
                                          VideoCamera.BOX_COLOR, 
                                          VideoCamera.LINE_THICKNESS)) 
    
    def display_class(self, frame, pred_class_name):
        x_min, _, _, y_max = self._get_prediction_coords()
        cv2.putText(frame, 
                    pred_class_name, 
                    (x_min + VideoCamera.PADDING, y_max - VideoCamera.PADDING), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    VideoCamera.TEXT_SCALE, 
                    VideoCamera.BOX_COLOR, 
                    VideoCamera.LINE_THICKNESS, 
                    cv2.LINE_AA) 
        
    def display_window(self, frame):
        cv2.imshow(VideoCamera.WINDOW_TITLE, frame)

    def is_interrupted(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False
    
    def _get_prediction_coords(self):
        """ For instance: top left of a box """
        ROWS = 2
        COLS = 2
        x_min = int(VideoCamera.PADDING)
        y_min = int(VideoCamera.PADDING)
        x_max = int(self.WINDOW_WIDTH - (self.WINDOW_WIDTH / COLS) + VideoCamera.PADDING)
        y_max = int(self.WINDOW_HEIGHT - (self.WINDOW_HEIGHT / COLS) + VideoCamera.PADDING)
        return (x_min, y_min, x_max, y_max)
    
    def _resize_frame(self, frame):
        x_min, y_min, x_max, y_max = self._get_prediction_coords()
        return frame[y_min:y_max, x_min:x_max]