from pathlib import Path
from ball_predicter import BallPredicter
from camera import VideoCamera

# Locate dependencies
MODEL_PATH = Path("./models")
MODEL_NAME = "ball-predicter-model.pkl"

CAM_SOURCE = 0 # default capturing device

# GUI Size
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

if __name__ == '__main__':
    
    model = BallPredicter(MODEL_PATH, MODEL_NAME)
    camera = VideoCamera(CAM_SOURCE, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    while not camera.is_interrupted():
        
        camera.update_frames()
          
        main_frame = camera.get_main_frame()
        pred_frame = camera.get_pred_frame()

        predicted_class = model.get_predicted_class(pred_frame)

        camera.draw_prediction_box(main_frame)
        camera.display_class(main_frame, predicted_class)
        camera.display_window(main_frame)
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
