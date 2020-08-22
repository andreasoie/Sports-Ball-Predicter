from fastai.vision import *
from fastai.metrics import error_rate
import numpy as np
import warnings

# Ignore some backend installation åath-errors
warnings.filterwarnings("ignore")

class BallPredicter():
    
    # Acceptable values
    THRESHOLD = 0.8
    
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self._set_learner(self.model_path, self.model_name)
        defaults.device = torch.device("cpu") # Use CPU for inference
        
    def _set_learner(self, model_path, model_name):
        """ inits a Learner object with a given model """
        self.learner = load_learner(model_path, model_name)
        
    def get_predicted_class(self, img):
        """Returns predicted class for a single object"""
        target_img = self._format_image(img)
        pred_class, pred_index, outputs = self.learner.predict(target_img)
        # Return valid results
        if outputs[pred_index] > float(BallPredicter.THRESHOLD):
            return str(pred_class)
        return ""
        
    def _format_image(self, img):
        """typechange -> numpy.ndarray to fastai.vision.Image"""
        return Image(torch.tensor(np.ascontiguousarray(np.flip(img, 2)).transpose(2,0,1)).float()/255)
        