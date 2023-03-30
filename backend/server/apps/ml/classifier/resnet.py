from PIL import Image
from libs.architectures import Resnet
import numpy as np
from libs.misc import load_config
import keras

config = load_config(r"../../nn_config.yml")
class Classifier():

    def __init__(self) -> None:
        path_to_artifacts = r"../../models/trained_model.h5"
    
        self.resnet = Resnet(input_shape=config.get("shape"), n=config.get("stack_n"), **config)
        self.model = keras.models.load_model(path_to_artifacts)

    
    def preprocessing(self, input_data):
        input_img = Image.open(input_data).convert("RGB")
        input_img = input_img.resize(config.get("shape")[:-1])
        input_img = np.asarray(input_img).reshape([1,] + config.get("shape"))

        return input_img
    
    def predict(self, input_data):
        return self.model.predict(input_data, verbose=0)
    
    def postprocessing(self, input_data):
        pst = np.max(input_data) / np.sum(np.abs(input_data))
        label = config.get("classes")[np.argmax(input_data)]
        # if pst < 0.5:
        #     label = "Low_assurance"
        return {
            "probability": pst,
            "label": label,
            "status": "OK"
        }
    
    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {
                "status": "Error",
                "message": str(e)
            }
        
        return prediction
