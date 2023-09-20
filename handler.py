import torch
from ts.torch_handler.base_handler import BaseHandler
import logging
import os
import torchvision
import PIL
import numpy as np
import io
from torch import nn
import json

logger = logging.getLogger(__name__)

class ModelHandler(BaseHandler):
    def initialize(self, context):
        # context - It's JSON object
        properties = context.system_properties
        self.manifest = context.manifest

        # logger.info(properties)
        # logger.info(self.manifest)

        model_dir = properties.get('model_dir')

        mapping_file_path = os.path.join(model_dir, 'index_to_name.json')
        with open(mapping_file_path) as f:
            self.mapping = json.load(f)


        # Check if GPU is available
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        logger.info(f'Using device: {self.device}')

        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)

        logger.info(f'MODEL FILE: {model_file}')
        logger.info(f'MODEL PATH: {model_path}')


        self.model = torchvision.models.resnet50()
        self.model.fc = nn.Linear(2048, 1)
        self.model.eval()
        self.model.to(self.device)

        self.model.eval()
        # TODO: Please check how to load weight from saved weights path.
        # self.model.load_state_dict(model_path)

        logger.info('MODEL LOADED')


    def preprocess(self, data):
        images = np.zeros((1, 3, 256, 256))
        # images = []
        for row in data:
            image = row.get('data') or row.get('body')
            image = PIL.Image.open(io.BytesIO(image))
            image = image.resize((256, 256))
            image = np.array(image).reshape((3, 256, 256))
            images[0] = np.array(image, dtype=np.float32)
        temp = torch.from_numpy(images).float().to(self.device)
        return temp
    
    def inference(self, inputs):
        # logger.info(inputs.shape)
        outputs = self.model(inputs)
        probabilities = nn.functional.softmax(outputs)
        return probabilities
    

    def postprocess(self, data):
        logger.info('FROM POSTPROCESS>>>>>>')
        data = data.tolist()

        final_preds = [self.mapping[str(int(x[0]))] for x in data]
        logger.info(f'FINAL PREDS: {final_preds}')
        return final_preds

# torch-model-archiver --model-name test_model --version 1.0  --model-file model_files/model.pth --handler handler.py  --extra-files index_to_name.json --export-path model_store