from django.test import TestCase
# from PIL import Image
from apps.ml.classifier.resnet import Classifier
from apps.ml.registry import MLRegistry
import inspect

class MLTests(TestCase):

    def test_algorithm(self):
    
        ml_alg = Classifier()
        response = ml_alg.compute_predicition(r"./apps/ml/sample.jpg")
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('ship', response['label'])

    
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "cifar10_classifier"
        algorithm_obj  =  Classifier()
        algorithm_name = "Resnet18 CNN"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "filip"
        algorithm_description = "Classifier for cifar 10 ds"
        algorithm_code = inspect.getsource(Classifier)
        registry.add_algorithm(endpoint_name, algorithm_obj, algorithm_name, algorithm_status,
                               algorithm_version, algorithm_owner, algorithm_description, algorithm_code)
        
        self.assertEqual(len(registry.endpoints), 1)