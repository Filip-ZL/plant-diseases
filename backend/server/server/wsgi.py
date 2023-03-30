"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

import inspect
from apps.ml.registry import MLRegistry
from apps.ml.classifier.resnet import Classifier

try:
    registry = MLRegistry()
    resnet = Classifier()

    registry.add_algorithm(endpoint_name="cifar10_classifier",
                           algorithm_object=resnet,
                           algorithm_name="Resnet classifier",
                           algorithm_status="dev",
                           algorithm_version="0.0.2",
                           owner="fj",
                           algorithm_description="Resnet for cifar10 classifier",
                           algortihm_code=inspect.getsource(Classifier))
    
    registry.add_algorithm(endpoint_name="cifar10_classifier",
                           algorithm_object=resnet,
                           algorithm_name="Resnet classifier",
                           algorithm_status="production",
                           algorithm_version="0.0.1",
                           owner="fj",
                           algorithm_description="Resnet for cifar10 classifier",
                           algortihm_code=inspect.getsource(Classifier))

except Exception as e:
    print("Exception occured while loading algorithms to the registry", str(e))
