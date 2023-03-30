from django.conf.urls import include
from django.urls import re_path as url
from rest_framework.routers import DefaultRouter

from apps.endpoints.views import EndpointViewSet, MLAlgorithmViewSet, \
                                 MLRequestViewSet, MLAlgorithmStatusViewSet

router = DefaultRouter(trailing_slash=False)
router.register(r"endpoints", 
                EndpointViewSet,
                basename="endpoints")
router.register(r"mlalgorithms", 
                MLAlgorithmViewSet,
                basename="mlalgorithms")
router.register(r"mlalgorithmstatuses", 
                MLAlgorithmStatusViewSet,
                basename="mlalgorithmstatuses")
router.register(r"mlrequests", 
                MLRequestViewSet,
                basename="mlrequests")

urlpatterns = [
    url(r"^api/v1/", include(router.urls)),
]
