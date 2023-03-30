from django.shortcuts import render
from django.db import transaction

# Create your views here.

from rest_framework import viewsets
from rest_framework import mixins
from rest_framework.exceptions import APIException
from rest_framework import views, status
from rest_framework.response import Response
from apps.endpoints.models import Endpoint, MLAlgorithm, MLAlgorithmStatus, MLRequest, ABTest
from apps.endpoints.serializers import EndpointSerializer, MLAlgorithmSerializer, MLAlgorithmStatusSerializer, \
                                       MLRequestSerializer, ABTestSerializer
from server.wsgi import registry
from django.db.models import F

import numpy as np
import datetime

class EndpointViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin,
                      viewsets.GenericViewSet):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()


class MLAlgorithmViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin,
                         viewsets.GenericViewSet):
    serializer_class = MLAlgorithmSerializer
    queryset = MLAlgorithm.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm=instance.parent_mlalgorithm,
                                                    created_at__lt=instance.created_at,
                                                    active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
        MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])


class MLAlgorithmStatusViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin,
                               viewsets.GenericViewSet, mixins.CreateModelMixin):
    serializer_class = MLAlgorithmStatusSerializer
    queryset = MLAlgorithmStatus.objects.all()
    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                deactivate_other_statuses(instance)
        except Exception as e:
            raise APIException(str(e))


class MLRequestViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin,
                       viewsets.GenericViewSet, mixins.UpdateModelMixin):
    serializer_class = MLRequestSerializer
    queryset = MLRequest.objects.all()


class PredictView(views.APIView):


    # queryset = MLRequest.objects.all()
    serializer_class = MLRequestSerializer

    def post(self, request, endpoint_name, format=None):
        
        algorithm_status = self.request.query_params.get("status", "production")
        algorithm_version = self.request.query_params.get("version")

        algs = MLAlgorithm.objects.filter(parent_endpoint__name=endpoint_name,
                                          status__status=algorithm_status,
                                          status__active=True)
        
        if algorithm_version is not None:
            algs = algs.filter(version=algorithm_version)

        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available."},
                status=status.HTTP_400_BAD_REQUEST)
        
        if len(algs) != 1 and algorithm_status != "ab_testing":
            return Response(
                {"status": "Error", "message": "ML algorithm selection is ambigous. Please specify version"}
            )
        
        alg_index = 0
        if algorithm_status == "ab_testing":
            alg_index = 0 if np.random.rand() < 0.5 else 1

        print(registry.endpoints)
        algorithm_object = registry.endpoints[algs[alg_index].id]
        prediction = algorithm_object.compute_prediction(request.data['input_data'])

        label = prediction["label"] if "label" in prediction else "error"
        ml_request = MLRequest(
            input_data=request.data['input_data'],
            full_response=prediction,
            response=label,
            feedback="None",
            parent_mlalgorithm=algs[alg_index]
        )
        ml_request.save()

        prediction["request_id"] = ml_request.id
        
        return Response(prediction)
    

class ABTestViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin,
                    mixins.CreateModelMixin, mixins.UpdateModelMixin,
                    viewsets.GenericViewSet):
    
    serializer_class = ABTestSerializer
    queryset = ABTest.objects.all()

    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance  = serializer.save()

                status_1 = MLAlgorithmStatus(status="ab_testing",
                                             created_by=instance.created_by,
                                             parent_mlalgorithm= instance.parent_mlalgorithm_1,
                                             active=True)
                
                status_1.save()

                status_2 = MLAlgorithmStatus(status="ab_testing",
                                             created_by=instance.created_by,
                                             parent_mlalgorithm= instance.parent_mlalgorithm_2,
                                             active=True)
                
                status_2.save()

                deactivate_other_statuses(status_2)
        except Exception as e:
            raise APIException(str(e))
        

class StopABTestView(views.APIView):

    def get(self, request, ab_test_id, format=None):
        
        try:
            ab_test = ABTest.objects.get(pk=ab_test_id)

            if ab_test.ended_at is not None:
                return Response({"message": "AB test finished"})
            
            date_now = datetime.datetime.now()

            all_responses_1 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_1,
                                                       created_at__gt=ab_test.created_at,
                                                       created_at__lt=date_now).count()
            
            correct_responses_1 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_1,
                                                           created_at__gt=ab_test.created_at,
                                                           created_at__lt=date_now,
                                                           response=F('feedback')).count()
            
            accuracy_1 = correct_responses_1 / float(all_responses_1)

            all_responses_2 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_2,
                                                       created_at__gt=ab_test.created_at,
                                                       created_at__lt=date_now).count()
            
            correct_responses_2 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_2,
                                                           created_at__gt=ab_test.created_at,
                                                           created_at__lt=date_now,
                                                           response=F('feedback')).count()
            
            accuracy_2 = correct_responses_2 / float(all_responses_2)

            alg_id_1, alg_id_2 = ab_test.parent_mlalgorithm_1, ab_test.parent_mlalgorithm_2

            if accuracy_1 < accuracy_2:
                alg_id_1, alg_id_2 = alg_id_2, alg_id_1

            status_1 = MLAlgorithmStatus(status="production",
                                         created_by=ab_test.created_by,
                                         parent_mlalgorithm=alg_id_1,
                                         active=True)
            status_1.save()
            deactivate_other_statuses(status_1)

            status_2 = MLAlgorithmStatus(status="dev",
                                         created_by=ab_test.created_by,
                                         parent_mlalgorithm=alg_id_2,
                                         active=True)
            status_2.save()
            deactivate_other_statuses(status_2)

            summary = "Algorithm 1 has accuracy: {:.3f} and algorithm 2 has accuracy: {:.3f}".format(accuracy_1, accuracy_2)

            ab_test.ended_at = date_now
            ab_test.summary = summary
            ab_test.save()

        except Exception as e:
            return Response({"status": "Error", "message": str(e)},
                            status=status.HTTP_400_BAD_REQUEST
                            )
        return Response({"message": "AB Test finished", "results": summary})