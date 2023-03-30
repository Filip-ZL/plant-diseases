from apps.endpoints.models import Endpoint, MLAlgorithm, MLAlgorithmStatus

class MLRegistry:

    def __init__(self) -> None:
        self.endpoints = {}

    def add_algorithm(self, endpoint_name, algorithm_object, algorithm_name,
                      algorithm_status, algorithm_version, owner,
                      algorithm_description, algortihm_code):
        
        endpoint, _ = Endpoint.objects.get_or_create(name=endpoint_name, owner=owner)


        database_obj, algorithm_created = MLAlgorithm.objects.get_or_create(
            name=algorithm_name,
            description=algorithm_description,
            code=algortihm_code,
            version=algorithm_version,
            owner=owner,
            parent_endpoint=endpoint
        )

        if algorithm_created:
            status = MLAlgorithmStatus(status=algorithm_status,
                                       created_by=owner,
                                       parent_mlalgorithm=database_obj,
                                       active=True)
            
            status.save()

        self.endpoints[database_obj.id] = algorithm_object