from django.db import models

# Create your models here.
class Endpoint(models.Model):

    name = models.CharField(max_length=128)
    owner = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True, blank=True)


class MLAlgorithm(models.Model):
    
    name = models.CharField(max_length=128)
    description = models.TextField()
    code = models.TextField()
    version = models.CharField(max_length=128)
    owner = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True, blank=True)
    parent_endpoint = models.ForeignKey(Endpoint, on_delete=models.CASCADE)

    def __str__(self) -> str:
        return '%s | version: %s' % (self.name, self.version)


class MLAlgorithmStatus(models.Model):

    status = models.CharField(max_length=128)
    active = models.BooleanField()
    created_by = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True, blank=True)
    parent_mlalgorithm = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE, related_name="status")


class MLRequest(models.Model):

    input_data = models.ImageField(upload_to='./media/')
    full_response = models.TextField()
    response = models.TextField()
    feedback = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    parent_mlalgorithm = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE)
        

class ABTest(models.Model):
    title = models.TextField()
    created_by = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True, blank=True)
    ended_at = models.DateTimeField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True)

    parent_mlalgorithm_1 = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE, related_name="parent_mlalgorithm_1")
    parent_mlalgorithm_2 = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE, related_name="parent_mlalgorithm_2")
