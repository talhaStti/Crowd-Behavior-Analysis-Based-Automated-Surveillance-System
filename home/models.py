from django.db import models
from django.contrib.auth.models import User
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
import os
import subprocess

class UploadedVideos(models.Model):
    user  = models.ForeignKey(User,on_delete=models.CASCADE)
    video = models.FileField(upload_to='uploadedVids')
    classified = models.BooleanField(default=False)
    violent  = models.BooleanField(default=False)
    violentFramesDir = models.CharField(max_length=255, blank=True, null=True)
    isVoilentThisIteration = models.BooleanField(default=False)
    # we could use this approach but the problem will be synchronization because once violence is detected frames are saved and then the loop will reset
    # this will send almost never that violene was detected 
    duration  = models.DecimalField(default = 0, decimal_places = 2 ,max_digits = 6)
    progress  = models.DecimalField(default = 0, decimal_places = 2 ,max_digits = 6)





class CustomUser(models.Model):
    user = models.OneToOneField(User,on_delete=models.CASCADE)
    pfp  = models.ImageField(upload_to='pfp',null=True,blank=True)
