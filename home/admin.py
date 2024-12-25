from django.contrib import admin
from .models import UploadedVideos,CustomUser
# Register your models here.
admin.site.register(UploadedVideos)
admin.site.register(CustomUser)
