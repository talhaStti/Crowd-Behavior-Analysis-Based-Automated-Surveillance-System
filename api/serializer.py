from rest_framework import serializers
from home.models import UploadedVideos
import base64
import base64
import os
from django.conf import settings



class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedVideos
        fields = ('id','classified','violent')


class FullVideoInfo(serializers.ModelSerializer):
    video_data = serializers.SerializerMethodField()
    violent_frames_data = serializers.SerializerMethodField()
    class Meta:
        model = UploadedVideos
        fields = ('id', 'classified', 'violent', 'violentFramesDir', 'video_data', 'violent_frames_data')
    def get_video_data(self, obj):
        video_path = os.path.join(settings.MEDIA_ROOT, obj.video.path)
        with open(video_path, 'rb') as video_file:
            video_binary_data = base64.b64encode(video_file.read())
        return video_binary_data.decode('utf-8')

    def get_violent_frames_data(self, obj):
        # Load all images from the violentFramesDir as binary data
        frames_dir = os.path.join(settings.MEDIA_ROOT, obj.violentFramesDir)
        if frames_dir:
            violent_frames_data = []
            for filename in os.listdir(frames_dir):
                file_path = os.path.join(frames_dir, filename)
                with open(file_path, 'rb') as frame_file:
                    frame_binary_data = base64.b64encode(frame_file.read())
                    violent_frames_data.append({
                        'filename': filename,
                        'data': frame_binary_data.decode('utf-8')
                    })
            return violent_frames_data
        else:
            return None