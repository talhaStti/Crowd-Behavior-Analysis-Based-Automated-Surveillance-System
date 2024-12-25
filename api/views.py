from django.shortcuts import render
from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.decorators import api_view,authentication_classes, permission_classes
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.serializers import AuthTokenSerializer
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
import threading
from home.models import UploadedVideos
from home.prediction import Pred, predict_single_frame
from .serializer import VideoSerializer,FullVideoInfo





@api_view(['POST'])
def register(request):
    if request.method == 'POST':
        email = request.data.get('email')
        username = request.data.get('username')
        password = request.data.get('password')
        password2 = request.data.get('password2')
        print(email,username,password,password2)
        try:
            existing_user = User.objects.get(email=email)
            return Response({"error":"Email already in use"}, status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            pass
        if email and username and password and password2:
            if password == password2:
                user_data = {
                        'username': username,  
                        'password': password,  
                        'email': email,  
                        }
                user = User.objects.create_user(**user_data)
                token, created = Token.objects.get_or_create(user=user)
                return Response({'token': token.key})
            else:
                return Response({"error":"passwords donot match"},status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({"error":"incomplete information"}, status=status.HTTP_400_BAD_REQUEST)
        
@api_view(['POST'])
def login(request):
    if request.method == 'POST':
        serializer = AuthTokenSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            token, created = Token.objects.get_or_create(user=user)
            if created:
                token.delete()
                token ,created = Token.objects.get_or_create(user = user)
            return Response({
                'token': token.key,
                },)            
        else:
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
def logout(request):
    key = request.data.get("token")
    if key:
        try:
            token = Token.objects.get(key=key)
            token.delete()
            return Response({"success": "Logged out successfully"})

        except Token.DoesNotExist:
            return Response({"error": "Token not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response({"error": "Token not provided"}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def classify(request):
    if request.method == 'POST':
        video = request.FILES['Insert']
        video.name = f"{request.user.username}_{video.name.split()[0]}"
        videoObj = UploadedVideos.objects.create(user = request.user, video = video)
        videoObj.save()
        context = {"video": videoObj}
        print(videoObj.video.url)
        t2 = threading.Thread(target=Pred, args=[videoObj])
        return Response({"id":videoObj.id})
    

@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def assign_value(request,id):
    try:
        video = UploadedVideos.objects.get(id = id)
    except:
        return Response({"error": "video doesnot exist"})  
    val = "Not classified yet" 
    if video.classified:
        if video.violent:
            val = "violent"
        else:
            val = "Normal"
    data = {'value': val}
    # print(video.classified)
    return Response(data)



@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def logs(request):
    user = request.user
    videos = UploadedVideos.objects.filter(user = user)
    serializer  = VideoSerializer(videos,many=True)
    return Response(serializer.data)



@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def viewVideo(request,id):
    user = request.user
    print(user)
    try:
        video = UploadedVideos.objects.get(user = user,id=id)
    except:
        return Response({"error":"couldnot find video"})
    print(video)
    serializer  =FullVideoInfo(video,many=False)
    return Response(serializer.data)



@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def live_classification(request):
    imageData = request.POST['image']
    predict_single_frame(imageData)