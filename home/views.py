import base64
from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.templatetags.static import static
from django.views.decorators.csrf import csrf_exempt
from allauth.account.forms import LoginForm
from django.http import FileResponse, Http404

import json
import os
from django.urls import reverse
from django.conf import settings
from django.http import JsonResponse
import execjs
import threading
# Create your views here.
from .prediction import Pred, livePred
from django.http import StreamingHttpResponse
import threading
from .send_Alert import send_alert
import uuid
import cv2
from io import BytesIO
import shutil
import datetime
import imageio
import time

BASE_DIR = settings.BASE_DIR

last_alert_time = 0
alert_threshold_seconds = 60  # Minimum seconds between alerts
violence_detection_threshold = 10  # Number of times violence must be detected to trigger an alert
violence_detection_count = 0  # Current count of consecutive violence detections

from .models import UploadedVideos,CustomUser
 
"""
    make a clear logs view nd delete single log view handle video deleteion manually 
    progress is being calculated for each video we could support multiple videos at once that is not the problem
    the problem will be synchronization for each loop the results are very real time 
    so i guess we might not run in to the synchronizatin issue at all

"""
def deleteFromStorage(filePaths:list[str] ,violentDirs:list[str]) -> None:
    for path in filePaths:
        if os.path.exists(path):
            os.remove(path)
    for dir in violentDirs:
        path = os.path.join(settings.MEDIA_ROOT,dir)
        try:
            shutil.rmtree(path)
            print(f"Directory '{dir}' successfully removed.")
        except FileNotFoundError:
            print(f"Directory '{dir}' not found.")
        except Exception as e:
            print(f"Error: {e}")
    return

def clearLogs(request):
    #delete all database entries saving violentDir names and videos name use a thread to delete all that data to but asynchronously 
    if not request.user.is_authenticated:
        return redirect('login_view')
    user =  request.user
    videos = UploadedVideos.objects.filter(user = user)
    filePaths = [video.video.path for video in videos]
    violentDirs = [video.violentFramesDir for video in videos if video.violent]
    videos.delete()
    t = threading.Thread(target=deleteFromStorage, args=[filePaths,violentDirs])
    t.start()
    return redirect('logs_view')
    
def clear_all_anonymous_logs(request):
    user, _ = User.objects.get_or_create(username='anonymous')
    videos = UploadedVideos.objects.filter(user=user)
    filePaths = [video.video.path for video in videos]
    violentDirs = [video.violentFramesDir for video in videos if video.violent]
    videos.delete()
    t = threading.Thread(target=deleteFromStorage, args=[filePaths, violentDirs])
    t.start()
    return redirect('anonymous_logs')

def deleteLog(request,id):
    if not request.user.is_authenticated:
        return redirect('login_view')
    user =  request.user
    try:
        video = UploadedVideos.objects.get(user = user,id=id)
    except:
        return redirect('logs_view')
    filePath = [video.video.path ]
    if video.violent:
        violentDir = [video.violentFramesDir]
    else:
        violentDir = []
    video.delete()
    t = threading.Thread(target=deleteFromStorage, args=[filePath,violentDir])
    t.start()
    return redirect('logs_view')

def delete_anonymous_log(request, id):
    user, _ = User.objects.get_or_create(username='anonymous')
    try:
        video = UploadedVideos.objects.get(user=user, id=id)
    except UploadedVideos.DoesNotExist:
        return redirect('anonymous_logs')
    filePath = [video.video.path]
    violentDir = [video.violentFramesDir] if video.violent else []
    video.delete()
    t = threading.Thread(target=deleteFromStorage, args=[filePath, violentDir])
    t.start()
    return redirect('anonymous_logs')

def landing(request):
    return render(request,'LandingPage.html')

def updateInfo(request):
    if not request.user.is_authenticated:
        return redirect('login_view')
    user = request.user
    username = request.POST['username']
    f_name = request.POST['F_name']
    L_name = request.POST['L_name']
    if username:
        user.username = username
    if f_name:
        user.first_name = f_name
    if L_name:
        user.last_name = L_name
    user.save()
    return redirect('profile')

def profile(request):
    if not request.user.is_authenticated:
        return redirect('login_view')
    user = request.user
    try:
        pfp = CustomUser.objects.get(user = request.user).pfp
    except:
        pfp = None

    return render (request,'Profile.html',{
        'username':user.username,
        'pfp':pfp,
        'user':user
    })

def updatePfp(request):
    if not request.user.is_authenticated:
        return redirect('login_view')
    try:
        user = CustomUser.objects.get(user = request.user)
    except:
        user = CustomUser.objects.create(user = request.user)
    
    pfp = request.FILES['pfp']
    print(pfp)
    user.pfp  = pfp
    user.save()
    return redirect('dashboard')

def register(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        username = request.POST['username']
        f_name = request.POST['F_name']
        L_name = request.POST['L_name']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']

        # validate user information
        if len(username) > 10:
            messages.error(request, 'Username must be under 10 characters')
            return render(request, 'Register.html')
        if not username.isalnum():
            messages.error(request, 'Username must be alphanumeric')
            return render(request, 'Register.html')
        if password != password2:
            messages.error(request, 'Passwords do not match')
            return render(request, 'Register.html')

        # save the user information
        myuser = User.objects.create_user(
            username, email, password, first_name=f_name, last_name=L_name)
        # myuser.name = name
        # myuser.organization = organization
        # myuser.phone = phone
        myuser.save()
        messages.success(request, 'Registration is successful')
        return redirect('login_view')

    else:
        print('here')
        return render(request, 'Register.html')

def Del():
    fs = FileSystemStorage
    dir = rf' {BASE_DIR}\demo\media'
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        fs.delete(name=file_path)
    file_path = None
    return


# def register(request):
#     return render(request, 'Register.html')


def login(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = LoginForm(request.POST or None)
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        print(user)
        if user is not None:
            auth_login(request, user)
            # messages.success(request, 'login is successful')
            return redirect('dashboard')
            # return render(request,'Dashboard.html',{'username':username})
        else:
            messages.error(request, 'Invalid username or password')
        return redirect('/accounts/google/login')
    else:
        return redirect('/accounts/google/login')





def dashboard(request):
    if not request.user.is_authenticated:
        return redirect('login_view')
    user  =request.user
    # video = {"id":0}
    # print(video)
    try:
        pfp = CustomUser.objects.get(user = user).pfp
    except:
        return render(request, 'Dashboard.html', {'username': request.user.username})

   
    return render(request, 'Dashboard.html', {'username': request.user.username,
                                              'pfp':pfp,
                                              })

# this function will return video and all the violent frames if any 
@login_required()
def viewVideo(request, id):
    try:
        video = UploadedVideos.objects.get(id = id)
    except:
        return redirect("dashboard")
    violentFrames = []
    if video.violent and video.violentFramesDir:
        dir_path = os.path.join(settings.MEDIA_ROOT, video.violentFramesDir)
        violentFrames = os.listdir(dir_path)
    print(violentFrames)
    return render(request,"viewVideo.html",{
        "username":request.user.username,
        "video":video,
        "violentFrames":violentFrames
        })

    
def live_classification(request):
    user = request.user
    imageData = []
    imageData.append( request.POST['image0'])
    imageData.append( request.POST['image1'])
    imageData.append(request.POST['image2'])
    imageData.append(request.POST['image3'])
    imageData.append(request.POST['image4'])
    imageData.append(request.POST['image5'])
    imageData.append(request.POST['image6'])
    imageData.append(request.POST['image7'])
    imageData.append(request.POST['image8'])
    imageData.append(request.POST['image9'])
    imageData.append( request.POST['image10'])
    imageData.append( request.POST['image11'])
    imageData.append( request.POST['image12'])
    imageData.append( request.POST['image13'])
    imageData.append( request.POST['image14'])
    imageData.append( request.POST['image15'])
    response = livePred(imageData)
    if response == 'Violent':
        t = threading.Thread(target=send_alert, args=[user.email])
        t.start()
    print(response)
    return HttpResponse( response)



def live(request):
   if not request.user.is_authenticated:
        return redirect('login_view')
   user  = request.user
   try:
        pfp = CustomUser.objects.get(user = user).pfp
   except:
        return render(request, 'live.html', {'username': request.user.username})
   return render(request, 'live.html', {'username': request.user.username,
                                              'pfp':pfp})

def send_json(request):
    with open(rf" {BASE_DIR}\demo\static\css\logs.json", "r") as f:
        data = json.load(f)
    return JsonResponse(data, safe=False)


def logs(request):
    if not request.user.is_authenticated:
        return redirect('login_view')
    user  = request.user   
    data = []
    videos = UploadedVideos.objects.filter(user = user)
    for video in videos:
        obj = {}
        obj['video'] = video
        if video.violent:
            dir_path = os.path.join(settings.MEDIA_ROOT, video.violentFramesDir)
            obj['violentVid']  = f'/media/violent_{video.id}/violent.mp4'
        data.append(obj)

    try:
        pfp = CustomUser.objects.get(user = user).pfp
    except:
        return render(request, 'logs.html', {'username': user.username,"data":data})
        
    return render(request, 'logs.html', {'username': user.username,"data":data,'pfp':pfp})



    try:
        video = UploadedVideos.objects.get(id = id)
    except:
        return redirect("dashboard")
    violentFrames = []
    if video.violent and video.violentFramesDir:
        dir_path = os.path.join(settings.MEDIA_ROOT, video.violentFramesDir)
        violentFrames = os.listdir(dir_path)
    print(violentFrames)
    return render(request,"viewVideo.html",{
        "username":request.user.username,
        "video":video,
        "violentFrames":violentFrames
        })


def get_video_base64(video_path):
    """Encode video file to base64."""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

def anonymous_logs(request):
    user, _ = User.objects.get_or_create(username='anonymous')
    videos = UploadedVideos.objects.filter(user=user)
    data = []
    for video in videos:
        video_data = {
            'id': video.id,
            'stream_url': request.build_absolute_uri(reverse('stream_video', args=[video.id])),
            'violent': video.violent,
        }

        if video.violent:
            video_data['violent_stream_url'] = request.build_absolute_uri(reverse('stream_violent_video', args=[video.id]))

        data.append(video_data)

    return JsonResponse(data, safe=False)

def stream_video(request, video_id):
    """Stream a video file directly to the client."""
    try:
        video = UploadedVideos.objects.get(id=video_id)
        file_path = video.video.path
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), content_type='video/mp4')
        else:
            raise Http404("Video not found.")
    except UploadedVideos.DoesNotExist:
        raise Http404("Video not found.")

def stream_violent_video(request, video_id):
    """Stream a violent video file directly to the client if it exists."""
    try:
        video = UploadedVideos.objects.get(id=video_id)
        if video.violent:
            violent_video_path = os.path.join(settings.MEDIA_ROOT, f'violent_{video.id}/violent.mp4')
            if os.path.exists(violent_video_path):
                return FileResponse(open(violent_video_path, 'rb'), content_type='video/mp4')
            else:
                raise Http404("Violent video not found.")
        else:
            raise Http404("No violent video available.")
    except UploadedVideos.DoesNotExist:
        raise Http404("Video not found.")


def Logout(request):
    # with open(rf' {BASE_DIR}\demo\static\css\var.js', 'r') as f:
    #     lines = f.readlines()

    # # iterate over the lines and replace the value of myVariable
    #     for i, line in enumerate(lines):
    #         if "Value =" in line:
    #             lines[i] = "Value = 'Not Classified yet';\n"

    # # write the modified lines back to the file
    #     with open(rf' {BASE_DIR}\demo\static\css\var.js', 'w') as f:
    #         f.writelines(lines)
    logout(request)
    
    # Del()

    return redirect('account_logout')


def load_video(request):
    f_path = request.FILES['Insert']
    print(f_path.name)
    store = FileSystemStorage()
    f_path = store.save(f_path.name, f_path)
    f_path = store.url(f_path)
    context = {'f_path': f_path, 'username': request.user.username}
    print(User.username)
    return render(request, 'Dashboard.html', context)


def Classify(request):

    if request.method == 'POST':
        video = request.FILES['Insert']
        print('here')
        print(video)
        print('content type',video.content_type)
        # Read the video file in-memory
      
        unique_filename = f"{uuid.uuid4()}{os.path.splitext(video.name)[1]}"
        video.name = unique_filename
        videoObj = UploadedVideos.objects.create(user=request.user, video=video)
        file_path = videoObj.video.path
        # cap = cv2.VideoCapture(file_path)
        # frame_count = 0
        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     frame_count += 1

        # print("Explicit Frame Count:", frame_count)
        # frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
        # print("implicit frames",frames)
       
        
        
        video_info = imageio.get_reader(file_path).get_meta_data()
        duration = video_info['duration']
        # calculate duration of the video 
        print(f"duration in seconds: {duration}") 
        # print(f"video time: {video_time}") 
        videoObj.duration = duration

        
        print('videoObj.video.name',videoObj.video.name)
        print(videoObj)
        videoObj.save()

        context = {"video": videoObj}
        # print(videoObj.video.url)
        # print(videoObj)
        t2 = threading.Thread(target=Pred, args=[videoObj])
        # use is_alive to check status of thread
        t2.start()
        return render(request,"dashboard.html",context)
        
    return redirect('dashboard')

@csrf_exempt
def upload_video_for_anonymous_user(request):
    if request.method == 'POST':
        user, _ = User.objects.get_or_create(username='anonymous')   
        video = request.FILES.get('video')
        if video:
            unique_filename = f"{uuid.uuid4()}{os.path.splitext(video.name)[1]}"
            video.name = unique_filename
            videoObj = UploadedVideos.objects.create(user=user, video=video)
            file_path = videoObj.video.path
            video_info = imageio.get_reader(file_path).get_meta_data()
            duration = video_info['duration']
            # calculate duration of the video 
            print(f"duration in seconds: {duration}")
            videoObj.duration = duration
            print('videoObj.video.name',videoObj.video.name)
            print(videoObj)
            videoObj.save()

            context = {"video": videoObj}
            # print(videoObj.video.url)
            # print(videoObj)
            t2 = threading.Thread(target=Pred, args=[videoObj])
            t2.start() 
            return JsonResponse({'message': 'Video uploaded successfully', 'video_id': videoObj.id})
        return JsonResponse({'message': 'No video file provided'}, status=400)
    return JsonResponse({'message': 'Invalid request method'}, status=405)


def write_to_json(request):
    if request.method == 'POST':
        current_data = None
        data = json.loads(request.body)
        with open(rf" {BASE_DIR}\demo\static\css\logs.json", "r") as f:
            if (os.stat(rf" {BASE_DIR}\demo\static\css\logs.json").st_size != 0):
                current_data = json.load(f)

        with open(rf" {BASE_DIR}\demo\static\css\logs.json", "w") as f:
            if current_data is None:
                l = [data]
                json.dump(l, f)
            else:
                current_data.append(data)
                json.dump(current_data, f)
        return JsonResponse("Successfully created", safe=False)


def assign_value(request,id):

    try:
        video = UploadedVideos.objects.get(id = id)
    except:
        return JsonResponse({"error": "video doesnot exist"})  
    val = "Not classified yet" 
    if video.classified:
        if video.violent:
            val = "violent"
        else:
            val = "Normal"
    data = {'value': val,
            'progress':video.progress,
            'currentState':"Violent" if video.isVoilentThisIteration else "Normal"}
    # print(data)
    # print(video.classified)
    return JsonResponse(data, safe=False)


def fetch_anonymous_logs(request):
    # Get or create anonymous user
    user, _ = User.objects.get_or_create(username='anonymous')

    videos = UploadedVideos.objects.filter(user=user).order_by('id')
    data = []
    for video in videos:
        video_data = {
            'video_id': video.id,
            'video_url': video.video.url,
            'is_violent': video.violent,
            # 'upload_date': video.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }
        if video.violent:
            # Assuming you store violent video clips in a specific directory
            dir_path = os.path.join(settings.MEDIA_ROOT, video.violentFramesDir)
            violent_video_path = f'/media/violent_{video.id}/violent.mp4'
            video_data['violent_vid_url'] = violent_video_path
        data.append(video_data)

    return JsonResponse(data, safe=False)