from django.contrib import admin
from django.urls import path, include
from home import views

urlpatterns = [
    path('', views.landing, name='landing'),

    path('register', views.register, name='register'),
    path('login', views.login, name="login_view"),
    # path('loginView', views.loginView, name="login"),
    path('clearalllogs', views.clearLogs, name="clearAllLogs"),
    path('deletelog/<int:id>', views.deleteLog, name="deleteLog"),

    path('logs', views.logs, name="logs_view"),
    path('live', views.live, name="live"),
    path('live_classification', views.live_classification,name="live_classification"),
    path('dashboard', views.dashboard, name="dashboard"),
    path('Classify', views.Classify, name="Classify_web"),
    path('load_video', views.load_video, name="load_video"),
    path('Logout', views.Logout, name="Logout"),
    path('assign_value/<int:id>', views.assign_value, name="assign_value"),
    path('viewVideo/<int:id>', views.viewVideo, name="viewVideo"),

    path('write_to_json', views.write_to_json, name="write_to_json"),
    path('send_json', views.send_json, name="send_json"),
    path('updatepfp', views.updatePfp, name="updatePfp"),
    path('profile', views.profile, name="profile"),
    path('updateinfo', views.updateInfo, name="updateInfo"),
    path('api/upload_video', views.upload_video_for_anonymous_user, name='upload_video_for_anonymous_user'),
    path('api/anonymous/logs', views.anonymous_logs, name='anonymous_logs'),
    path('api/anonymous/clear_logs', views.clear_all_anonymous_logs, name='clear_all_anonymous_logs'),
    path('api/anonymous/delete_log/<int:id>', views.delete_anonymous_log, name='delete_anonymous_log'),

    path('stream_video/<int:video_id>/', views.stream_video, name='stream_video'),
    path('stream_violent_video/<int:video_id>/', views.stream_violent_video, name='stream_violent_video'),


]
