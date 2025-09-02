from django.urls import path
from . import views

urlpatterns = [
    path('function', views.hello_world, name='hello_world'),
    path('class', views.HelloPakistan.as_view(), name='hello_pakistan'),
    path('reservation', views.home, name='home'),
]