from django.conf.urls import url

from pre_process.api import views


urlpatterns = [
    url(r'^image_to_string', views.ImageToString.as_view(), name='image_to_string'),
]
