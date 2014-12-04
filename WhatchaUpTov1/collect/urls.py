from django.conf.urls import patterns, url
from django.views.generic import TemplateView

from collect import views
from django.conf import settings


urlpatterns = patterns('',
    #url(r'^$', views.index.as_view(), name='index'),
    url(r'^$', TemplateView.as_view(template_name='index.php'),
    url(r'^post', views.facebook_data),
)
