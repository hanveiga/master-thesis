from django.conf.urls import patterns, include, url
from django.contrib import admin
from django.http import HttpResponseRedirect
from django.views.generic import TemplateView

from collect import views
import settings

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'WhatchaUpTov1.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^$', lambda r: HttpResponseRedirect(settings.STATIC_URL + 'index.php')),
    #url(r'^post', post.as_view()),
    url(r'^post', views.buttonExample),
    #url(r'^$', include('collect.urls')),
    #url('^$', TemplateView.as_view(template_name='index.php')),
)
