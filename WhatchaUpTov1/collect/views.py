from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views.generic import View
from django.views.generic import TemplateView
from django.conf.urls import patterns

from django.views.decorators.csrf import csrf_exempt
from models import User, FB_User, FB_Likes

import re
#class index(TemplateView):
#    template_name = 'index.php'

class index(View):
	template_name = 'index_test.php'

@csrf_exempt
def facebook_data(request):

    print 'RECEIVED REQUEST: ' + request.method

    if request.method == 'POST':
        #print request
        user = User(userID = request.POST['userid'])
        #print user.userID
        #print request.POST['username']
        #user.save()
        fb_user = FB_User(Id=user, FB_ID=request.POST['userid'],
                          first_name=request.POST['username'], last_name=request.POST['username'],
                          hometown=request.POST['hometown'], location=request.POST['local'],
                          URL=request.POST['local'])
        print user
        #fb_user.save()
        #print fb_user
        #print request.POST['likes_cats']
        categories = re.findall(r'\"(.+?)\"', request.POST['likes_cats'])
        likes = re.findall(r'\"(.+?)\"', request.POST['likes'])

        #print bool(len(categories)==len(likes))
        for i in range(len(likes)):
            fb_likes = FB_Likes(Id=fb_user, category= categories[i],
                                name = likes[i] )
        #    fb_likes.save()

        return HttpResponseRedirect('http://yahoo.com') 
    else: #GET
        print 'Nothing'
        return HttpResponseRedirect('http://google.com') 