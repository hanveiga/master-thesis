from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views.generic import View
from django.views.generic import TemplateView
from django.conf.urls import patterns
from django.db.utils import IntegrityError
from django.views.decorators.csrf import csrf_exempt
import logging
import random
import re

from models import User, FB_User, FB_Likes
import managers

#class index(TemplateView):
#    template_name = 'index.php'

class index(View):
	template_name = 'index_test.php'

@csrf_exempt
def facebook_data(request):
	logger = logging.getLogger(__name__)
	#print 'RECEIVED REQUEST: ' + request.method

	if request.method == 'POST':
		user = User(userID = request.POST['userid'])
		
		try:
			user.save()
			logger.info("saved user %s", request.POST['userid'])
		except IntegrityError, error:
			logger.exception(error)
			continue

		fb_user = FB_User(Id=user, FB_ID=request.POST['userid'],
						  name=request.POST['username'], gender=request.POST['gender'],
						  hometown=request.POST['hometown'], location=request.POST['local'],
						  URL=request.POST['url'])
		
		try:
			fb_user.save()
			logger.info("saved user %s (%s).", request.POST['username'], request.POST['userid'])
		except IntegrityError, error:
			logger.exception(error)
			continue

		categories = re.findall(r'\"(.+?)\"', request.POST['likes_cats'])
		likes = re.findall(r'\"(.+?)\"', request.POST['likes'])

		for i in range(len(likes)):
			fb_likes = FB_Likes(Id=user, category= categories[i],
								name = likes[i] )
		    fb_likes.save()

		# Check for handles, create each entry in corresponding model
		# if exists
		#if request.POST['twitter']=!'':
		#	managers.get_twitter_user(user,request.POST['twitter'])
		#if request.POST['twitter']=!'':
		#  twitter_user = Twitter_User(Id=user)

		return HttpResponseRedirect('http://yahoo.com') 
	else: #GET
		print 'Nothing'
		return HttpResponseRedirect('http://google.com') 