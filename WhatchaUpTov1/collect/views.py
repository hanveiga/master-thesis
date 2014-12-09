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

from models import User, FB_User, FB_Like
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
		if (FB_User.objects.filter(FB_ID = request.POST['userid']).exists()):
			print "exists already"
			u = FB_User.objects.filter(FB_ID = request.POST['userid']).delete()
			return HttpResponseRedirect('http://yahoo.com') 
		else:

			user = User()
			#print "made user"
			
			try:
				user.save()
				#print "saved user %s", request.POST['userid']
				#logger.info("saved user %s", request.POST['userid'])
			except IntegrityError, error:
				logger.exception(error)
				pass

			fb_user = FB_User(Id=user, FB_ID=request.POST['userid'],
							  name=request.POST['username'], gender=request.POST['gender'],
							  hometown=request.POST['hometown'], location=request.POST['local'],
							  URL=request.POST['url'])
			print "made FB user"
			try:

				fb_user.save()
				logger.info("saved user %s (%s).", request.POST['username'], request.POST['userid'])
			except IntegrityError, error:
				logger.exception(error)
				pass

			categories = eval(request.POST['likes_cats']) #re.findall(r'\"(.+?)\"', request.POST['likes_cats'])
			likes = eval(request.POST['likes']) #re.findall(r'\"(.+?)\"', request.POST['likes'])
			print len(categories)
			print len(likes)
			print likes
			for i in range(len(likes)):
				print "making likes"
				fb_likes = FB_Like(Id=user, category = categories[i],
									name = likes[i] )
				print "makde fb_like"
				fb_likes.save()

		# Check for handles, create each entry in corresponding model
		# if exists
		if len(request.POST['twitter'].strip())>0:
			managers.get_twitter(user,request.POST['twitter'])
		#if request.POST['twitter']=!'':
		#  twitter_user = Twitter_User(Id=user)

		return HttpResponseRedirect('http://yahoo.com') 
	else: #GET
		print 'Nothing'
		return HttpResponseRedirect('http://google.com') 