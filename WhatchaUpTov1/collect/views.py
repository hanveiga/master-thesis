from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views.generic import View
from django.views.generic import TemplateView
from django.conf.urls import patterns

from django.views.decorators.csrf import csrf_exempt

#class index(TemplateView):
#    template_name = 'index.php'

class index(View):
	template_name = 'index_test.php'

@csrf_exempt
def buttonExample(request):
    print 'RECEIVED REQUEST: ' + request.method
    if request.method == 'POST':
        print 'Hello'
        print 'dude'
        print request
        return HttpResponseRedirect('http://google.com') 
    else: #GET
        print 'Nothing'
        return HttpResponseRedirect('http://google.com') 