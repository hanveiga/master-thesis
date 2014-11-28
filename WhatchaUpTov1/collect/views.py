from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.generic import View
from django.views.generic import TemplateView
from django.conf.urls import patterns

class IndexView(TemplateView):
    template_name = 'templates/index.php'