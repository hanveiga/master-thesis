from django.contrib import admin

# Register your models here.
from models import User, FB_User, FB_Like, Twitter_User, Tweets

admin.site.register(User)
admin.site.register(FB_User)
admin.site.register(FB_Like)
admin.site.register(Twitter_User)	
admin.site.register(Tweets)	