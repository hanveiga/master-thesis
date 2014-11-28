from django.db import models

class User(models.Model):
  userID = models.CharField(max_length=200)

class FB_User(models.Model):
  Id = models.ForeignKey(User) 
  FB_ID = models.CharField(max_length=200)
  first_name = models.CharField(max_length=200)
  last_name = models.CharField(max_length=200)
  hometown = models.CharField(max_length=200)
  location = models.CharField(max_length=200)
  URL = models.URLField(max_length=200, unique=True, blank=False)

class FB_Likes(models.Model):
  Id = models.ForeignKey(FB_User)
  category = models.CharField(max_length=200)
  name = models.CharField(max_length=200)
  created_at = models.DateTimeField()
 

# Create your models here.
