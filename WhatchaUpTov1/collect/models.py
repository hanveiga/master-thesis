from django.db import models

class User(models.Model):
  userID = models.AutoField(primary_key=True)

class FB_User(models.Model):
  Id = models.ForeignKey(User) 
  FB_ID = models.CharField(max_length=200)
  name = models.CharField(max_length=200)
  hometown = models.CharField(max_length=200)
  location = models.CharField(max_length=200)
  gender = models.CharField(max_length=200)
  URL = models.URLField(max_length=200, blank=False, unique=True)

class FB_Like(models.Model):
  Id = models.ForeignKey(User)
  category = models.CharField(max_length=200)
  name = models.CharField(max_length=200)
 
class Twitter_User(models.Model):
  Id = models.ForeignKey(User)
  username = models.CharField(max_length=200)
  twitter_id = models.CharField(max_length=200)
  location = models.CharField(max_length=200)
  description = models.CharField(max_length=200)
  followers = models.CharField(max_length=200)

class Tweets(models.Model):
  Id = models.ForeignKey(User)
  created_at = models.CharField(max_length=200)
  text = models.CharField(max_length=160)
  coords = models.CharField(max_length=200)
  hashtags = models.CharField(max_length=200)
  links = models.CharField(max_length=200)