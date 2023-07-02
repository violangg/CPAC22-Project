from django.db import models

class Coordinates(models.Model):
    x = models.FloatField()
    y = models.FloatField()