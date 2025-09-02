from django.db import models
from django.core.validators import MinValueValidator

# Create your models here.
class MenuItem(models.Model):
    name = models.CharField(max_length=255)
    price = models.IntegerField()


class Reservation(models.Model):
    name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    guest_count = models.IntegerField(
        default=0,
        validators=[MinValueValidator(1)]
    )
    reservation_time = models.TimeField(auto_now=True)
    comment = models.CharField(max_length=1000)
    
    def __str__(self):
            return self.name + " " + self.last_name # Or f"Reservation by {self.person_name}" for more context