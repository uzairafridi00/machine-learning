from django import forms

from .models import Reservation

class ReservationForm(forms.ModelForm):
    class Meta:
        model = Reservation
        #fields = ['name', 'last_name', 'guest_count', 'reservation_time', 'comment']
        fields = '__all__'