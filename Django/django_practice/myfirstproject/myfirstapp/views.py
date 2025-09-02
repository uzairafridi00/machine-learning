from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from .forms import ReservationForm

# Create your views here.
def hello_world(request):
    return HttpResponse("Hello, World!!")

class HelloPakistan(View):
    def get(self, request):
        return HttpResponse("Hello, Pakistan!!")

def home(request):
    form = ReservationForm()

    if request.method == 'POST':
        form = ReservationForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponse("Reservation made successfully!")
    
    return render(request, 'index.html', {'form': form})
