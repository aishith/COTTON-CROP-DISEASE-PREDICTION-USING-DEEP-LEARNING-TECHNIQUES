from django.shortcuts import render

from .models import predict
from django.shortcuts import render


class_names = ['diseased cotton leaf','diseased cotton Plant', 'fresh cotton leaf','fresh cotton Plant']
# Create your views here.
def home(request):
	return render(request,'index.html')
def input(request):
	return render(request,'input.html')
def output(request):
	img=request.FILES['file']
	algo=request.POST.get('algo')
	out=predict(img,algo)
	classes = class_names[int(out)]
	print(classes)
	return render(request,'output.html',{'out':classes})
