from django.urls import path
from .views import upload_image,SignupView,LoginView,UserDetailsView

urlpatterns = [
    path('upload/', upload_image, name='upload_image'),
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('user-details/', UserDetailsView.as_view(), name='user-details'),
]
