import os
import tensorflow as tf
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import tempfile
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from .models import CreditCardTransaction

def custom_abs_diff(tensors):
    x, y = tensors
    return tf.abs(x - y)

# Ensure custom Lambda function is registered during model loading
custom_objects = {'custom_abs_diff': custom_abs_diff}
model = load_model('signet_model_augmented.keras', custom_objects=custom_objects, compile=False)

@api_view(['POST'])
def upload_image(request):
    if 'image' not in request.FILES:
        return Response({"error": "No image uploaded"}, status=status.HTTP_400_BAD_REQUEST)

    image = request.FILES['image']
    ac_number = request.POST.get('account_number')
    image_path = os.path.join(settings.MEDIA_ROOT, image.name)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        for chunk in image.chunks():
            temp_file.write(chunk)
        temp_file_path = temp_file.name
    transaction = CreditCardTransaction.objects.get(account_number=ac_number)
    model = load_model('signet_model_augmented.keras', custom_objects=custom_objects, compile=False)
    path=transaction.signature_image
    print(path)
    genuineness = evaluate_signature(temp_file_path,"media/"+str(path),model)
    os.remove(temp_file_path)
    return JsonResponse({'message': "Fake" if genuineness >75 else "Genuine" }, status=status.HTTP_200_OK)



# Function to preprocess an image
def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(155, 220))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img



# Load the model with custom objects

def evaluate_signature(img1_path, img2_path,model):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)
    prediction = model.predict([img1, img2])
    return prediction[0][0] * 100



#
#Login part
from django.contrib.auth.models import User
from rest_framework import generics
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserSerializer
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

class UserDetailsView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user = request.user
        serializer = UserSerializer(user)
        return Response(serializer.data)

class SignupView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(generics.GenericAPIView):
    serializer_class = UserSerializer

    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        password = request.data.get('password')
        user = User.objects.filter(username=username).first()
        if user and user.check_password(password):
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            })
        return Response({'error': 'Invalid Credentials'}, status=400)
