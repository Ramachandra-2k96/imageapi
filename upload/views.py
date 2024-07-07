import os
import cv2
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import io

# Load your TensorFlow
model_path = os.path.join(settings.BASE_DIR, "enhanced_signature_verification_model3.keras")
model = load_model(model_path, compile=False)

@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_image(request):
    if 'real_image' not in request.FILES or 'fake_image' not in request.FILES:
        return Response({"error": "No image uploaded"}, status=status.HTTP_400_BAD_REQUEST)

    real = request.FILES['real_image']
    fake = request.FILES['fake_image']
    real= io.BytesIO(real.read())
    fake = io.BytesIO(fake.read())

    # Preprocess and predict images in memory
    genuineness = predict_signature(model, real, fake)
    print(genuineness)
    return JsonResponse({'message': "Fake" if genuineness > 10 else "Genuine" }, status=status.HTTP_200_OK)


def preprocess_image(image_path):
    # Load the image in grayscale
    img = load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    img = img_to_array(img).astype('uint8')
    
    # Apply edge detection
    edges = cv2.Canny(img, threshold1=30, threshold2=100)
    
    # Dilate the edges to broaden the lines
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Invert the edges: detected edges should be black, and the rest should be white
    edges = cv2.bitwise_not(edges)
    
    # Normalize the image
    edges = edges / 255.0
    edges = np.expand_dims(edges, axis=-1)
    
    return edges
def preprocess_image1(image_path):
    processed_img = preprocess_image(image_path)
    processed_img = processed_img / 255.0
    processed_img = processed_img.reshape(1, 128, 128, 1)
    return processed_img

def predict_signature(model, real_signature_path, test_signature_path):
    real_img = preprocess_image1(real_signature_path)
    test_img = preprocess_image1(test_signature_path)
    
    real_pred = model.predict(real_img)
    test_pred = model.predict(test_img)
    
    difference = np.abs(real_pred - test_pred)
    return difference*100

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
