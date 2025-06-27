import numpy as np
import cv2
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageUploadSerializer
from tensorflow.keras.models import load_model
from django.conf import settings
import os

# Load model once globally
#MODEL_PATH = os.path.join(settings.BASE_DIR, 'your_model.h5')
model = load_model("D:\Ai\umairpy.h5")
threshold = 0.6517

class PredictView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            image_data = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224)) / 255.0
            img = np.expand_dims(img, axis=0)

            prob = model.predict(img)[0][0]
            label = "Melanoma" if prob >= threshold else "Benign"

            return Response({
                'prediction': label,
                'confidence': round(float(prob), 4)
            }, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
