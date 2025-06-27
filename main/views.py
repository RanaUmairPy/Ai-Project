from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .serializers import ImageUploadSerializer
import numpy as np
import cv2
import os
from django.conf import settings

threshold = 0.6517  # Confidence threshold

class PredictViewSet(viewsets.ViewSet):

    @action(detail=False, methods=['post'], url_path='image')
    def predict_image(self, request):
        from keras.saving.legacy import load_model  # Import lazily to avoid startup overhead
        MODEL_PATH = os.path.join(settings.BASE_DIR, 'umairpy_legacy.h5')
        model = load_model(MODEL_PATH)

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
