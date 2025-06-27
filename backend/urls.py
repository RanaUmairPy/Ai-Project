from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from main.views import PredictViewSet  # <- import your ViewSet

# Set up router
router = DefaultRouter()
router.register(r'predict', PredictViewSet, basename='predict')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(router.urls)),  # <- plug in the router
]
