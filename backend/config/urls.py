from django.urls import path

from config.views import health

urlpatterns = [
    path("api/health/", health),
]
