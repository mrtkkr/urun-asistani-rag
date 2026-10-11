from rest_framework import status
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework.views import APIView

from chat.serializers import ChatRequestSerializer
from chat.service import ChatUnavailableError, answer_question


class ChatThrottle(AnonRateThrottle):
    scope = "chat"  # rate comes from REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"]["chat"]


class ChatView(APIView):
    throttle_classes = [ChatThrottle]

    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            result = answer_question(data["message"], data["history"])
        except ChatUnavailableError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response(result)
