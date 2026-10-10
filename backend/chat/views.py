from rest_framework import status
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework.views import APIView

from llm.base import RateLimitError

from chat.serializers import ChatRequestSerializer
from chat.service import answer_question


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
        except RateLimitError:
            return Response({"detail": "Model şu an yoğun, lütfen biraz sonra tekrar deneyin."},
                            status=status.HTTP_429_TOO_MANY_REQUESTS)
        return Response(result)
