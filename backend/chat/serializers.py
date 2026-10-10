from django.conf import settings
from rest_framework import serializers


class HistoryItemSerializer(serializers.Serializer):
    role = serializers.ChoiceField(choices=["user", "assistant"])
    content = serializers.CharField(allow_blank=False, trim_whitespace=True)

    def validate_content(self, value):
        if len(value) > settings.CHAT_MAX_HISTORY_ITEM_CHARS:
            raise serializers.ValidationError(
                f"En fazla {settings.CHAT_MAX_HISTORY_ITEM_CHARS} karakter olabilir."
            )
        return value


class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(allow_blank=False, trim_whitespace=True)
    history = HistoryItemSerializer(many=True, required=False, default=list)

    def validate_message(self, value):
        if len(value) > settings.CHAT_MAX_MESSAGE_CHARS:
            raise serializers.ValidationError(
                f"Mesaj en fazla {settings.CHAT_MAX_MESSAGE_CHARS} karakter olabilir."
            )
        return value

    def validate_history(self, value):
        if len(value) > settings.CHAT_MAX_HISTORY_ITEMS:
            raise serializers.ValidationError(
                f"Geçmiş en fazla {settings.CHAT_MAX_HISTORY_ITEMS} mesaj olabilir."
            )
        return value
