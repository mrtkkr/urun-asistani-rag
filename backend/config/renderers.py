from rest_framework.renderers import JSONRenderer


class UTF8JSONRenderer(JSONRenderer):
    """JSON with an explicit charset, so clients that default to a legacy encoding (e.g. Windows
    PowerShell's Invoke-RestMethod) still decode Turkish characters correctly."""

    charset = "utf-8"
