from django.conf.urls import url, include


urlpatterns = [
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^api/viet_ocr/', include('viet_ocr.api.urls', namespace="viet_ocr-api")),
    url(r'^api/post_process/', include('post_process.api.urls', namespace="post_process-api")),
    url(r'^api/pre_process/', include('pre_process.api.urls', namespace="pre_process-api")),
    url(r'^api/doc_ocr/', include('doc_ocr.api.urls', namespace="doc_ocr-api")),
]
