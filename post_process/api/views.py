import cv2
import numpy as np
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from viet_ocr.models import VietOCR
from post_process.models import PostProcess


class ImageToString(APIView):
    def post(self, request):
        img = request.data.get('image')
        language = request.data.get('lang')
        if not (img and language):
            return Response(status=status.HTTP_400_BAD_REQUEST)
        
        img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        viet_ocr = VietOCR()
        text = viet_ocr.image_to_string(img, lang=language)
         
        post_process = PostProcess()
        processed_text = post_process.process(text)
         
        return Response({'text': processed_text}, status=status.HTTP_200_OK)
