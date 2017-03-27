import cv2
import numpy as np
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from viet_ocr.models import VietOCR, VietOCRError
from pre_process.models import AdvancedPreProcess
from post_process.models import PostProcess


class ImageToString(APIView):
    def get(self, request):
        # test django
        return Response(status=status.HTTP_200_OK)
        
    def post(self, request):
        img = request.data.get('image')
        language = request.data.get('lang')
        if not (img and language):
            return Response(status=status.HTTP_400_BAD_REQUEST)

        img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        pre = AdvancedPreProcess()
        img = pre.process(img)
        
        viet_ocr = VietOCR()
        try:
            text = viet_ocr.image_to_string(img, lang=language)
        except VietOCRError:            
            return Response(status=status.HTTP_400_BAD_REQUEST)

        if language == 'vie':
            post_process = PostProcess(text)
            processed_text = post_process.process()
        else:
            processed_text = text

        return Response({'text': processed_text}, status=status.HTTP_200_OK)
