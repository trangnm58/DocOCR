import subprocess
import tempfile
import os
import cv2
from sys import platform


class VietOCRError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message
        self.args = (status, message)


class VietOCR():
    def __init__(self):
        if platform == "win32":
            self.vietocr_cmd = 'D:/Program Files/VietOCR3/VietOCR.bat'
        else:
            self.vietocr_cmd = '/home/nguyentrang/VietOCR3/tesseract-ocr/tesseract'

    def image_to_string(self, image, lang=None):
        input_file_name = '%s.bmp' % self._tempnam()
        output_file_name_base = self._tempnam()
    
        output_file_name = '%s.txt' % output_file_name_base

        try:
            cv2.imwrite(input_file_name, image)
            status, error_string = self._run(input_file_name,
                                            output_file_name_base,
                                            lang=lang)
            if status:
                errors = self._get_errors(error_string)
                raise VietOCRError(status, errors)
            f = open(output_file_name, encoding='utf8')
            try:
                return f.read().strip()
            finally:
                f.close()
        finally:
            self._cleanup(input_file_name)
            self._cleanup(output_file_name)

    def _run(self, input_filename, output_filename_base, lang=None):
        command = [self.vietocr_cmd, input_filename, output_filename_base]
        
        if lang:
            command += ['-l', lang]
        
        # utilize VietOCR post-processing
        command += ['text+']
        
        proc = subprocess.Popen(command, stderr=subprocess.PIPE)
        return (proc.wait(), proc.stderr.read())

    def _cleanup(self, filename):
        """ tries to remove the given filename. Ignores non-existent files """
        try:
            os.remove(filename)
        except OSError:
            pass

    def _get_errors(self, error_string):
        """ returns all lines in the error_string that start with the string "error" """
        lines = error_string.splitlines()
        error_lines = tuple(line for line in lines if line.find('Error') >= 0)
        if len(error_lines) > 0:
            return '\n'.join(error_lines)
        else:
            return error_string.strip()

    def _tempnam(self):
        """ returns a temporary file-name """
        tmpfile = tempfile.NamedTemporaryFile(prefix="tess_")
        return tmpfile.name
