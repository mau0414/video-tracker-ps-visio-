import cv2 as cv
import numpy as np
import os
import sys

class VideoProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.cap = self.read_video()
    
    '''
    processa video frame a frame, modificando cada um e salvando video final
    '''
    def process_video(self, cameraMatrix, distCoeffs, top_left, bottom_right):
        out = self.create_writer(top_left, bottom_right)
        if (not self.cap.isOpened()):
            print('erro ao abrir video')
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                print('erro ao ler frame')
                break
        
            frame_processado = self.undistort_frame(frame, cameraMatrix, distCoeffs)
            frame_processado = self.crop_frame(frame_processado, top_left, bottom_right)

            out.write(frame_processado)
            if cv.waitKey(15) == ord('q'):
                break 

    '''
    corta e retorna um frame com base na regiao descrita em top_left e bottom_right
    '''
    def crop_frame(self, frame, top_left, bottom_right):
        frame = frame[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1] ]
        return frame
    
    '''
    abre video e retorna sua capture
    '''
    def read_video(self):
        cap = cv.VideoCapture(self.input_path)
        if not cap:
            print("erro ao ler video")
            exit()
        
        return cap
    
    '''
    cria escritor para salver video
    '''
    def create_writer(self, top_left, bottom_right):
        width = int(bottom_right[1] - top_left[1])
        height = int(bottom_right[0] - top_left[0]) 
        fps = self.cap.get(cv.CAP_PROP_FPS)
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        
        # Se o diretório não existir, cria
        output_directory = os.path.dirname(self.output_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        
        out = cv.VideoWriter(self.output_path, fourcc, fps, (width, height))
        return out
    
    '''
    aplica operacao de undistort no frame e retorna
    '''
    def undistort_frame(self, frame, cameraMatrix, distCoeffs):
        return cv.undistort(frame, cameraMatrix, distCoeffs)

    '''
    libera recursos alocados para processamento do video
    '''
    def release_resources(self):
        self.cap.release()
        cv.destroyAllWindows()