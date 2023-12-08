import cv2 as cv
import numpy as np
import os
from code.VideoProcessor import VideoProcessor

# Testar a leitura e gravação de vídeo
def test_video_paths():
    input_path = '../src/1.mp4'
    processed_video_path = '../src-gen-processed/teste.mp4'
    top_left = (128, 288)
    bottom_right = (800, 928)

    video_processor = VideoProcessor(input_path, processed_video_path)
    assert video_processor.cap.isOpened(), "Falha ao abrir o vídeo de entrada: Verificar caminho de entrada passado" 
    assert video_processor.create_writer(top_left, bottom_right) is not None, "Falha ao criar o gravador de vídeo: Verificar pametros fps, width e height"
    video_processor.release_resources()

# teste de liberacao de recursos
def test_release_resources():
    input_path = '../src/1.mp4'
    processed_video_path = '../src-gen-processed/teste.mp4'

    video_processor = VideoProcessor(input_path, processed_video_path)
    video_processor.release_resources()
   
    assert video_processor.cap is None, "Falha ao fechar o vídeo de entrada"
