import cv2 as cv
import sys

'''
funcao auxiliar de debug para rodar um video frame a frame e exibir na tela
'''
def execute_video(video_path):   
    cap = cv.VideoCapture(video_path)
    if (not cap.isOpened()):
        print('falha ao abrir o video')
    
    while cap.isOpened():
        ret, frame = cap.read()
        cv.imshow('Video', frame)
        if not ret:
            print('erro ao ler frame')
            break

        cv.imshow('Video', frame)

        # espera tecla q ser pressionada
        if cv.waitKey(15) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

video_path = sys.argv[1]
print('video_path passado: ', video_path)
execute_video(video_path)