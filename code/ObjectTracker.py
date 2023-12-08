import numpy as np
import cv2 as cv
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import sys

class ObjectTracker:
    def __init__(self, roi, max_min_obj_dimensions, min_score, model, weights, input_path, output_path):
        self.device = self.define_device()
        self.model = self.create_model(model)
        self.roi = roi
        self.max_min_obj_dimensions = max_min_obj_dimensions
        self.input_path = input_path
        self.output_path = output_path
        self.min_score = min_score
        self.preprocess = weights.transforms()
        self.tracker = self.init_tracker()

    '''
    verifica se objeto descrito em box esta na regiao de interesse (roi)
    '''
    def check_is_in_roi(self, box):
        x_min, y_min, x_max, y_max = box

        return (x_min >= self.roi['x_min'] and
                x_max <= self.roi['x_max'] and
                y_min >= self.roi['y_min'] and
                y_max <= self.roi['y_max'])

    def create_model(self, model):
        model.to(self.device)
        model.eval()

        return model

    '''
    verifica se gpu esta disponivel para uso; senao usa cpu
    '''
    def define_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('cuda em uso')
        else:
            device = torch.device('cpu')
            print('cpu em uso')

        return device

    '''
    formata tempo em segundos para formato minutos:segundos
    '''
    def formatar_tempo(self, tempo_seg):
      minutos = int(tempo_seg//60)
      segundos = int(tempo_seg -60*minutos)
      return str(minutos) + ':' + str(segundos)

    '''
    responsavel por rastrear objeto ao longo dos frames e calcular tempo de atendimento
    '''
    def track_object(self):

        cap, out = self.create_cap_out(self.input_path, self.output_path)
        fps = cap.get(cv.CAP_PROP_FPS)
        n_frames = 0

        if (not cap.isOpened()):
            print('falha ao abrir o video.')
            sys.exit()

        while True:
            n_frames += 1
            ret, img = cap.read()

            if not ret:
                print('finalizando')
                break

            img_as_tensor = self.prepare_image(img)
            ret, box = self.tracker.update(img)

            if ret:
                tempo_atendimento =  n_frames//fps
                self.draw_rectangle(img, self.formatar_tempo(tempo_atendimento), box)
            else:
                continue
            out.write(img)

            if cv.waitKey(15) == ord('q'):
                break

        cap.release()

    '''
    inicializa tracker com deteccao inicial do objeto desejado
    '''
    def init_tracker(self):
      tracker = cv.TrackerCSRT_create()
      img, bbox = self.find_object()
      ret = tracker.init(img, tuple(bbox))

      if (not ret):
        print('falhou na inicializacao')

      return tracker

    '''
    encontra pessoa (objeto desejado) dentro do frame passado
    '''
    def find_object(self):
      cap = cv.VideoCapture(self.input_path)

      ret, img = cap.read()

      if not ret:
        print('erro ao abrir o video no caminho', self.input_path)

      img_as_tensor = self.prepare_image(img)
      boxes, scores, labels = self.make_prediction(img_as_tensor)

      for box, score, label in zip(boxes, scores, labels):
            if (score >= self.min_score and self.check_obj_dimension(box) and self.check_is_in_roi(box)):
                box = box.cpu().detach().numpy().astype(int)
                x_min, y_min, x_max, y_max = box
                bbox_for_tracker = (x_min, y_min, x_max - x_min, y_max - y_min)
                return img, bbox_for_tracker

      print('objeto de interesse nao encontrado')
      sys.exit()

    '''
    verifica se objeto passado em box atende as dimensoes maximas e minimas que uma pessoa deveria ter dentro do video
    '''
    def check_obj_dimension(self, box):
        min_height = min_width = 100
        max_height = max_width = 200

        x_min, y_min, x_max, y_max = box
        height = x_max - x_min
        width = y_max - y_min

        return (height >= self.max_min_obj_dimensions['min_height'] and
                height <= self.max_min_obj_dimensions['max_height'] and
                width >= self.max_min_obj_dimensions['min_width'] and
                width <= self.max_min_obj_dimensions['max_width'])

    '''
    cria e retorna capture e escritor dos video sde entrada e saida
    '''
    def create_cap_out(self, input_path, output_path):
        cap = cv.VideoCapture(input_path)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS)
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        return cap, out

    '''
    desenha retangulo no objeto descrito em box, juntamento com o tempo de atendimento calculado
    '''
    def draw_rectangle(self, img, tempo_atendimento, box):
        x, y, width, height = map(int, box)

        img = cv.rectangle(img, (x, y), (x + width, y + height), (64, 255, 0), 1)
        rectang_txt = ((x, y-10), (x + 200, y - 30))
        img = cv.rectangle(img, rectang_txt[0], rectang_txt[1], (64, 255, 0), -1)
        img = cv.putText(img, 'tempo:' + tempo_atendimento,
                        (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv.LINE_AA, False)

    '''
    retorna imagem em rgb e transformada em tensor para entrar no modelo
    '''
    def prepare_image(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_as_tensor = ToTensor()(img_rgb)

        return img_as_tensor

    '''
    aplica predicao para detectar objetos na imagem
    '''
    def make_prediction(self, img_as_tensor):
        batch = self.preprocess(img_as_tensor).unsqueeze(0)
        batch = batch.to(self.device)
        prediction = self.model(batch)[0]

        return prediction['boxes'], prediction['scores'], prediction['labels']