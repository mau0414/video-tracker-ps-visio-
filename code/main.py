from VideoProcessor import VideoProcessor
from ObjectTracker import ObjectTracker
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn
import numpy as np

# caminhos do video de entrada, intermediario e de saida
input_path = '../src/1.mp4'
processed_video_path = '../src-gen-processed/1.mp4'
output_path = '../src-gen/1.mp4'

# definicoes do processamento inicial do video
camera_matrix = np.array([
    [
        331.3076549,
        0.0,
        630.16629165
    ],
    [
        0.0,
        333.06212044,
        458.44728206
    ],
    [
        0.0,
        0.0,
        1.0
    ]
])

dist_coeffs = np.array([
    1.40813986e+01,
    7.83197268e+00,
    -3.97740954e-04,
    -4.44407448e-05,
    2.72931297e-01,
    1.42843953e+01,
    1.27794122e+01,
    1.65578909e+00,
    -7.07882378e-04,
    9.66729468e-05,
    3.69522034e-05,
    7.40627933e-05,
    4.73807169e-03,
    4.47503227e-03
])

top_left = (128, 288)
bottom_right = (800, 928)

# definicoes do algoritmo de tracking
roi = { 'x_min': 100, 'x_max': 350, 'y_min': 150, 'y_max': 700 }
max_min_obj_dimensions = { 'min_height': 70, 'min_width': 70, 'max_height': 200, 'max_width': 200 }
min_score = 0.25
weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
model = retinanet_resnet50_fpn(weights=weights)


# aplicacao processamento inicial do video
video_processor = VideoProcessor(input_path, processed_video_path)
video_processor.process_video(camera_matrix, dist_coeffs, top_left, bottom_right)


# aplicao do tracking
obj_tracker = ObjectTracker(roi, max_min_obj_dimensions, min_score, model, weights, processed_video_path, output_path)
obj_tracker.track_object()