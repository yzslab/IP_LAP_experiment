import os
import numpy as np
import cv2
import torch
import mediapipe as mp

from draw_landmark import draw_landmarks

drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# the following is the index sequence for fical landmarks detected by mediapipe
ori_sequence_idx = [162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
                    361, 323, 454, 356, 389,  #
                    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  #
                    336, 296, 334, 293, 300, 276, 283, 282, 295, 285,  #
                    168, 6, 197, 195, 5,  #
                    48, 115, 220, 45, 4, 275, 440, 344, 278,  #
                    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,  #
                    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,  #
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,  #
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# the following is the connections of landmarks for drawing sketch image
FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])
FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])
FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])
FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])
FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])
FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])
FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
                           (4, 45), (45, 220), (220, 115), (115, 48),
                           (4, 275), (275, 440), (440, 344), (344, 278), ])
FACEMESH_CONNECTION = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])

full_face_landmark_sequence = [*list(range(0, 4)), *list(range(21, 25)), *list(range(25, 91)),  # upper-half face
                               *list(range(4, 21)),  # jaw
                               *list(range(91, 131))]  # mouth

full_face_landmark_sequence = [*list(range(0, 4)), *list(range(21, 25)), *list(range(25, 91)),  # upper-half face
                               *list(range(4, 21)),  # jaw
                               *list(range(91, 131))]  # mouth

FACEMESH_CONNECTION = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])


class LandmarkDict(dict):  # Makes a dictionary that behave like an object to represent each landmark
    def __init__(self, idx, x, y):
        self['idx'] = idx
        self['x'] = x
        self['y'] = y

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


base_dir = "/data/nfs4a100/IP_LAP/cnthv_preprocessed/"
video_name = "BJJT_bc954211a7784321853545ba96152525.ts"
clip_name = "00000"
frame_idx = 0

landmark_dict = np.load(
    "{}/landmarks/{}/{}/{}.npy".format(
        base_dir,
        video_name,
        clip_name,
        frame_idx,
    ),
    allow_pickle=True).item()

pose = torch.zeros((2, 74))  # 74 landmark
content = torch.zeros((2, 57))  # 57 landmark
pose_landmarks = sorted(landmark_dict["pose_landmarks"], key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
content_landmarks = sorted(landmark_dict["content_landmarks"],
                           key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))

pose[0, :] = torch.FloatTensor([pose_landmarks[i][1] for i in range(len(pose_landmarks))])  # x
pose[1, :] = torch.FloatTensor([pose_landmarks[i][2] for i in range(len(pose_landmarks))])  # y

content[0, :] = torch.FloatTensor(
    [content_landmarks[i][1] for i in range(len(content_landmarks))])  # x
content[1, :] = torch.FloatTensor(
    [content_landmarks[i][2] for i in range(len(content_landmarks))])  # y

full_landmarks = torch.cat([pose, content], dim=-1).cpu().numpy()

SKETCH_IMAGE_SIZE = 128

annotated_image = np.zeros((SKETCH_IMAGE_SIZE, SKETCH_IMAGE_SIZE, 3))
mediapipe_format_landmarks = [LandmarkDict(
    ori_sequence_idx[full_face_landmark_sequence[idx]],
    full_landmarks[0, idx],
    full_landmarks[1, idx],
) for idx in
    range(full_landmarks.shape[1])]
drawn_sketech = draw_landmarks(annotated_image, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                               connection_drawing_spec=drawing_spec)

gt_sketch = cv2.imread("{}/sketch/{}/{}/{}.png".format(
    base_dir,
    video_name,
    clip_name,
    frame_idx,
))
gt_sketch = cv2.resize(gt_sketch, (SKETCH_IMAGE_SIZE, SKETCH_IMAGE_SIZE))

cv2.imwrite("visualized.jpg", np.concatenate([gt_sketch, annotated_image], axis=0))
