import os
import subprocess
import argparse
import random
import time
import math
import numpy as np
import cv2
import torch
import dataclasses
import pickle
from glob import glob
from tqdm import tqdm
import mediapipe as mp
from models import audio, Landmark_generator as Landmark_transformer
from typing import Tuple, List
from landmark_constants import all_landmark_idx, pose_landmark_idx, content_landmark_idx, ori_sequence_idx, \
    FACEMESH_FULL
from draw_pose_and_content import draw_pose_and_content, draw_landmarks_from_preprocess_video

# define constants
AUDIO_SAMPLE_RATE = 16_000
Nl = 15
T = 5
fps = 25
mel_step_size = 16
lip_index = [0, 17]  # the index of the midpoints of the upper lip and lower lip


def extract_audio_from_video(video_filename: str, output_path: str):
    assert subprocess.call([
        "ffmpeg",
        "-y",
        "-i",
        video_filename,
        "-f",
        "wav",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        output_path,
    ], shell=False) == 0
    subprocess.call(["stty", "sane"])


# define functions
def extract_landmark_from_video(video_filename: str, padding_pixels: int) -> Tuple[List, List, List, List, List]:
    # read video frames
    video_stream = cv2.VideoCapture(video_filename)
    assert video_stream.get(cv2.CAP_PROP_FPS) == 25.
    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    # extract landmarks from frames
    frame_pose_landmark_list = []
    frame_content_landmark_list = []
    frame_face_list = []
    frame_sketch_list = []
    lip_dist_list = []

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
    ) as face_mesh, tqdm(enumerate(frames), total=len(frames), desc="extracting landmark from video frames") as t:
        for frame_idx, full_frame in t:
            h, w = full_frame.shape[0], full_frame.shape[1]
            results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                raise RuntimeError("face not detected on frame #{}".format(frame_idx))
            face_landmarks = results.multi_face_landmarks[0]

            ## calculate the lip dist
            dx = face_landmarks.landmark[lip_index[0]].x - face_landmarks.landmark[lip_index[1]].x
            dy = face_landmarks.landmark[lip_index[0]].y - face_landmarks.landmark[lip_index[1]].y
            dist = np.linalg.norm((dx, dy))
            lip_dist_list.append((frame_idx, dist))

            # (1)normalize landmarks
            x_min = 999
            x_max = -999
            y_min = 999
            y_max = -999
            pose_landmarks, content_landmarks = [], []
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in all_landmark_idx:
                    if landmark.x < x_min:
                        x_min = landmark.x
                    if landmark.x > x_max:
                        x_max = landmark.x

                    if landmark.y < y_min:
                        y_min = landmark.y
                    if landmark.y > y_max:
                        y_max = landmark.y
                ######
                if idx in pose_landmark_idx:
                    pose_landmarks.append((idx, landmark.x, landmark.y))
                if idx in content_landmark_idx:
                    content_landmarks.append((idx, landmark.x, landmark.y))
            ##########plus 5 pixel to size##########
            x_min = max(x_min - padding_pixels / w, 0)
            x_max = min(x_max + padding_pixels / w, 1)
            #
            y_min = max(y_min - padding_pixels / h, 0)
            y_max = min(y_max + padding_pixels / h, 1)
            face_frame = cv2.resize(full_frame[int(y_min * h):int(y_max * h), int(x_min * w):int(x_max * w)],
                                    (128, 128))

            # update landmarks
            pose_landmarks = [ \
                [idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in pose_landmarks]
            content_landmarks = [ \
                [idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in content_landmarks]
            # update drawed landmarks
            for idx, x, y in pose_landmarks + content_landmarks:
                face_landmarks.landmark[idx].x = x
                face_landmarks.landmark[idx].y = y
            # draw sketch
            h_new = (y_max - y_min) * h
            w_new = (x_max - x_min) * w
            annotated_image = np.zeros((int(h_new * 128 / min(h_new, w_new)), int(w_new * 128 / min(h_new, w_new)), 3))
            draw_landmarks_from_preprocess_video(
                image=annotated_image,
                landmark_list=face_landmarks,  # FACEMESH_CONTOURS  FACEMESH_LIPS
                connections=FACEMESH_FULL,
                connection_drawing_spec=drawing_spec)  # landmark_drawing_spec=None,
            annotated_image = cv2.resize(annotated_image, (128, 128))

            # store
            frame_pose_landmark_list.append(pose_landmarks)
            frame_content_landmark_list.append(content_landmarks)
            frame_face_list.append(face_frame)
            frame_sketch_list.append(annotated_image)

        return frame_pose_landmark_list, frame_content_landmark_list, frame_face_list, frame_sketch_list, lip_dist_list


def get_mel_spectrogram_from_file(filename: str, sample_rate: int) -> np.ndarray:
    """
    input file must be 16kHz and wav format
    """
    wav = audio.load_wav(filename, sample_rate)
    return audio.melspectrogram(wav).T


def get_smoothened_landmarks(all_landmarks, windows_T=1):
    for i in range(len(all_landmarks)):  # frame i
        if i + windows_T > len(all_landmarks):
            window = all_landmarks[len(all_landmarks) - windows_T:]
        else:
            window = all_landmarks[i: i + windows_T]
        #####
        for j in range(len(all_landmarks[i])):  # landmark j
            all_landmarks[i][j][1] = np.mean([frame_landmarks[j][1] for frame_landmarks in window])  # x
            all_landmarks[i][j][2] = np.mean([frame_landmarks[j][2] for frame_landmarks in window])  # y
    return all_landmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", "-v", type=str, required=True)
    parser.add_argument("--input-wav", "-w", type=str, default=None)
    parser.add_argument("--checkpoint", "-c", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-name", "-n", type=str, default=None)
    parser.add_argument("--smooth", action="store_true", default=False)
    parser.add_argument("--padding", type=int, default=5)
    # parser.add_argument("--chunk-mel", action="store_true", default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_output_dir = args.output
    if base_output_dir is None:
        base_output_dir = os.path.join(
            os.path.expanduser("~"),
            "data",
            "IP_LAP",
            "infer_output",
        )
    preprocess_output = os.path.join(
        base_output_dir,
        args.input_video.replace("/", "_"),
    )
    face_output_dir = os.path.join(preprocess_output, "faces")
    os.makedirs(face_output_dir, exist_ok=True)
    sketch_output_dir = os.path.join(preprocess_output, "sketches")
    os.makedirs(sketch_output_dir, exist_ok=True)
    face_and_sketch_output_dir = os.path.join(preprocess_output, "face_with_sketch")
    os.makedirs(face_and_sketch_output_dir, exist_ok=True)

    landmark_output_path = os.path.join(preprocess_output, "landmarks-pp_{}".format(args.padding))

    # extract audio from video if audio path not provided
    if args.input_wav is None:
        args.input_wav = os.path.join(preprocess_output, "audio.wav")
        extract_audio_from_video(args.input_video, args.input_wav)

    # extract frames from video
    if os.path.exists(landmark_output_path):
        print("loading landmarks from {}".format(landmark_output_path))
        with open(landmark_output_path, "rb") as f:
            frame_pose_landmark_list, frame_content_landmark_list, frame_face_list, frame_sketch_list, lip_dist_list = pickle.load(
                f)
    else:
        frame_pose_landmark_list, frame_content_landmark_list, frame_face_list, frame_sketch_list, lip_dist_list = extract_landmark_from_video(
            args.input_video, padding_pixels=args.padding)
        with tqdm(range(len(frame_face_list)), desc="saving frame landmarks") as t:
            for i in t:
                cv2.imwrite(
                    os.path.join(face_output_dir, "{:06d}.jpg".format(i)), frame_face_list[i],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )
                cv2.imwrite(
                    os.path.join(sketch_output_dir, "{:06d}.jpg".format(i)),
                    frame_sketch_list[i],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )
                cv2.imwrite(
                    os.path.join(face_and_sketch_output_dir, "{:06d}.jpg".format(i)),
                    np.concatenate([frame_face_list[i], frame_sketch_list[i]], axis=1),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )
        with open(landmark_output_path + ".tmp", "wb") as f:
            pickle.dump((frame_pose_landmark_list, frame_content_landmark_list, frame_face_list, frame_sketch_list,
                         lip_dist_list), f)
        os.rename(landmark_output_path + ".tmp", landmark_output_path)

    # smooth landmarks
    if args.smooth:
        print("smooth landmarks")
        print("smoothing pose landmarks")
        frame_pose_landmark_list = get_smoothened_landmarks(frame_pose_landmark_list, windows_T=1)
        print("smoothing content landmarks")
        frame_content_landmark_list = get_smoothened_landmarks(frame_content_landmark_list, windows_T=1)

    video_frame_count = len(frame_face_list)

    # get Mel Spectrogram from audio
    print("extracting mel spectrogram from wav")
    mel = get_mel_spectrogram_from_file(args.input_wav, AUDIO_SAMPLE_RATE)
    # mel_chunks = None
    # if args.chunk_mel is True:
    print("mel chunk")
    mel_chunks = []  # each mel chunk correspond to 5 video frames, used to generate one video frame
    mel_idx_multiplier = 80. / fps
    mel_chunk_idx = 0
    mel = mel.T
    while 1:
        start_idx = int(mel_chunk_idx * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])  # mel for generate one video frame
        mel_chunk_idx += 1

    # randomly select N_l reference landmarks for landmark transformer
    ## select Nl frames
    dists_sorted = sorted(lip_dist_list, key=lambda x: x[1])
    lip_dist_idx = np.asarray([idx for idx, dist in dists_sorted])  # the frame idxs sorted by lip openness
    Nl_idxs = [lip_dist_idx[int(i)] for i in torch.linspace(0, video_frame_count - 1, steps=Nl)]
    print("selected reference {} frames: {}".format(Nl, Nl_idxs))

    Nl_pose_landmarks, Nl_content_landmarks = [], []  # Nl_pose + Nl_content=Nl

    ## extract landmarks from selected frame
    for reference_idx in Nl_idxs:
        frame_pose_landmarks = frame_pose_landmark_list[reference_idx]
        frame_content_landmarks = frame_content_landmark_list[reference_idx]
        Nl_pose_landmarks.append(frame_pose_landmarks)
        Nl_content_landmarks.append(frame_content_landmarks)

    ## build landmark tensors
    Nl_pose = torch.zeros((Nl, 2, 74))  # 74 landmark
    Nl_content = torch.zeros((Nl, 2, 57))  # 57 landmark
    for idx in range(Nl):
        Nl_pose_landmarks[idx] = sorted(Nl_pose_landmarks[idx],
                                        key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
        Nl_content_landmarks[idx] = sorted(Nl_content_landmarks[idx],
                                           key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))

        Nl_pose[idx, 0, :] = torch.FloatTensor(
            [Nl_pose_landmarks[idx][i][1] for i in range(len(Nl_pose_landmarks[idx]))])  # x
        Nl_pose[idx, 1, :] = torch.FloatTensor(
            [Nl_pose_landmarks[idx][i][2] for i in range(len(Nl_pose_landmarks[idx]))])  # y

        Nl_content[idx, 0, :] = torch.FloatTensor(
            [Nl_content_landmarks[idx][i][1] for i in range(len(Nl_content_landmarks[idx]))])  # x
        Nl_content[idx, 1, :] = torch.FloatTensor(
            [Nl_content_landmarks[idx][i][2] for i in range(len(Nl_content_landmarks[idx]))])  # y

    Nl_content = Nl_content.unsqueeze(0)  # (1,Nl, 2, 57)
    Nl_pose = Nl_pose.unsqueeze(0)  # (1,Nl,2,74)

    # load model
    d_model = 512
    dim_feedforward = 1024
    nlayers = 4
    nhead = 4
    dropout = 0.1  # 0.5
    model = Landmark_transformer(T, d_model, nlayers, nhead, dim_feedforward, dropout)
    print("load checkpoint {}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    model_dict = model.state_dict()
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    state_dict_needed = {k: v for k, v in new_s.items() if k in model_dict.keys()}  # we need in model
    model_dict.update(state_dict_needed)
    model.load_state_dict(model_dict)
    model = model.cuda()
    model.eval()

    # infer for each `T` frames
    with torch.no_grad():
        experiment_name = args.experiment_name
        if experiment_name is None:
            experiment_name = args.input_wav.replace("/", "_")
        base_output_dir = os.path.join(
            preprocess_output,
            "experiments",
            experiment_name,
            str(int(time.time()))
        )

        predicted_T_landmark_output_dir = os.path.join(base_output_dir, "predicted_T_landmarks")
        os.makedirs(predicted_T_landmark_output_dir)
        predicted_frame_landmark_output_dir = os.path.join(base_output_dir, "predicted_frame_landmarks")
        os.makedirs(predicted_frame_landmark_output_dir)

        loss_fn = torch.nn.L1Loss()

        # with tqdm(enumerate(range(0, video_frame_count - 5)), desc="predicting landmarks") as t, open(
        #         os.path.join(output_dir, "metrics.txt"), "w") as metric:
        with tqdm(enumerate(range(0, len(mel_chunks) - 2)), desc="predicting landmarks", total=len(mel_chunks) - 2) as t, open(
                os.path.join(predicted_T_landmark_output_dir, "metrics.txt"), "w") as metric:
            for batch_idx, i in t:
                # select 5 frames from current frame (include current frame)
                T_idxs = list(range(i, i + T))
                T_idxs = [i % video_frame_count for i in T_idxs]  # make sure not overflow
                # if T_idxs[-1] >= video_frame_count:
                #     break
                t.set_description("predicting frame #{} with pose landmarks from frames #{}".format(batch_idx, T_idxs))

                # get landmarks of T frames
                T_pose_landmarks, T_content_landmarks = [], []
                for frame_idx in T_idxs:
                    T_pose_landmarks.append(frame_pose_landmark_list[frame_idx])
                    T_content_landmarks.append(frame_content_landmark_list[frame_idx])
                # build landmarks as tensor
                T_pose = torch.zeros((T, 2, 74))  # 74 landmark
                T_content = torch.zeros((T, 2, 57))  # 57 landmark
                for idx in range(T):
                    T_pose_landmarks[idx] = sorted(T_pose_landmarks[idx],
                                                   key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
                    T_content_landmarks[idx] = sorted(T_content_landmarks[idx],
                                                      key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))

                    T_pose[idx, 0, :] = torch.FloatTensor(
                        [T_pose_landmarks[idx][i][1] for i in range(len(T_pose_landmarks[idx]))])  # x
                    T_pose[idx, 1, :] = torch.FloatTensor(
                        [T_pose_landmarks[idx][i][2] for i in range(len(T_pose_landmarks[idx]))])  # y

                    T_content[idx, 0, :] = torch.FloatTensor(
                        [T_content_landmarks[idx][i][1] for i in range(len(T_content_landmarks[idx]))])  # x
                    T_content[idx, 1, :] = torch.FloatTensor(
                        [T_content_landmarks[idx][i][2] for i in range(len(T_content_landmarks[idx]))])  # y
                T_pose = T_pose.unsqueeze(0)
                T_content = T_content.unsqueeze(0)

                # get mel
                T_mels = []
                # if args.chunk_mel is False:
                #     for frame_idx in T_idxs:
                #         mel_start_frame_idx = frame_idx - 2  ###around the frame
                #         if mel_start_frame_idx < 0:
                #             mel_start_frame_idx = 0
                #         start_idx = int(80. * (mel_start_frame_idx / float(fps)))
                #         m = mel[start_idx: start_idx + mel_step_size, :]  # get five frames around
                #         if m.shape[0] != mel_step_size:  # in the end of vid
                #             break
                #         T_mels.append(m.T)  # transpose
                #
                #     if len(T_mels) != T:
                #         break
                # else:
                for mel_chunk_idx in T_idxs:
                    T_mels.append(mel_chunks[max(0, mel_chunk_idx - 2)])
                # if len(T_mels) != T:
                #     break

                T_mels = np.asarray(T_mels)  # (T,hv,wv)
                T_mels = torch.FloatTensor(T_mels).unsqueeze(1)  # (T,1,hv,wv)
                T_mels = T_mels.unsqueeze(0)

                Nl_pose = Nl_pose.cuda()  # (1, Nl, 2, 74)
                Nl_content = Nl_content.cuda()  # (1, Nl, 2, 57)
                T_pose = T_pose.cuda()  # (1, T, 2, 74)
                T_content = T_content.cuda()  # (1, T, 2, 57)
                T_mels = T_mels.cuda()  # (1, T, 1, 80, mel_step_size=16)

                # infer
                predict_content = model(T_mels, T_pose, Nl_pose, Nl_content)  # (1*T, 2, 57)
                loss = loss_fn(predict_content, T_content.squeeze(0)).item()

                # save predict
                metric.write("{:06d}-{:06d}: {}\n".format(T_idxs[0], T_idxs[-1], loss))
                T_pose = T_pose.squeeze(0).cpu().numpy()
                predict_content = predict_content.cpu().numpy()

                T_face_list = []
                T_sketch_list = []
                predict_sketch_list = []
                for idx, i in enumerate(T_idxs):
                    T_face_list.append(frame_face_list[i])
                    T_sketch_list.append(frame_sketch_list[i])
                    predict_sketch_list.append(draw_pose_and_content(T_pose[idx], predict_content[idx]))

                predict_visualized = np.concatenate([
                    np.concatenate(T_face_list, axis=1),  # concat T faces
                    np.concatenate(T_sketch_list, axis=1),  # concat T input frames
                    np.concatenate(predict_sketch_list, axis=1),  # concat T predicted landmarks
                ], axis=0)
                cv2.imwrite(os.path.join(predicted_T_landmark_output_dir, "{:06d}.jpg".format(batch_idx)), predict_visualized)
    assert subprocess.call([
        "ffmpeg",
        "-y",
        "-i",
        os.path.join(predicted_T_landmark_output_dir, "%06d.jpg"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        os.path.join(predicted_T_landmark_output_dir, "video-without-audio.mp4"),
    ]) == 0
    assert subprocess.call([
        "ffmpeg",
        "-y",
        "-i",
        os.path.join(predicted_T_landmark_output_dir, "video-without-audio.mp4"),
        "-i",
        args.input_wav,
        os.path.join(predicted_T_landmark_output_dir, "video.mp4"),
    ]) == 0
    print("infer results saved to {}".format(base_output_dir))


main()
