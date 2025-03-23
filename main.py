import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import mediapipe as mp
from PIL import Image
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import av
import threading
import os
from typing import Union


import matplotlib.pyplot as plt

st.title("Mesure corp humain !")

# Initialisation de MediaPipe Pose
# Initialiser PoseLandmarker
model_path_pose = os.path.abspath('pose_landmarker.task')  # Assurez-vous que ce fichier existe
base_options = python.BaseOptions(model_asset_path=model_path_pose)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

def plus_longue_sequence_non_nuls(row_pecs):
    # Trouver les indices des éléments non nuls
    nonzero_indices = [index for index, value in enumerate(row_pecs) if value > 0]

    # Initialiser les variables pour suivre la séquence actuelle et la plus longue séquence
    max_sequence = []
    current_sequence = []

    # Parcourir les indices non nuls pour trouver la plus longue séquence
    for i in range(len(nonzero_indices)):
        if i == 0 or nonzero_indices[i] == nonzero_indices[i - 1] + 1:
            current_sequence.append(nonzero_indices[i])
        else:
            if len(current_sequence) > len(max_sequence):
                max_sequence = current_sequence
            current_sequence = [nonzero_indices[i]]

    # Vérifier une dernière fois à la fin de la boucle
    if len(current_sequence) > len(max_sequence):
        max_sequence = current_sequence

    return max_sequence

class VideoTransformer(VideoTransformerBase):

    def __init__(self, detector):
        self.detector = detector
        self.frame_lock = threading.Lock()
        self.out_image = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        frame2 = img.copy()
        # # Détection avec le nouveau modèle
        detection_result = self.detector.detect(mp_image)

        if detection_result.pose_landmarks:
            for landmarks in detection_result.pose_landmarks:
                for landmark in landmarks:
                    h, w, _ = frame2.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame2, (x, y), 5, (0, 255, 0), -1)  # Afficher les landmarks en vert
        
        with self.frame_lock:
            self.out_image = img
        
        return av.VideoFrame.from_ndarray(frame2, format="bgr24")

# Interface Streamlit

ctx = webrtc_streamer(
    key="streamer",
    video_processor_factory=lambda: VideoTransformer(detector),
    sendback_audio=False
)

if ctx and ctx.video_transformer:

    snap = st.button("Capture")

    if snap:
        with ctx.video_transformer.frame_lock:
            out_image = ctx.video_transformer.out_image
        
        if out_image is not None:

            st.image(out_image, channels="BGR")
            image_rgb = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

            h, w, _ = out_image.shape

            # Afficher l'image originale
            plt.figure(figsize=(40, 30))
            plt.subplot(1, 4, 1)
            plt.imshow(image_rgb)
            plt.title("Image originale")
            plt.axis("off")

            # Détecter les landmarks et le masque de segmentation
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)
            segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
            binary_mask = (segmentation_mask > 0.8).astype(np.uint8) * 255

            # Afficher le masque de segmentation
            plt.subplot(1, 4, 1)
            plt.imshow(binary_mask, cmap='gray')
            plt.title("Masque de segmentation")
            plt.axis("off")

            # Récupérer les points clés du corps
            if detection_result.pose_landmarks:
                pose_landmarks = detection_result.pose_landmarks[0]

                # Points clés pour les hanches et les épaules
                hip_left = pose_landmarks[23]  # Hanche gauche
                hip_right = pose_landmarks[24]  # Hanche droite
                shoulder_left = pose_landmarks[11]  # Épaule gauche
                shoulder_right = pose_landmarks[12]  # Épaule droite

                # Points pour mesurer la taille du torse au-dessus des pectoraux
                upper_chest = pose_landmarks[0]  # Partie supérieure du torse

                # Convertir les coordonnées normalisées en pixels
                x_hip_left, y_hip_left = int(hip_left.x * w), int(hip_left.y * h)
                x_hip_right, y_hip_right = int(hip_right.x * w), int(hip_right.y * h)
                x_shoulder_left, y_shoulder_left = int(shoulder_left.x * w), int(shoulder_left.y * h)
                x_shoulder_right, y_shoulder_right = int(shoulder_right.x * w), int(shoulder_right.y * h)
                x_upper_chest, y_upper_chest = int(upper_chest.x * w), int(upper_chest.y * h)

                # Calculer la position des mesures
                y_hips = (y_hip_left + y_hip_right) // 2  # Milieu des hanches
                y_upper_pecs = int(y_upper_chest + (y_hips - y_upper_chest) * 0.55)  # 40% en dessous des épaules pour les pectoraux

                y_hips = int(y_upper_chest + (y_hips - y_upper_chest) * 0.9)  # Milieu des hanches

                y_bides = int(y_upper_chest + (y_hips - y_upper_chest) * 0.85)  # Milieu des hanches

                # Trouver la largeur du corps au niveau des hanches
                row_hips = binary_mask[y_hips, :]
                nonzero_hips = plus_longue_sequence_non_nuls(row_hips)
                if len(nonzero_hips) > 0:
                    x_hip_left_px, x_hip_right_px = nonzero_hips[0], nonzero_hips[-1]
                    hip_width_px = x_hip_right_px - x_hip_left_px

                # Trouver la largeur du corps au niveau du ventre
                row_bides = binary_mask[y_bides, :]
                nonzero_bides = plus_longue_sequence_non_nuls(row_bides)
                if len(nonzero_bides) > 0:
                    x_bide_left_px, x_bide_right_px = nonzero_bides[0], nonzero_bides[-1]
                    bide_width_px = x_bide_right_px - x_bide_left_px

                # Trouver la largeur du corps au niveau des pectoraux
                row_pecs = binary_mask[y_upper_pecs, :]
                nonzero_pecs = plus_longue_sequence_non_nuls(row_pecs)
                if len(nonzero_pecs) > 0:
                    x_pec_left_px, x_pec_right_px = nonzero_pecs[0], nonzero_pecs[-1]
                    pec_width_px = x_pec_right_px - x_pec_left_px

                # Calculer la distance entre les yeux et l'utiliser comme référence pour les mesures
                # Points des yeux
                left_eye = pose_landmarks[5]
                right_eye = pose_landmarks[2]

                # Convertir les coordonnées des yeux en pixels
                x_left_eye, y_left_eye = int(left_eye.x * w), int(left_eye.y * h)
                x_right_eye, y_right_eye = int(right_eye.x * w), int(right_eye.y * h)

                # Calculer la distance entre les yeux
                eye_distance_px = np.sqrt((x_right_eye - x_left_eye) ** 2 + (y_right_eye - y_left_eye) ** 2)
        
                # Distance moyenne entre les yeux en cm (environ 6,5 cm pour un adulte)
                eye_distance_cm = 6.5

                # Calculer le ratio pixels/cm
                ratio_px_per_cm = eye_distance_px / eye_distance_cm

                # Convertir les mesures en centimètres
                hip_width_cm = hip_width_px / ratio_px_per_cm
                pec_width_cm = pec_width_px / ratio_px_per_cm
                bide_width_cm = bide_width_px / ratio_px_per_cm

                # Dessiner les lignes de mesure sur l'image
                cv2.line(out_image, (nonzero_hips[0], y_hips), (nonzero_hips[-1], y_hips), (255, 0, 0), 4)
                cv2.putText(out_image, f"Hanches: {hip_width_cm:.2f} cm", (nonzero_hips[0], y_hips - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                
                cv2.line(out_image, (nonzero_bides[0], y_bides), (nonzero_bides[-1], y_bides), (255, 165, 0), 4)
                cv2.putText(out_image, f"Ventre: {bide_width_cm:.2f} cm", (nonzero_bides[0], y_bides - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 3)

                cv2.line(out_image, (nonzero_pecs[0], y_upper_pecs), (nonzero_pecs[-1], y_upper_pecs), (0, 255, 0), 4)
                cv2.putText(out_image, f"Pecs: {pec_width_cm:.2f} cm", (nonzero_pecs[0], y_upper_pecs - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # Afficher les landmarks utilisés
                landmarks_image = out_image.copy()
                keypoints = [(x_hip_left, y_hip_left), (x_hip_right, y_hip_right), 
                            (x_shoulder_left, y_shoulder_left), (x_shoulder_right, y_shoulder_right), 
                            (x_left_eye, y_left_eye), (x_right_eye, y_right_eye)]

                for x, y in keypoints:
                    cv2.circle(landmarks_image, (x, y), 5, (0, 0, 255), -1)  # Cercle rouge pour chaque landmark

                # Afficher l'image avec les landmarks
                plt.subplot(2, 4, 1)
                plt.imshow(cv2.cvtColor(landmarks_image, cv2.COLOR_BGR2RGB))
                plt.title("Landmarks utilisés")
                plt.axis("off")

            # Afficher l'image avec les mesures
            plt.subplot(2, 4, 1)
            plt.imshow(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
            plt.title("Mesures superposées")
            plt.axis("off")

            st.pyplot(plt)

        else:
            st.warning("No frames available yet.")
