import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

import matplotlib.pyplot as plt

# Initialisation de MediaPipe Pose
# Initialiser PoseLandmarker
model_path_pose = os.path.abspath('pose_landmarker.task')  # Assurez-vous que ce fichier existe
base_options = python.BaseOptions(model_asset_path=model_path_pose)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Initialisation de session_state
if "saved_image" not in st.session_state:
    st.session_state.saved_image = None

if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = True

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

def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Webcam Display Streamlit App")
    st.caption("Powered by OpenCV & Streamlit")

    frame_placeholder = st.empty()

    # Boutons Streamlit
    start_button = st.button("Démarrer la webcam")
    stop_button = st.button("Arrêter la webcam")
    capture_button = st.button("Capturer")

    # Gérer l'état de la webcam
    if start_button:
        st.session_state.webcam_active = True
    if stop_button:
        st.session_state.webcam_active = False

    # Ouvrir la webcam uniquement si elle est active
    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(-1)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Échec de la capture vidéo.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            frame2 = frame.copy()
            # Détection avec le nouveau modèle
            detection_result = detector.detect(mp_image)

            if detection_result.pose_landmarks:
                for landmarks in detection_result.pose_landmarks:
                    for landmark in landmarks:
                        h, w, _ = frame2.shape
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame2, (x, y), 5, (0, 255, 0), -1)  # Afficher les landmarks en vert

            frame_placeholder.image(frame2, channels="RGB")

            # Capture d'image au clic du bouton
            if capture_button:
                st.session_state.saved_image = frame.copy()
                st.success("Image capturée !")

            # Si on appuie sur "Arrêter", on quitte la boucle
            if not st.session_state.webcam_active:
                break

        cap.release()
        cv2.destroyAllWindows()


    # Afficher l'image capturée après la fin de la webcam
    if st.session_state.saved_image is not None:

        st.image(st.session_state.saved_image, caption="Image Capturée", channels="RGB")

        # Sauvegarde de l'image
        img = Image.fromarray(st.session_state.saved_image)
        img.save("captured_image.jpg")
        image = cv2.imread("captured_image.jpg")
        if image is None:
            raise FileNotFoundError(f"Impossible de charger l'image à partir de captured_image.jpg.")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Afficher l'image originale
        plt.figure(figsize=(30, 20))
        plt.subplot(1, 4, 1)
        plt.imshow(image_rgb)
        plt.title("Image originale")
        plt.axis("off")

        # Détecter les landmarks et le masque de segmentation
        mp_image = mp.Image.create_from_file("captured_image.jpg")
        detection_result = detector.detect(mp_image)
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        binary_mask = (segmentation_mask > 0.8).astype(np.uint8) * 255

        # Afficher le masque de segmentation
        plt.subplot(1, 4, 2)
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
            cv2.line(image, (nonzero_hips[0], y_hips), (nonzero_hips[-1], y_hips), (255, 0, 0), 4)
            cv2.putText(image, f"Hanches: {hip_width_cm:.2f} cm", (nonzero_hips[0], y_hips - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            
            cv2.line(image, (nonzero_bides[0], y_bides), (nonzero_bides[-1], y_bides), (255, 165, 0), 4)
            cv2.putText(image, f"Ventre: {bide_width_cm:.2f} cm", (nonzero_bides[0], y_bides - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 3)

            cv2.line(image, (nonzero_pecs[0], y_upper_pecs), (nonzero_pecs[-1], y_upper_pecs), (0, 255, 0), 4)
            cv2.putText(image, f"Pecs: {pec_width_cm:.2f} cm", (nonzero_pecs[0], y_upper_pecs - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Afficher les landmarks utilisés
            landmarks_image = image.copy()
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
        plt.subplot(2, 4, 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Mesures superposées")
        plt.axis("off")

        st.pyplot(plt)

        # img.save("captured_image.jpg")
        # st.success("Image sauvegardée sous 'captured_image.jpg'.")

if __name__ == "__main__":
    main()
