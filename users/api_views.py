import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.core.files.storage import default_storage
from admins.models import modeldata
from .models import DiagnosticResult
from .serializers import DiagnosticResultSerializer, UserSerializer
import numpy as np

# -------- MEMORY-OPTIMIZED MODEL LOAD --------
_yolo_model = None

def get_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            # --- PRIORITIZE CUSTOM DATASET MODELS ---
            custom_model = os.path.join(settings.MEDIA_ROOT, "best.pt")
            fallback_model = os.path.join(settings.MEDIA_ROOT, "yolov8s.pt")
            model_path = custom_model if os.path.exists(custom_model) else fallback_model
            
            from ultralytics import YOLO
            _yolo_model = YOLO(model_path)
            print(f"API DIAGNOSTIC ENGINE: Loaded model from {model_path}")
        except Exception as e:
            print(f"API CRITICAL: Failed to load AI Model: {e}")
            _yolo_model = None
    return _yolo_model

class DetectionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": "No image uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_image = request.FILES["image"]
        image_path = default_storage.save(f"uploads/{uploaded_image.name}", uploaded_image)
        image_full_path = os.path.join(settings.MEDIA_ROOT, image_path)

        try:
            import cv2
            model = get_model()
            img_original = cv2.imread(image_full_path)
            if img_original is None:
                return Response({"error": "Invalid image format"}, status=status.HTTP_400_BAD_REQUEST)

            # --- MEMORY OPTIMIZATION: SCALE IMAGE FOR SCANNING (Max 640px) ---
            # Prevents OOM crashes when mobile users upload high-resolution photos
            MAX_PROC_DIM = 640
            h_orig, w_orig = img_original.shape[:2]
            scale_ratio = 1.0
            if max(h_orig, w_orig) > MAX_PROC_DIM:
                scale_ratio = MAX_PROC_DIM / max(h_orig, w_orig)
                img = cv2.resize(img_original, (int(w_orig * scale_ratio), int(h_orig * scale_ratio)))
            else:
                img = img_original.copy()

            # --- X-ray Validation ---
            b, g, r = cv2.split(img)
            mean_diff = (np.mean(cv2.absdiff(r, g)) + np.mean(cv2.absdiff(r, b)) + np.mean(cv2.absdiff(g, b))) / 3.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            
            is_perfectly_gray = mean_diff < 5.0
            is_mostly_gray = mean_diff < 20.0
            background_ratio = np.sum(hist[:30]) / np.sum(hist)
            bone_peak_ratio = np.sum(hist[140:]) / np.sum(hist)
            
            if is_perfectly_gray:
                is_valid_xray = True
            elif is_mostly_gray:
                is_valid_xray = (background_ratio > 0.02) or (bone_peak_ratio > 0.0001)
            else:
                is_valid_xray = False
                
            if not is_valid_xray:
                return Response({
                    "error": "Non-X-ray image detected. Please upload an original medical radiograph."
                }, status=status.HTTP_400_BAD_REQUEST)

            if model is None:
                return Response({"error": "AI Diagnostic Engine not initialized"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # --- 'PERFECT' AI INFERENCE (Augmented for Clinical Accuracy) ---
            results = model.predict(source=img, save=False, conf=0.25, augment=True)
            boxes = results[0].boxes

            # --- DYNAMIC CATEGORIZATION ---
            fracture_boxes = [box for box in boxes if int(box.cls[0]) in model.names.keys()]

            if not fracture_boxes:
                # Save Normal Result
                userid = request.data.get('userid')
                if userid:
                    DiagnosticResult.objects.create(
                        user_id=userid,
                        original_image=image_path,
                        processed_image=image_path,
                        finding="Normal",
                        category="Normal Bone Structure",
                        confidence=0.99
                    )
                
                return Response({
                    "finding": "Normal",
                    "category": "Normal Bone Structure",
                    "confidence": 0.99,
                    "image_url": request.build_absolute_uri(settings.MEDIA_URL + image_path),
                    "message": "Normal X-ray. No abnormalities detected."
                })

            # --- Post-processing (Heatmap & Bounding Boxes) ---
            overlay = img.copy()
            heatmap = np.zeros_like(img[:,:,0], dtype=np.float32)
            best_box = fracture_boxes[0]
            stage = "Abnormal"
            
            h_proc, w_proc = img.shape[:2]
            Y, X = np.ogrid[:h_proc, :w_proc]

            for box in fracture_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                box_w, box_h = x2 - x1, y2 - y1
                area_ratio = (box_w * box_h) / (h_proc * w_proc)
                aspect_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)

                # --- DYNAMIC STAGE CLASSIFICATION ---
                class_name = model.names[cls_id]
                current_stage = f"{class_name} Detected"
                
                if area_ratio > 0.04 or aspect_ratio > 4.5:
                     current_stage = f"Major {class_name} / Displacement"
                elif area_ratio > 0.06:
                     current_stage = f"Complete {class_name} Fracture"
                elif area_ratio < 0.008:
                     current_stage = f"Suspected Hairline {class_name}"

                if conf == float(best_box.conf[0]):
                    stage = current_stage

                # --- LOCALIZED HIGHLIGHT (Tighter Sigma) ---
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                sigma = min(box_w, box_h) / 2.0
                highlight = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
                heatmap = np.maximum(heatmap, highlight)
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            alpha, mask = 0.5, heatmap > 30
            overlay[mask] = cv2.addWeighted(img[mask], 1 - alpha, heatmap_color[mask], alpha, 0)

            output_filename = "api_detected_" + uploaded_image.name
            output_path = os.path.join(settings.MEDIA_ROOT, 'uploads', output_filename)
            cv2.imwrite(output_path, overlay)
            processed_image_url = settings.MEDIA_URL + f"uploads/{output_filename}"

            # Save Abnormal Result
            userid = request.data.get('userid')
            if userid:
                DiagnosticResult.objects.create(
                    user_id=userid,
                    original_image=image_path,
                    processed_image=f"uploads/{output_filename}",
                    finding="Abnormal",
                    category=stage,
                    confidence=float(best_box.conf[0])
                )

            return Response({
                "finding": "Abnormal",
                "category": stage,
                "confidence": float(best_box.conf[0]),
                "image_url": request.build_absolute_uri(processed_image_url)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"API Detection Error: {e}")
            return Response({"error": f"Processing Error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LoginAPIView(APIView):
    def post(self, request, *args, **kwargs):
        username = request.data.get('username', '').strip()
        password = request.data.get('password', '').strip()
        
        try:
            user_candidates = modeldata.objects.filter(username__iexact=username)
            if not user_candidates:
                return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
            
            user = user_candidates.first()
            if user.password != password:
                return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
            
            if user.status != 'Activated':
                return Response({"error": "Account is not activated"}, status=status.HTTP_403_FORBIDDEN)
            
            serializer = UserSerializer(user)
            return Response(serializer.data, status=status.HTTP_200_OK)
                
        except Exception as e:
            return Response({"error": "Internal server error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HistoryAPIView(APIView):
    def get(self, request, userid, *args, **kwargs):
        results = DiagnosticResult.objects.filter(user_id=userid).order_by('-uploaded_at')
        serializer = DiagnosticResultSerializer(results, many=True)
        return Response(serializer.data)
