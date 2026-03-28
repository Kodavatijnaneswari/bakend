import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.core.files.storage import default_storage
from admins.models import modeldata
from .models import DiagnosticResult
from .serializers import DiagnosticResultSerializer, UserSerializer
from ultralytics import YOLO
import cv2
import numpy as np

# Load model once at module level
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
        
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'runs/detect/optimized_bone_model/weights/best.pt')
    try:
        if os.path.exists(MODEL_PATH):
            _model = YOLO(MODEL_PATH)
            return _model
    except:
        pass
    return None

class DetectionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": "No image uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_image = request.FILES["image"]
        image_path = default_storage.save(f"uploads/{uploaded_image.name}", uploaded_image)
        image_full_path = os.path.join(settings.MEDIA_ROOT, image_path)

        try:
            model = get_model()
            img = cv2.imread(image_full_path)
            if img is None:
                return Response({"error": "Invalid image format"}, status=status.HTTP_400_BAD_REQUEST)

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
                    "error": "Non-X-ray image detected. Please upload an original grayscale medical X-ray."
                }, status=status.HTTP_400_BAD_REQUEST)

            if model is None:
                return Response({"error": "AI Diagnostic Engine not initialized"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # --- YOLO Inference ---
            results = model.predict(source=image_full_path, save=False, conf=0.10)
            boxes = results[0].boxes
            if not boxes:
                results = model.predict(source=image_full_path, save=False, conf=0.03)
                boxes = results[0].boxes

            fracture_boxes = [box for box in boxes if int(box.cls[0]) in [0, 1, 2, 3, 4, 5, 6]]

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
            
            for box in fracture_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                box_w, box_h = x2 - x1, y2 - y1
                area_ratio = (box_w * box_h) / (img.shape[0] * img.shape[1])
                aspect_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)

                if (cls_id in [0, 5, 6] and area_ratio > 0.04) or (aspect_ratio > 4.5):
                     current_stage = "Dislocated Fracture / Major Displacement"
                elif area_ratio > 0.06 or aspect_ratio > 3.0:
                     current_stage = "Complete Transverse Fracture"
                elif area_ratio < 0.008 or conf < 0.08:
                     current_stage = "Incomplete / Hairline Fracture (Suspected)"
                else:
                     current_stage = "Fracture Abnormality Detected"

                if conf == float(best_box.conf[0]):
                    stage = current_stage
                    if cls_id == 6: stage = f"Wrist {current_stage}"
                    if cls_id == 0: stage = f"Elbow {current_stage}"

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                sigma = max(box_w / 3.0, box_h / 3.0, 5.0)
                Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
                gauss = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
                heatmap = np.maximum(heatmap, gauss)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            alpha, mask = 0.4, heatmap > 20
            overlay[mask] = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)[mask]

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
