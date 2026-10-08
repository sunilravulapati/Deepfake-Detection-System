import cv2
import torch
from typing import Optional, Tuple, List, Dict, Any
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import time
from collections import Counter
import os

from pipeline import (
    RobustFaceDetector,
    get_detector,
    predict_deepfake as pipeline_predict,
    predict_deepfake_with_fallback,
    get_fallback_detector,
    TemporalAggregator,
    aggregate_video_predictions
)
from custom_fallback import DecisionMode

# Load Deep-Fake-Detector pre-trained model from Hugging Face
print("Loading AI model & detector...")
model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()  # Set to evaluation mode
face_detector = get_detector()
fallback_detector = get_fallback_detector()
print(f"Model, {face_detector.backend} detector, & QMC-FD fallback loaded successfully!")


def get_face_from_frame(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Extract faces from frame using the robust detector"""
    return face_detector.detect_faces(frame, max_faces=4)


def predict_deepfake(face_image: np.ndarray) -> Tuple[Optional[str], float]:
    """Predict if face is real or deepfake"""
    label, conf, _, _ = pipeline_predict(face_image, processor, model)
    return label, conf


def calculate_final_verdict(predictions: List[Dict[str, Any]]) -> Tuple[str, float, int, int]:
    """Calculate final verdict using weighted temporal confidence aggregation"""
    return aggregate_video_predictions(predictions)


def draw_analysis_ui(frame: np.ndarray, time_remaining: float, predictions: List[Dict[str, Any]], current_verdict: Optional[str], real_count: int, fake_count: int, avg_confidence: float, mode: str = "Live", current_mode_tag: str = "ViT") -> None:
    """Draw analysis UI on frame with QMC-FD decision mode indicator"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for stats
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (width - 10, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    title = f"DEEPFAKE ANALYSIS - {mode.upper()} MODE"
    cv2.putText(frame, title, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    
    # Decision Mode Tag
    mode_color = (100, 255, 100) if "VIT" in current_mode_tag.upper() else (100, 200, 255)
    cv2.putText(frame, f"Method: {current_mode_tag}", (max(20, width - 240), 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
    
    # Time remaining
    cv2.putText(frame, f"Time Remaining: {int(time_remaining)}s", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    
    # Predictions collected
    cv2.putText(frame, f"Predictions Collected: {len(predictions)}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Current stats
    cv2.putText(frame, f"Real: {real_count} | Fake: {fake_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Current verdict (if available)
    if current_verdict:
        verdict_color = (0, 255, 0) if current_verdict == "Realism" else (0, 0, 255)
        verdict_text = "REAL" if current_verdict == "Realism" else "FAKE"
        cv2.putText(frame, f"Leading: {verdict_text} ({avg_confidence*100:.1f}%)", (20, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, verdict_color, 2)


def draw_final_verdict(frame: np.ndarray, final_label: str, avg_confidence: float, real_count: int, fake_count: int, total_predictions: int, wait_text: str = "") -> None:
    """Draw final verdict overlay"""
    height, width = frame.shape[:2]
    
    # Create full overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Determine verdict display
    if final_label == "Realism":
        verdict_text = "REAL"
        verdict_color = (0, 255, 0)
        status = "AUTHENTIC"
    else:
        verdict_text = "FAKE"
        verdict_color = (0, 0, 255)
        status = "DEEPFAKE DETECTED"
    
    # Main verdict text
    cv2.putText(frame, "FINAL VERDICT", (width//2 - 200, height//2 - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.putText(frame, verdict_text, (width//2 - 100, height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 3, verdict_color, 5)
    
    cv2.putText(frame, status, (width//2 - 180, height//2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, verdict_color, 2)
    
    # Statistics
    cv2.putText(frame, f"Confidence: {avg_confidence*100:.2f}%", (width//2 - 150, height//2 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Analysis: {real_count} Real | {fake_count} Fake", (width//2 - 200, height//2 + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Total Predictions: {total_predictions}", (width//2 - 150, height//2 + 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if wait_text:
        cv2.putText(frame, wait_text, (width//2 - 220, height//2 + 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)


def run_video_file_detection(video_path: str) -> None:
    """Run deepfake detection on uploaded video file"""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo Properties:")
    print(f"- FPS: {fps}")
    print(f"- Total Frames: {total_frames}")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"\nStarting video analysis. Press 'q' to quit.\n")
    
    # Analysis parameters
    ANALYSIS_WINDOW = min(120, duration)  # Use video duration if less than 2 minutes
    FRAME_SKIP = max(1, fps // 3)  # Process ~3 frames per second
    
    # State variables
    predictions = []
    frame_count = 0
    fallback_detector.reset_temporal_state()
    current_mode_tag = "ViT"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        elapsed_time = frame_count / fps if fps > 0 else 0
        
        # Process frame for face detection
        face_data = get_face_from_frame(frame)
        
        # Draw face boxes
        for face_info in face_data:
            x, y, w, h = face_info['coords']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        # Run prediction on sampled frames
        if frame_count % FRAME_SKIP == 0 and len(face_data) > 0:
            best_face = face_data[0]
            face_img = best_face['image']
            eval_res = predict_deepfake_with_fallback(
                face_img, processor, model, quality_score=best_face.get('quality', 50.0),
                fallback_detector=fallback_detector,
                landmarks=best_face.get('landmarks'),
                frame_interval=FRAME_SKIP
            )
            
            label = eval_res['final_label']
            confidence = eval_res['final_confidence']
            current_mode_tag = eval_res['decision_mode']
            
            predictions.append({
                'label': label,
                'confidence': confidence,
                'p_real': eval_res['p_real'],
                'p_fake': eval_res['p_fake'],
                'quality': best_face.get('quality', 50.0),
                'timestamp': elapsed_time,
                'frame_index': frame_count,
                'decision_mode': eval_res['decision_mode'],
                'vit_confidence': eval_res.get('vit_confidence', 0.0),
                's_spatial': eval_res.get('s_spatial', 0.0),
                's_temporal': eval_res.get('s_temporal', 0.0),
                's_structural': eval_res.get('s_structural', 0.0)
            })
            print(f"Frame {frame_count}/{total_frames}: {label} ({confidence*100:.2f}%) [Method: {current_mode_tag}] [Quality: {best_face.get('quality', 0):.0f}] - Total: {len(predictions)}")
        
        # Calculate current leading verdict
        current_verdict, temp_confidence, temp_real, temp_fake = calculate_final_verdict(predictions)
        
        # Draw analysis UI
        time_remaining = ANALYSIS_WINDOW - elapsed_time
        draw_analysis_ui(frame, time_remaining, predictions, current_verdict, temp_real, temp_fake, temp_confidence, mode="Video", current_mode_tag=current_mode_tag)
        
        # Display frame
        cv2.imshow("Video Deepfake Detection - Press 'q' to quit", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate final verdict
    agg_res = TemporalAggregator.aggregate(predictions)
    final_label = agg_res['final_label']
    avg_confidence = agg_res['avg_confidence']
    real_count = agg_res['real_count']
    fake_count = agg_res['fake_count']
    
    print(f"\n{'='*50}")
    print(f"VIDEO ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Final Verdict: {final_label}")
    print(f"Confidence: {avg_confidence*100:.2f}%")
    print(f"Summary: {agg_res['verdict_summary']}")
    print(f"Real Count: {real_count}")
    print(f"Fake Count: {fake_count}")
    print(f"Total Predictions: {len(predictions)}")
    print(f"Fallback Frames: {agg_res.get('fallback_frames', 0)}")
    print(f"Decision Modes: {agg_res.get('mode_counts', {})}")
    print(f"{'='*50}\n")
    
    # Show final verdict for 10 seconds or until key press
    ret, last_frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, last_frame = cap.read()
    
    if ret:
        draw_final_verdict(last_frame, final_label, avg_confidence, real_count, fake_count, len(predictions), 
                          wait_text="Press any key to continue...")
        cv2.imshow("Video Deepfake Detection - Press 'q' to quit", last_frame)
        cv2.waitKey(10000)  # Wait 10 seconds or until key press
    
    cap.release()
    cv2.destroyAllWindows()


def run_live_webcam_detection() -> None:
    """Run deepfake detection with 2-minute analysis windows on webcam"""
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting 2-minute analysis detection. Press 'q' to quit.")
    
    # Analysis parameters
    ANALYSIS_WINDOW = 120  # 2 minutes in seconds
    VERDICT_DISPLAY_TIME = 5  # Show verdict for 5 seconds
    FRAME_SKIP = 6  # Process every 6th frame for responsive updates
    
    # State variables
    predictions = []
    start_time = time.time()
    frame_count = 0
    analysis_round = 1
    show_verdict = False
    verdict_start_time = 0
    fallback_detector.reset_temporal_state()
    current_mode_tag = "ViT"
    
    final_label = None
    avg_confidence = 0.0
    real_count = 0
    fake_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Show verdict screen
        if show_verdict:
            draw_final_verdict(frame, final_label, avg_confidence, real_count, fake_count, len(predictions),
                             wait_text="Starting new analysis in 5 seconds...")
            cv2.imshow("Live Deepfake Detection - Press 'q' to quit", frame)
            
            # Check if verdict display time is over
            if current_time - verdict_start_time >= VERDICT_DISPLAY_TIME:
                show_verdict = False
                predictions = []
                start_time = time.time()
                analysis_round += 1
                fallback_detector.reset_temporal_state()
                current_mode_tag = "ViT"
                print(f"\n--- Starting Analysis Round {analysis_round} ---")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Process frame for face detection
        face_data = get_face_from_frame(frame)
        
        # Draw face boxes
        for face_info in face_data:
            x, y, w, h = face_info['coords']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        # Run prediction on sampled frames
        if frame_count % FRAME_SKIP == 0 and len(face_data) > 0:
            best_face = face_data[0]
            face_img = best_face['image']
            eval_res = predict_deepfake_with_fallback(
                face_img, processor, model, quality_score=best_face.get('quality', 50.0),
                fallback_detector=fallback_detector,
                landmarks=best_face.get('landmarks'),
                frame_interval=FRAME_SKIP
            )
            
            label = eval_res['final_label']
            confidence = eval_res['final_confidence']
            current_mode_tag = eval_res['decision_mode']
            
            predictions.append({
                'label': label,
                'confidence': confidence,
                'p_real': eval_res['p_real'],
                'p_fake': eval_res['p_fake'],
                'quality': best_face.get('quality', 50.0),
                'timestamp': elapsed_time,
                'decision_mode': eval_res['decision_mode'],
                'vit_confidence': eval_res.get('vit_confidence', 0.0),
                's_spatial': eval_res.get('s_spatial', 0.0),
                's_temporal': eval_res.get('s_temporal', 0.0),
                's_structural': eval_res.get('s_structural', 0.0)
            })
            print(f"Frame {frame_count}: {label} ({confidence*100:.2f}%) [Method: {current_mode_tag}] - Total: {len(predictions)}")
        
        # Calculate current leading verdict
        current_verdict, temp_confidence, temp_real, temp_fake = calculate_final_verdict(predictions)
        
        # Draw analysis UI
        time_remaining = ANALYSIS_WINDOW - elapsed_time
        draw_analysis_ui(frame, time_remaining, predictions, current_verdict, temp_real, temp_fake, temp_confidence, mode="Live", current_mode_tag=current_mode_tag)
        
        # Check if analysis window is complete
        if elapsed_time >= ANALYSIS_WINDOW:
            # Calculate final verdict
            agg_res = TemporalAggregator.aggregate(predictions)
            final_label = agg_res['final_label']
            avg_confidence = agg_res['avg_confidence']
            real_count = agg_res['real_count']
            fake_count = agg_res['fake_count']
            
            print(f"\n{'='*50}")
            print(f"ANALYSIS ROUND {analysis_round} COMPLETE")
            print(f"{'='*50}")
            print(f"Final Verdict: {final_label}")
            print(f"Average Confidence: {avg_confidence*100:.2f}%")
            print(f"Real Count: {real_count}")
            print(f"Fake Count: {fake_count}")
            print(f"Total Predictions: {len(predictions)}")
            print(f"Fallback Frames: {agg_res.get('fallback_frames', 0)}")
            print(f"Decision Modes: {agg_res.get('mode_counts', {})}")
            print(f"{'='*50}\n")
            
            # Show verdict screen
            show_verdict = True
            verdict_start_time = time.time()
        
        # Display frame
        cv2.imshow("Live Deepfake Detection - Press 'q' to quit", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal analysis rounds completed: {analysis_round}")


def display_menu() -> str:
    """Display main menu and get user choice"""
    print("\n" + "="*60)
    print(" " * 15 + "DEEPFAKE DETECTION SYSTEM")
    print("="*60)
    print("\nChoose detection mode:")
    print("\n1. Live Webcam Detection (2-minute analysis windows)")
    print("2. Upload Video File Detection (full video analysis)")
    print("3. Exit")
    print("\n" + "="*60)
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def main() -> None:
    """Main function with menu system"""
    while True:
        choice = display_menu()
        
        if choice == '1':
            print("\n[Starting Live Webcam Detection...]")
            run_live_webcam_detection()
            
        elif choice == '2':
            print("\n[Upload Video File Detection]")
            video_path = input("Enter the full path to your video file: ").strip()
            # Remove quotes if user added them
            video_path = video_path.strip('"').strip("'")
            run_video_file_detection(video_path)
            
        elif choice == '3':
            print("\nExiting Deepfake Detection System. Goodbye!")
            break
        
        # Ask if user wants to continue
        if choice != '3':
            continue_choice = input("\nReturn to main menu? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\nExiting Deepfake Detection System. Goodbye!")
                break


if __name__ == "__main__":
    main()

