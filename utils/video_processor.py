import os
import logging
import tempfile
from typing import List
import subprocess

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_video(file_path: str) -> List[str]:
    """
    Process a video file and extract text content through frames and audio.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        List of text chunks extracted from the video
    """
    try:
        # Extract frames from video
        frame_texts = _extract_frames_and_process(file_path)
        
        # Extract and transcribe audio
        audio_text = _extract_and_transcribe_audio(file_path)
        
        # Combine results
        results = []
        
        if frame_texts:
            results.extend(frame_texts)
        
        if audio_text:
            results.append(f"Audio Transcript: {audio_text}")
        
        if not results:
            results.append("No content could be extracted from this video.")
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing video {file_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ["Error processing video file."]

def _extract_frames_and_process(file_path: str) -> List[str]:
    """Extract frames from video and process them for text content."""
    try:
        import cv2
        from utils.image_processor import process_image
        
        results = []
        with tempfile.TemporaryDirectory() as temp_dir:
            # Open the video file
            video = cv2.VideoCapture(file_path)
            
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Extract frames at intervals (1 frame every 5 seconds)
            interval = 5 * fps
            frame_num = 0
            
            frame_texts = []
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_num % int(interval) == 0:
                    # Save frame as image
                    frame_path = os.path.join(temp_dir, f"frame_{frame_num}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    # Process the frame image
                    text_chunks = process_image(frame_path)
                    
                    # Add non-empty results
                    for chunk in text_chunks:
                        if chunk and chunk.strip() and "No text content" not in chunk:
                            timestamp = frame_num / fps
                            minutes = int(timestamp / 60)
                            seconds = int(timestamp % 60)
                            frame_texts.append(f"[{minutes:02d}:{seconds:02d}] {chunk}")
                
                frame_num += 1
            
            # Release the video capture object
            video.release()
            
            # Combine similar consecutive frames
            if frame_texts:
                current_text = frame_texts[0]
                for i in range(1, len(frame_texts)):
                    # If texts are similar, combine them
                    if _similarity(current_text.split("] ")[1], frame_texts[i].split("] ")[1]) > 0.7:
                        continue
                    else:
                        results.append(current_text)
                        current_text = frame_texts[i]
                
                # Add the last text
                results.append(current_text)
        
        return results
    
    except ImportError:
        logger.warning("OpenCV not available, video frame extraction disabled")
        return []
    except Exception as e:
        logger.error(f"Frame extraction error: {str(e)}")
        return []

def _extract_and_transcribe_audio(file_path: str) -> str:
    """Extract audio from video and transcribe it."""
    try:
        # Check if ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("ffmpeg not available, audio extraction disabled")
            return ""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract audio using ffmpeg
            audio_path = os.path.join(temp_dir, "audio.mp3")
            subprocess.run(
                ["ffmpeg", "-i", file_path, "-q:a", "0", "-map", "a", audio_path],
                capture_output=True, 
                check=True
            )
            
            # Transcribe the audio
            transcription = _transcribe_audio(audio_path)
            return transcription
    
    except Exception as e:
        logger.error(f"Audio extraction error: {str(e)}")
        return ""

def _transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file to text."""
    try:
        # Currently, Anthropic API doesn't have direct audio transcription functionality
        # So we provide a simpler workaround here
        logger.info("Note: Anthropic API does not directly support audio transcription.")
        
        # Use a simpler approach - extract key frames and rely on those instead
        logger.info("Using extracted video frames for content instead of audio transcription")
        
        # Return a message explaining the limitation
        return "Audio transcription not available with current Anthropic API. Using video frame content instead."
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return f"Audio transcription error: {str(e)}"

def _similarity(text1: str, text2: str) -> float:
    """Calculate a simple similarity metric between two texts."""
    # A very basic Jaccard similarity
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0
