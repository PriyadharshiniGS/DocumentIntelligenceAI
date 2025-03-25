import os
import logging
import tempfile
import subprocess
import base64
from PIL import Image
import io
import numpy as np
import cv2
import anthropic
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(
    api_key=ANTHROPIC_API_KEY,
)

def process_video(file_path):
    """
    Process video file to extract key frames and analyze content.
    
    Args:
        file_path (str): Path to the video file
    
    Returns:
        list: List of text chunks containing extracted information
    """
    try:
        # Extract basic video metadata
        metadata = extract_video_metadata(file_path)
        
        # Extract key frames from the video
        frame_paths = extract_key_frames(file_path)
        
        # Analyze key frames with Claude Vision
        frame_descriptions = []
        for i, frame_path in enumerate(frame_paths):
            description = analyze_frame_with_claude(frame_path)
            frame_descriptions.append(f"Frame {i+1}: {description}")
        
        # Compile results
        results = []
        
        # Add metadata
        results.append(f"Video metadata:\n{metadata}")
        
        # Add frame descriptions
        if frame_descriptions:
            frame_analysis = "Key frames analysis:\n" + "\n\n".join(frame_descriptions)
            results.append(frame_analysis)
        
        # Add overall video summary
        video_summary = generate_video_summary(metadata, frame_descriptions)
        results.append(f"Video summary:\n{video_summary}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

def extract_video_metadata(file_path):
    """
    Extract basic metadata from video using OpenCV.
    
    Args:
        file_path (str): Path to the video file
    
    Returns:
        str: Video metadata information
    """
    try:
        cap = cv2.VideoCapture(file_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration in seconds
        duration_seconds = frame_count / fps if fps > 0 else 0
        minutes, seconds = divmod(duration_seconds, 60)
        
        # Format metadata as a string
        metadata = (
            f"Duration: {int(minutes)} minutes {int(seconds)} seconds\n"
            f"Frame rate: {fps:.2f} FPS\n"
            f"Resolution: {width}x{height} pixels\n"
            f"Total frames: {frame_count}"
        )
        
        cap.release()
        return metadata
    
    except Exception as e:
        logger.error(f"Error extracting video metadata: {str(e)}")
        return "Could not extract video metadata"

def extract_key_frames(file_path, max_frames=5):
    """
    Extract key frames from the video using scene detection.
    
    Args:
        file_path (str): Path to the video file
        max_frames (int): Maximum number of frames to extract
    
    Returns:
        list: List of paths to extracted key frames
    """
    try:
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate interval for uniform sampling
        if frame_count <= max_frames:
            frame_indices = list(range(frame_count))
        else:
            # Distribute frames evenly
            frame_indices = [int(i * frame_count / max_frames) for i in range(max_frames)]
        
        # Create temporary directory for frames
        frame_dir = tempfile.mkdtemp()
        frame_paths = []
        
        # Extract frames
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_path = os.path.join(frame_dir, f"frame_{i}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
        
        cap.release()
        return frame_paths
    
    except Exception as e:
        logger.error(f"Error extracting key frames: {str(e)}")
        return []

def analyze_frame_with_claude(frame_path):
    """
    Analyze a video frame using Claude Vision.
    
    Args:
        frame_path (str): Path to the frame image
    
    Returns:
        str: Description of the frame
    """
    try:
        # Read the image file
        with open(frame_path, "rb") as img_file:
            image_data = img_file.read()
        
        # Set up the Claude Vision request
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022", # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a frame from a video. Describe what you see in this frame concisely but with important details."
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(image_data).decode('utf-8')
                            }
                        }
                    ]
                }
            ]
        )
        
        # Extract the description
        description = response.content[0].text
        
        return description
    
    except Exception as e:
        logger.error(f"Claude Vision frame analysis error: {str(e)}")
        return "Could not analyze this frame"

def generate_video_summary(metadata, frame_descriptions):
    """
    Generate an overall summary of the video based on metadata and frame descriptions.
    
    Args:
        metadata (str): Video metadata
        frame_descriptions (list): Descriptions of key frames
    
    Returns:
        str: Overall video summary
    """
    try:
        # Combine frame descriptions
        frames_text = "\n".join(frame_descriptions)
        
        # Generate summary using Claude
        prompt = f"""
        Based on the following video metadata and key frame descriptions, provide a concise summary of what this video is about:
        
        VIDEO METADATA:
        {metadata}
        
        KEY FRAME DESCRIPTIONS:
        {frames_text}
        
        Please provide a 2-3 paragraph summary of what appears to be happening in this video.
        """
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022", # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        summary = response.content[0].text
        return summary
    
    except Exception as e:
        logger.error(f"Error generating video summary: {str(e)}")
        return "Could not generate a summary for this video."
