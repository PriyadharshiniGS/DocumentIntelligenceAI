import os
import logging
import tempfile
import subprocess
import base64
from PIL import Image
import io
import cv2
import anthropic
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(api_key=ANTHROPIC_API_KEY)

def process_video(file_path):
    """Process video file to extract key frames and analyze content."""
    try:
        # Extract basic video metadata
        metadata = extract_video_metadata(file_path)
        if not metadata:
            return ["Could not process video metadata"]

        # Extract key frames from the video (reduced number for efficiency)
        frame_paths = extract_key_frames(file_path, max_frames=3)
        if not frame_paths:
            return ["Could not extract frames from video"]

        # Analyze frames in sequence
        frame_descriptions = []
        for i, frame_path in enumerate(frame_paths):
            try:
                description = analyze_frame_with_claude(frame_path, timeout=30)
                if description:
                    frame_descriptions.append(f"Frame {i+1}: {description}")
            except Exception as e:
                logger.error(f"Error analyzing frame {i}: {str(e)}")
                continue

        # Compile results
        results = []
        results.append(f"Video metadata:\n{metadata}")

        if frame_descriptions:
            frame_analysis = "Key frames analysis:\n" + "\n\n".join(frame_descriptions)
            results.append(frame_analysis)

        # Generate a brief summary
        summary = f"Video contains {len(frame_descriptions)} analyzed frames."
        results.append(f"Summary:\n{summary}")

        return results

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return ["Error processing video file. Please try a shorter video."]

def extract_video_metadata(file_path):
    """Extract basic metadata from video using OpenCV."""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate duration
        duration_seconds = frame_count / fps if fps > 0 else 0
        minutes, seconds = divmod(duration_seconds, 60)

        metadata = (
            f"Duration: {int(minutes)} minutes {int(seconds)} seconds\n"
            f"Resolution: {width}x{height} pixels\n"
            f"Frame rate: {fps:.2f} FPS"
        )

        cap.release()
        return metadata
    except Exception as e:
        logger.error(f"Error extracting video metadata: {str(e)}")
        return None

def extract_key_frames(file_path, max_frames=3):
    """Extract key frames from the video."""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return []

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame intervals
        if frame_count <= max_frames:
            frame_indices = list(range(frame_count))
        else:
            # Take frames from start, middle, and end
            frame_indices = [
                0,
                frame_count // 2,
                frame_count - 1
            ]

        # Create temporary directory for frames
        frame_dir = tempfile.mkdtemp()
        frame_paths = []

        # Extract frames
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(frame_dir, f"frame_{i}.jpg")
                # Resize frame to reduce processing time
                resized_frame = cv2.resize(frame, (800, 450))
                cv2.imwrite(frame_path, resized_frame)
                frame_paths.append(frame_path)

        cap.release()
        return frame_paths
    except Exception as e:
        logger.error(f"Error extracting key frames: {str(e)}")
        return []

def analyze_frame_with_claude(frame_path, timeout=30):
    """Analyze a video frame using Claude Vision."""
    try:
        with open(frame_path, "rb") as img_file:
            image_data = img_file.read()

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Briefly describe what you see in this video frame in 1-2 sentences."
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

        return response.content[0].text
    except Exception as e:
        logger.error(f"Claude Vision frame analysis error: {str(e)}")
        return None

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
            model="claude-3-5-sonnet-20241022", 
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