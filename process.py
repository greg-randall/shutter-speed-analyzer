# Organize imports by standard library, third-party, and local
import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime

# Third-party imports
import colorama
from colorama import Fore, Style
import cv2
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require a display server
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def calculate_frame_times(frame_idx, container_fps, slowmo_factor=1.0):
    """
    Calculate consistent frame times in both video time and real-world time.
    
    Args:
        frame_idx: Frame index (zero-based)
        container_fps: Container frame rate
        slowmo_factor: Slow motion factor (real_fps / container_fps)
        
    Returns:
        Tuple of (video_time_ms, real_time_ms)
    """
    # Time in the video timeline (as played back)
    ms_per_frame = 1000.0 / container_fps
    video_time_ms = frame_idx * ms_per_frame
    
    # Real-world time (adjusted for slow motion if detected)
    real_time_ms = video_time_ms / slowmo_factor if slowmo_factor > 1.0 else video_time_ms
    
    return video_time_ms, real_time_ms

def create_frame_brightness_map(brightness_values, start_frame):
    """
    Create a mapping of frame indices to their brightness values.
    
    Args:
        brightness_values: List of brightness values from analysis
        start_frame: Starting frame index of the analysis
        
    Returns:
        Dictionary mapping frame indices to brightness values
    """
    return {start_frame + i: value for i, value in enumerate(brightness_values)}

def reliable_frame_seek(cap, target_frame):
    """
    Reliably seek to a specific frame in a video.
    If direct seeking fails, falls back to sequential reading.
    
    Args:
        cap: OpenCV VideoCapture object
        target_frame: Frame number to seek to
        
    Returns:
        True if successful, False otherwise
    """
    # Try direct seeking first
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    # Verify we're at the expected position
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # If we're at the right position, we're done
    if current_pos == target_frame:
        return True
    
    print(f"{Fore.YELLOW}Warning: Direct frame seeking failed. Requested frame {target_frame}, "
          f"got {current_pos}. Falling back to sequential reading.{Style.RESET_ALL}")
    
    # If direct seeking failed and we're past the target, reset to beginning
    if current_pos > target_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current_pos = 0
    
    # Read frames sequentially until we reach the target
    frames_to_skip = target_frame - current_pos
    for _ in range(frames_to_skip):
        ret = cap.read()[0]
        if not ret:
            print(f"{Fore.RED}Error: Failed to reach target frame {target_frame} "
                  f"via sequential reading.{Style.RESET_ALL}")
            return False
    
    return True

def get_video_metadata(video_path):
    """
    Extract metadata from video file using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video metadata or None if extraction failed
    """
    try:
        # Try using ffprobe to get detailed metadata
        cmd = [
            'ffprobe', 
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', 
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)
        
        # Extract basic information
        info = {
            'container_fps': None,
            'real_fps': None,
            'width': None,
            'height': None,
            'duration': None,
        }
        
        # Get container FPS and other video stream information
        extract_stream_metadata(metadata, info)
        
        # Get duration from format section
        if 'format' in metadata and 'duration' in metadata['format']:
            info['duration'] = float(metadata['format']['duration'])
        
        # Check for Android capture FPS in format tags
        extract_android_fps(metadata, info)
        
        # If we couldn't find real_fps in metadata, check filename
        if info['real_fps'] is None:
            extract_fps_from_filename(video_path, info)
        
        return info
    
    except Exception as e:
        print(f"Warning: Could not extract metadata using ffprobe: {e}")
        return None

def create_visualization_plots(output_dir, timestamps_array, brightness_array, 
                              white_percentage_threshold, shutter_intervals):
    """
    Create visualization plots of the analysis results.
    
    Args:
        output_dir: Directory to save the plots
        timestamps_array: Array of frame timestamps
        brightness_array: Array of brightness values
        white_percentage_threshold: Threshold for considering a frame part of a shutter event
        shutter_intervals: List of detected shutter events
        
    Returns:
        Path to the generated plot
    """
    if len(brightness_array) == 0:
        return None
        
    plt.figure(figsize=(12, 8))
    
    # Plot brightness over time
    plt.subplot(2, 1, 1)
    plt.plot(timestamps_array, brightness_array)
    plt.axhline(y=white_percentage_threshold, color='r', linestyle='--', 
                label=f'Threshold ({white_percentage_threshold}%)')
    plt.title('White Pixel Percentage in ROI over Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('White Pixel Percentage (%)')
    plt.grid(True)
    plt.legend()
    
    # Highlight shutter events
    for event in shutter_intervals:
        plt.axvspan(
            event['start_time_ms'], 
            event['end_time_ms'], 
            alpha=0.3, 
            color='green'
        )
    
    # Plot shutter event durations
    if len(shutter_intervals) > 0:
        plt.subplot(2, 1, 2)
        event_numbers = list(range(1, len(shutter_intervals) + 1))
        durations = [event['duration_ms'] for event in shutter_intervals]
        
        avg_duration = np.mean(durations)
        
        plt.bar(event_numbers, durations)
        plt.axhline(y=avg_duration, color='r', linestyle='--', 
                    label=f'Average ({avg_duration:.2f}ms)')
        plt.title('Shutter Event Durations')
        plt.xlabel('Event Number')
        plt.ylabel('Duration (ms)')
        plt.grid(True, axis='y')
        plt.legend()
    
    # Save the figure
    plt_path = os.path.join(output_dir, "shutter_analysis_plot.png")
    plt.tight_layout()
    plt.savefig(plt_path)
    
    return plt_path




def print_video_info(video_path, container_fps, ms_per_frame, real_fps, real_ms_per_frame, 
                    frame_width, frame_height, roi, threshold, white_percentage_threshold,
                    start_time_seconds, end_time_seconds, max_duration_seconds):
    """
    Print formatted video and analysis information.
    
    Args:
        video_path: Path to the video file
        container_fps: Container frame rate
        ms_per_frame: Milliseconds per frame
        real_fps: Real capture FPS for slow motion videos
        real_ms_per_frame: Real-world milliseconds per frame
        frame_width: Video frame width
        frame_height: Video frame height
        roi: Region of interest as (x1, y1, x2, y2)
        threshold: Brightness threshold
        white_percentage_threshold: Threshold for considering a frame part of a shutter event
        start_time_seconds: Start time of analysis in seconds
        end_time_seconds: End time of analysis in seconds
        max_duration_seconds: Maximum duration analyzed in seconds
    """
    x1, y1, x2, y2 = roi
    
    print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Video Analysis Configuration{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Video file: {Style.BRIGHT}{os.path.basename(video_path)}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Container FPS: {Style.BRIGHT}{container_fps:.2f} {Style.RESET_ALL}(each frame is {ms_per_frame:.2f}ms)")
    if real_fps:
        print(f"{Fore.WHITE}Real capture FPS: {Style.BRIGHT}{real_fps:.2f} {Style.RESET_ALL}(each frame represents {real_ms_per_frame:.2f}ms in real time)")
    print(f"{Fore.WHITE}Video resolution: {Style.BRIGHT}{frame_width}x{frame_height}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}ROI: {Style.BRIGHT}({x1}, {y1}) to ({x2}, {y2}){Style.RESET_ALL}")
    print(f"{Fore.WHITE}Brightness threshold: {Style.BRIGHT}{threshold}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}White percentage threshold: {Style.BRIGHT}{white_percentage_threshold}%{Style.RESET_ALL}")
    
    # Print analysis time range
    if start_time_seconds is not None:
        print(f"{Fore.WHITE}Start time: {Style.BRIGHT}{start_time_seconds} seconds{Style.RESET_ALL}")
    if end_time_seconds is not None:
        print(f"{Fore.WHITE}End time: {Style.BRIGHT}{end_time_seconds} seconds{Style.RESET_ALL}")
    if max_duration_seconds is not None:
        print(f"{Fore.WHITE}Maximum duration: {Style.BRIGHT}{max_duration_seconds} seconds{Style.RESET_ALL}")
    if start_time_seconds is None and end_time_seconds is None and max_duration_seconds is None:
        print(f"{Fore.WHITE}Analyzing: {Style.BRIGHT}Entire video{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}\n")


def create_event_visualization(frame, roi, white_percentage, is_event_frame):
    """
    Create a visualization frame for an event with appropriate markings.
    
    Args:
        frame: Original video frame
        roi: Region of interest as (x1, y1, x2, y2)
        white_percentage: Calculated white pixel percentage
        is_event_frame: Whether this frame is part of a shutter event
        
    Returns:
        Marked frame with visualizations
    """
    x1, y1, x2, y2 = roi
    
    # Draw the ROI rectangle on the frame
    marked_frame = frame.copy()
    cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add white percentage text with three decimal places
    cv2.putText(
        marked_frame, 
        f"White %: {white_percentage:.3f}%", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )

    if is_event_frame:
        # Add brightness info to the event frame label
        cv2.putText(
            marked_frame, 
            f"SHUTTER EVENT (BRIGHTNESS: {white_percentage:.3f}%)", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 0, 255), 
            2
        )
        # Add a red rectangle around the edge of the frame
        h, w = marked_frame.shape[:2]
        cv2.rectangle(marked_frame, (0, 0), (w-1, h-1), (0, 0, 255), 3)
    
    return marked_frame


def process_shutter_events(video_path, shutter_intervals, roi, threshold, 
                          white_percentage_threshold, frame_brightness_map, 
                          container_fps, slowmo_factor, total_frames, output_dir):
    """
    Process and save frames for each detected shutter event.
    
    Args:
        video_path: Path to the video file
        shutter_intervals: List of detected shutter events
        roi: Region of interest as (x1, y1, x2, y2)
        threshold: Brightness threshold for binary conversion
        white_percentage_threshold: Threshold for considering a frame part of a shutter event
        frame_brightness_map: Dictionary mapping frame indices to brightness values
        container_fps: Container frame rate
        slowmo_factor: Slow motion factor
        total_frames: Total frames in the video
        output_dir: Directory to save the output
    """
    # We need to store all frames to include context frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{Fore.RED}Error: Could not reopen video file {video_path}{Style.RESET_ALL}")
        return
    
    # Process each event
    for i, event in enumerate(shutter_intervals):
        # Create folder for this event
        event_dir = os.path.join(output_dir, f"shutter-event-{i+1:03d}")
        os.makedirs(event_dir, exist_ok=True)
        
        # Calculate frame range with context (5 frames before and 10 frames after)
        start_frame_with_context = max(event['start_frame'] - 5, 0)
        end_frame_with_context = min(event['end_frame'] + 10, total_frames - 1)
        
        # Set position to start frame with context
        if not reliable_frame_seek(cap, start_frame_with_context):
            print(f"{Fore.RED}Error: Could not seek to frame {start_frame_with_context} for event {i+1}.{Style.RESET_ALL}")
            continue  # Skip this event but continue with others
        
        # Process frames for this event with a mini progress bar
        frames_to_process = end_frame_with_context - start_frame_with_context + 1
        with tqdm(total=frames_to_process, desc=f"Event {i+1}", unit="frame", ncols=80, 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as event_pbar:
            for frame_idx in range(start_frame_with_context, end_frame_with_context + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate frame timestamps consistently
                video_time_ms, real_time_ms = calculate_frame_times(frame_idx, container_fps, slowmo_factor)
                
                # Process the ROI and calculate brightness
                if frame.shape[0] >= roi[3] and frame.shape[1] >= roi[2]:
                    white_percentage, thresholded, roi_frame = process_roi_frame(frame, roi, threshold)
                    
                    # Verify brightness against stored value from first pass
                    if frame_idx in frame_brightness_map:
                        stored_brightness = frame_brightness_map[frame_idx]
                        if abs(white_percentage - stored_brightness) > 1.0:  # Allow small differences
                            # If this is a critical frame (event frame), try to recover the brightness
                            expected_event_frame = event['start_frame'] <= frame_idx <= event['end_frame']
                            if expected_event_frame and white_percentage < 0.5:
                                print(f"{Fore.RED}Recovering brightness for event frame {frame_idx}. "
                                      f"Using first pass value: {stored_brightness:.3f}% instead of {white_percentage:.3f}%{Style.RESET_ALL}")
                                # Use the stored brightness from first pass for this frame
                                white_percentage = stored_brightness
                            else:
                                print(f"{Fore.YELLOW}Warning: Frame {frame_idx} brightness mismatch. "
                                      f"First pass: {stored_brightness:.3f}%, Second pass: {white_percentage:.3f}%{Style.RESET_ALL}")
                    
                    # Determine if this is an event frame based on the first pass analysis
                    is_event_frame = event['start_frame'] <= frame_idx <= event['end_frame']
                    actual_event_frame = white_percentage > white_percentage_threshold
                    
                    # Log any discrepancies for debugging
                    if is_event_frame != actual_event_frame:
                        print(f"{Fore.YELLOW}Warning: Frame {frame_idx} index-based classification ({is_event_frame}) " 
                              f"doesn't match brightness-based classification ({actual_event_frame}). "
                              f"Brightness: {white_percentage:.3f}%{Style.RESET_ALL}")
                    
                    # Create visualization frame
                    marked_frame = create_event_visualization(frame, roi, white_percentage, is_event_frame)
                    
                    # Use a consistent naming pattern that sorts properly but still indicates event vs context
                    marker = "e" if is_event_frame else "c"
                    output_path = os.path.join(
                        event_dir, 
                        f"frame_{frame_idx:06d}_{marker}_{white_percentage:.1f}pct_{video_time_ms:.1f}ms.jpg"
                    )
                    cv2.imwrite(output_path, marked_frame)
                
                event_pbar.update(1)
        
        # Add event folder path to the event dictionary for reporting
        event['folder'] = event_dir
        
        # Print event summary
        if real_fps := (container_fps * slowmo_factor if slowmo_factor > 1.0 else None):
            shutter_speed_denominator = int(1000 / event['real_duration_ms'])
            print(f"{Fore.GREEN}Event {i+1}: {Style.BRIGHT}Frames {event['start_frame']} to {event['end_frame']} "
                  f"{Style.RESET_ALL}(white: {event['max_brightness']:.1f}%, duration: {event['duration_ms']:.1f}ms, "
                  f"real: {event['real_duration_ms']:.1f}ms, ~1/{shutter_speed_denominator}s)")
        else:
            shutter_speed_denominator = int(1000 / event['duration_ms'])
            print(f"{Fore.GREEN}Event {i+1}: {Style.BRIGHT}Frames {event['start_frame']} to {event['end_frame']} "
                  f"{Style.RESET_ALL}(white: {event['max_brightness']:.1f}%, duration: {event['duration_ms']:.1f}ms, "
                  f"~1/{shutter_speed_denominator}s)")
    
    cap.release()



def find_shutter_events(brightness_array, threshold, start_frame, container_fps, slowmo_factor=1.0):
    """
    Find and group shutter events based on brightness threshold.
    
    Args:
        brightness_array: Array of brightness values
        threshold: Threshold for considering a frame part of a shutter event
        start_frame: Starting frame index of the analysis
        container_fps: Container frame rate
        slowmo_factor: Slow motion factor (real_fps / container_fps)
        
    Returns:
        List of dictionaries containing shutter event information
    """
    # Find shutter events (frames above threshold)
    shutter_events = brightness_array > threshold
    shutter_frames = np.where(shutter_events)[0]
    
    # Group consecutive frames into single events
    shutter_intervals = []
    if len(shutter_frames) > 0:
        # Find gaps in consecutive frame sequences
        gaps = np.where(np.diff(shutter_frames) > 1)[0]
                
        # Split into individual events
        shutter_events_grouped = np.split(shutter_frames, gaps + 1)
                
        for event in shutter_events_grouped:
            if len(event) > 0:
                # Add start_frame to convert from relative to absolute frame indices
                start_frame_idx = event[0] + start_frame
                end_frame_idx = event[-1] + start_frame
                
                # Calculate times
                ms_per_frame = 1000.0 / container_fps
                start_time_ms, start_real_time_ms = calculate_frame_times(
                    start_frame_idx, container_fps, slowmo_factor)
                end_time_ms, end_real_time_ms = calculate_frame_times(
                    end_frame_idx, container_fps, slowmo_factor)
                
                # Calculate durations
                duration_ms = end_time_ms - start_time_ms + ms_per_frame  # Add one frame duration
                real_duration_ms = end_real_time_ms - start_real_time_ms + (ms_per_frame / slowmo_factor)
                        
                shutter_intervals.append({
                    "start_frame": start_frame_idx,
                    "end_frame": end_frame_idx,
                    "start_time_ms": start_time_ms,
                    "end_time_ms": end_time_ms,
                    "duration_ms": duration_ms,
                    "real_start_time_ms": start_real_time_ms,
                    "real_end_time_ms": end_real_time_ms,
                    "real_duration_ms": real_duration_ms,
                    "max_brightness": np.max(brightness_array[event[0]:event[-1]+1])
                })
    
    return shutter_intervals


def analyze_shutter(video_path, roi, threshold, max_duration_seconds=None, 
                   start_time_seconds=None, end_time_seconds=None, 
                   output_visualization=True, debug=False, metadata=None, 
                   white_percentage_threshold=10.0):
    """
    Analyze a video to detect camera shutter events.
    
    Args:
        video_path: Path to the video file
        roi: Region of interest as (x1, y1, x2, y2)
        threshold: Brightness threshold for binary conversion
        max_duration_seconds: Maximum duration to analyze in seconds
        start_time_seconds: Start time in seconds to begin analysis
        end_time_seconds: End time in seconds to stop analysis
        output_visualization: Whether to generate visualization plots
        debug: Whether to save debug frames
        metadata: Pre-extracted video metadata
        white_percentage_threshold: Threshold for considering a frame part of a shutter event
        
    Returns:
        Tuple of (output_directory, shutter_intervals)
    """
    # Initialize colorama
    colorama.init()
    
    # Create output folder with timestamp
    timestamp = int(time.time())
    output_dir = f"shutter_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create debug subfolder if debug mode is enabled
    debug_dir = None
    if debug:
        debug_dir = os.path.join(output_dir, "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"{Fore.CYAN}Debug mode enabled: {Style.BRIGHT}Saving thresholded frames to {debug_dir}{Style.RESET_ALL}")
    
    # Parse region of interest
    x1, y1, x2, y2 = roi
    
    # Get video metadata if not provided
    if metadata is None:
        metadata = get_video_metadata(video_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{Fore.RED}Error: Could not open video file {video_path}{Style.RESET_ALL}")
        return None, []
    
    # Get video properties
    container_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine the actual FPS to use for calculations
    fps = container_fps
    real_fps = None
    slowmo_factor = 1.0
    
    if metadata:
        print(f"{Fore.CYAN}Video metadata: {metadata}{Style.RESET_ALL}")
        if metadata.get('real_fps') is not None:
            real_fps = metadata['real_fps']
            slowmo_factor = real_fps / container_fps
            print(f"{Fore.GREEN}Detected slow motion video: {Style.BRIGHT}{real_fps}fps captured, "
                  f"played at {container_fps}fps{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Slow motion factor: {Style.BRIGHT}{slowmo_factor:.2f}x{Style.RESET_ALL}")
    
    # Time per frame in milliseconds (using container FPS for frame timing)
    ms_per_frame = 1000.0 / container_fps
    
    # Real-world time per frame (adjusted for slow motion if detected)
    real_ms_per_frame = ms_per_frame / slowmo_factor if slowmo_factor > 1.0 else ms_per_frame
    
    # Print video information
    print_video_info(
        video_path, container_fps, ms_per_frame, real_fps, real_ms_per_frame,
        frame_width, frame_height, roi, threshold, white_percentage_threshold,
        start_time_seconds, end_time_seconds, max_duration_seconds
    )
    
    # Calculate frame numbers for start and end times
    start_frame = 0
    if start_time_seconds is not None:
        start_frame = int(start_time_seconds * fps)
        # Seek to the start position
        if not reliable_frame_seek(cap, start_frame):
            print(f"{Fore.RED}Error: Could not seek to start frame {start_frame}.{Style.RESET_ALL}")
            return output_dir, []
    
    end_frame = total_frames
    if end_time_seconds is not None:
        end_frame = int(end_time_seconds * fps)
        if end_frame > total_frames:
            end_frame = total_frames
    
    # Calculate total frames to process
    total_frames_to_process = end_frame - start_frame
    print(f"{Fore.CYAN}Processing {Style.BRIGHT}{total_frames_to_process} frames{Style.RESET_ALL} "
          f"(from frame {start_frame} to {end_frame})")
    
    # First pass: analyze all frames to detect events
    frame_count = start_frame
    brightness_values = []
    frame_timestamps = []
    
    # Process the video frame by frame with tqdm progress bar
    with tqdm(total=total_frames_to_process, desc="Analyzing frames", unit="frame", ncols=100, 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate frame timestamps consistently
            video_time_ms, real_time_ms = calculate_frame_times(frame_count, container_fps, slowmo_factor)
            frame_timestamps.append(video_time_ms)
            
            # Process the ROI and calculate brightness
            if y2 <= frame.shape[0] and x2 <= frame.shape[1]:
                white_percentage, thresholded, roi_frame = process_roi_frame(frame, roi, threshold)
                brightness_values.append(white_percentage)
                
                # Save debug frames if debug mode is enabled
                if debug:
                    debug_frame = create_debug_frame(frame, roi, white_percentage, thresholded)
                    debug_path = os.path.join(debug_dir, f"frame_{frame_count:06d}_{video_time_ms:.1f}ms.jpg")
                    cv2.imwrite(debug_path, debug_frame)
            else:
                print(f"{Fore.RED}Warning: ROI coordinates ({x1}, {y1}, {x2}, {y2}) "
                      f"out of frame bounds ({frame_width}x{frame_height}){Style.RESET_ALL}")
                brightness_values.append(0)
            
            frame_count += 1
            pbar.update(1)
            
            # Check if we've reached the maximum duration to analyze
            if max_duration_seconds is not None and video_time_ms >= max_duration_seconds * 1000:
                print(f"\n{Fore.CYAN}Reached maximum analysis duration of {max_duration_seconds} seconds{Style.RESET_ALL}")
                break
                
            # Check if we've reached the end time
            if end_time_seconds is not None and video_time_ms >= end_time_seconds * 1000:
                print(f"\n{Fore.CYAN}Reached end time of {end_time_seconds} seconds{Style.RESET_ALL}")
                break
    
    cap.release()
    
    # Convert to numpy arrays for analysis
    brightness_array = np.array(brightness_values)
    timestamps_array = np.array(frame_timestamps)
    
    # Create a mapping of frame indices to brightness values for verification
    frame_brightness_map = create_frame_brightness_map(brightness_values, start_frame)
    
    # Find and group shutter events
    shutter_intervals = find_shutter_events(
        brightness_array, white_percentage_threshold, start_frame, container_fps, slowmo_factor)
    
    # Print summary of detected events
    print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Analysis Results{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    
    if len(shutter_intervals) > 0:
        print(f"{Fore.GREEN}Detected {Style.BRIGHT}{len(shutter_intervals)} shutter events{Style.RESET_ALL}\n")
    else:
        print(f"{Fore.RED}No shutter events detected. Try adjusting the threshold or ROI.{Style.RESET_ALL}\n")
    
    # Second pass: save event frames with context
    if len(shutter_intervals) > 0:
        process_shutter_events(
            video_path, shutter_intervals, roi, threshold, white_percentage_threshold,
            frame_brightness_map, container_fps, slowmo_factor, total_frames, output_dir
        )
    
    # Generate report
    report_path = generate_report(
        output_dir, video_path, fps, ms_per_frame, frame_count, start_frame,
        roi, threshold, white_percentage_threshold, shutter_intervals,
        start_time_seconds, end_time_seconds, max_duration_seconds,
        slowmo_factor, real_fps
    )
    
    print(f"\n{Fore.CYAN}Analysis report saved to: {Style.BRIGHT}{report_path}{Style.RESET_ALL}")
    
    # Create visualization plots
    if output_visualization and len(brightness_values) > 0:
        plt_path = create_visualization_plots(
            output_dir, timestamps_array, brightness_array, 
            white_percentage_threshold, shutter_intervals
        )
        if plt_path:
            print(f"{Fore.CYAN}Analysis plot saved to: {Style.BRIGHT}{plt_path}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}All analysis results saved to: {Style.BRIGHT}{output_dir}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    
    return output_dir, shutter_intervals

def process_roi_frame(frame, roi, threshold):
    """
    Process a region of interest in a frame and calculate white pixel percentage.
    
    Args:
        frame: Input video frame
        roi: Region of interest as (x1, y1, x2, y2)
        threshold: Brightness threshold for binary conversion
        
    Returns:
        Tuple of (white_percentage, thresholded_image, roi_frame)
    """
    x1, y1, x2, y2 = roi
    
    # Extract region of interest
    roi_frame = frame[y1:y2, x1:x2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create binary image
    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of white pixels (brightness > threshold)
    white_pixel_count = np.count_nonzero(thresholded)
    total_pixels = thresholded.size
    white_percentage = (white_pixel_count / total_pixels) * 100
    
    return white_percentage, thresholded, roi_frame


def generate_report(output_dir, video_path, fps, ms_per_frame, frame_count, start_frame, 
                   roi, threshold, white_percentage_threshold, shutter_intervals,
                   start_time_seconds, end_time_seconds, max_duration_seconds, 
                   slowmo_factor=1.0, real_fps=None):
    """
    Generate a text report of the analysis results.
    
    Args:
        output_dir: Directory to save the report
        video_path: Path to the analyzed video
        fps: Video frame rate
        ms_per_frame: Milliseconds per frame
        frame_count: Total frames processed
        start_frame: Starting frame of analysis
        roi: Region of interest as (x1, y1, x2, y2)
        threshold: Brightness threshold
        white_percentage_threshold: Threshold for considering a frame part of a shutter event
        shutter_intervals: List of detected shutter events
        start_time_seconds: Start time of analysis in seconds
        end_time_seconds: End time of analysis in seconds
        max_duration_seconds: Maximum duration analyzed in seconds
        slowmo_factor: Slow motion factor
        real_fps: Real capture FPS for slow motion videos
        
    Returns:
        Path to the generated report
    """
    x1, y1, x2, y2 = roi
    
    report_path = os.path.join(output_dir, "shutter_analysis_report.txt")
    with open(report_path, "w") as report_file:
        report_file.write(f"Shutter Speed Analysis Report\n")
        report_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report_file.write(f"Video: {video_path}\n")
        report_file.write(f"FPS: {fps} (each frame is {ms_per_frame:.2f}ms)\n")
        report_file.write(f"Total frames analyzed: {frame_count - start_frame}\n")
        report_file.write(f"ROI: ({x1}, {y1}) to ({x2}, {y2})\n")
        report_file.write(f"Brightness threshold: {threshold} (pixels above this value are considered 'white')\n")
        report_file.write(f"White percentage threshold: {white_percentage_threshold}%\n")
        if start_time_seconds is not None:
            report_file.write(f"Analysis start time: {start_time_seconds} seconds\n")
        if end_time_seconds is not None:
            report_file.write(f"Analysis end time: {end_time_seconds} seconds\n")
        if max_duration_seconds is not None:
            report_file.write(f"Maximum duration analyzed: {max_duration_seconds} seconds\n")
        report_file.write("\n")
        
        if len(shutter_intervals) > 0:
            report_file.write(f"Detected {len(shutter_intervals)} shutter events:\n\n")
            
            for i, event in enumerate(shutter_intervals):
                report_file.write(f"Event {i+1}:\n")
                report_file.write(f"  Frames: {event['start_frame']} to {event['end_frame']}\n")
                report_file.write(f"  Video time: {event['start_time_ms']:.2f}ms to {event['end_time_ms']:.2f}ms\n")
                report_file.write(f"  Video duration: {event['duration_ms']:.2f}ms\n")
                report_file.write(f"  Max brightness: {event['max_brightness']:.1f}\n")
                if 'folder' in event:
                    report_file.write(f"  Event folder: {os.path.basename(event['folder'])}\n")
                
                # Use pre-calculated real-world duration
                if real_fps:
                    report_file.write(f"  Real-world duration: {event['real_duration_ms']:.2f}ms\n")
                    # Convert to traditional shutter speed notation (1/x sec)
                    shutter_speed_denominator = int(1000 / event['real_duration_ms'])
                    report_file.write(f"  Approximate shutter speed: 1/{shutter_speed_denominator} sec\n\n")
                else:
                    # Convert to traditional shutter speed notation (1/x sec)
                    shutter_speed_denominator = int(1000 / event['duration_ms'])
                    report_file.write(f"  Approximate shutter speed: 1/{shutter_speed_denominator} sec\n\n")
            
            # Calculate average shutter duration
            durations = [event['duration_ms'] for event in shutter_intervals]
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            
            report_file.write(f"Average shutter duration (video time): {avg_duration:.2f}ms ± {std_duration:.2f}ms\n")
            
            if real_fps:
                # Calculate real-world durations
                real_durations = [d / slowmo_factor for d in durations]
                real_avg_duration = np.mean(real_durations)
                real_std_duration = np.std(real_durations)
                
                report_file.write(f"Average shutter duration (real-world): {real_avg_duration:.2f}ms ± {real_std_duration:.2f}ms\n")
                report_file.write(f"Approximate average shutter speed: 1/{int(1000/real_avg_duration)} sec\n")
            else:
                report_file.write(f"Approximate average shutter speed: 1/{int(1000/avg_duration)} sec\n")
        else:
            report_file.write("No shutter events detected. Try adjusting the threshold or ROI.\n")
    
    return report_path

def main():
    """Main function to parse arguments and run the analysis."""
    # Print a nice banner
    colorama.init()
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}Camera Shutter Speed Analysis Tool{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    parser = argparse.ArgumentParser(description='Analyze video to measure camera shutter speed')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--roi', nargs=4, type=int, required=True, 
                        help='Region of interest as x1 y1 x2 y2 (top-left and bottom-right coordinates)')
    parser.add_argument('--threshold', type=int, default=100, 
                        help='Brightness threshold (0-255) for converting to binary image')
    parser.add_argument('--white-percentage', type=float, default=0.1,
                        help='Percentage of white pixels in ROI to consider as a shutter event (default: 0.1)')
    parser.add_argument('--start-time', type=float, 
                        help='Start time in seconds to begin analysis')
    parser.add_argument('--end-time', type=float, 
                        help='End time in seconds to stop analysis')
    parser.add_argument('--max-duration', type=float, 
                        help='Maximum duration to analyze in seconds (deprecated, use --start-time and --end-time instead)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating visualization plots')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to save thresholded frames')
    parser.add_argument('--real-fps', type=float,
                        help='Specify the real capture FPS for slow motion videos')
    
    args = parser.parse_args()
    
    # If real FPS is specified, create a minimal metadata structure
    metadata = None
    if args.real_fps:
        metadata = {'real_fps': args.real_fps}
    
    # Handle deprecated max_duration parameter
    start_time = args.start_time
    end_time = args.end_time
    
    if args.max_duration is not None:
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = start_time + args.max_duration
        print(f"{Fore.YELLOW}Warning: --max-duration is deprecated. Please use --start-time and --end-time instead.{Style.RESET_ALL}")
    
    analyze_shutter(
        args.video_path, 
        args.roi, 
        args.threshold,
        max_duration_seconds=args.max_duration,
        start_time_seconds=start_time,
        end_time_seconds=end_time,
        output_visualization=not args.no_plot,
        debug=args.debug,
        metadata=metadata,
        white_percentage_threshold=args.white_percentage
    )

if __name__ == "__main__":
    main()
def extract_stream_metadata(metadata, info):
    """
    Extract metadata from video streams.
    
    Args:
        metadata: Raw metadata from ffprobe
        info: Dictionary to populate with extracted information
    """
    for stream in metadata.get('streams', []):
        if stream.get('codec_type') == 'video':
            # Get resolution
            info['width'] = stream.get('width')
            info['height'] = stream.get('height')
            
            # Get FPS from container
            if 'r_frame_rate' in stream:
                fps_str = stream['r_frame_rate']
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    info['container_fps'] = num / den if den != 0 else None
            
            # Look for metadata tags that might indicate slow motion
            tags = stream.get('tags', {})
            for key, value in tags.items():
                # Look for common slow motion indicators in metadata
                if 'slow' in key.lower() or 'fps' in key.lower() or 'frame_rate' in key.lower():
                    # Try to extract a number from the value
                    fps_match = re.search(r'(\d+)(?:\.\d+)?(?:\s*fps)?', str(value))
                    if fps_match:
                        potential_fps = float(fps_match.group(1))
                        # If it's significantly higher than container FPS, it might be the real capture rate
                        if info['container_fps'] is None or potential_fps > info['container_fps'] * 1.5:
                            info['real_fps'] = potential_fps


def extract_android_fps(metadata, info):
    """
    Extract Android-specific FPS information from metadata.
    
    Args:
        metadata: Raw metadata from ffprobe
        info: Dictionary to populate with extracted information
    """
    if 'format' in metadata and 'tags' in metadata['format']:
        format_tags = metadata['format']['tags']
        if 'com.android.capture.fps' in format_tags:
            try:
                android_fps = float(format_tags['com.android.capture.fps'])
                # If this is significantly higher than container FPS, use it as real_fps
                if info['container_fps'] is None or android_fps > info['container_fps'] * 1.5:
                    info['real_fps'] = android_fps
                    print(f"Found Android capture FPS: {android_fps}")
            except (ValueError, TypeError):
                pass


def extract_fps_from_filename(video_path, info):
    """
    Try to extract FPS information from the video filename.
    
    Args:
        video_path: Path to the video file
        info: Dictionary to populate with extracted information
    """
    # Check filename for common slow motion indicators (like "240fps" or "240p")
    fps_in_filename = re.search(r'(\d+)(?:fps|FPS|p)', os.path.basename(video_path))
    if fps_in_filename:
        potential_fps = float(fps_in_filename.group(1))
        if potential_fps > 60:  # Assume it's slow motion if > 60fps
            info['real_fps'] = potential_fps



def create_debug_frame(frame, roi, white_percentage, thresholded):
    """
    Create a debug frame with visualization of the ROI and thresholding.
    
    Args:
        frame: Original video frame
        roi: Region of interest as (x1, y1, x2, y2)
        white_percentage: Calculated white pixel percentage
        thresholded: Thresholded binary image
        
    Returns:
        Debug frame with visualizations
    """
    x1, y1, x2, y2 = roi
    
    # Convert thresholded image back to BGR for visualization
    thresholded_color = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    
    # Create a full-size thresholded image (same size as original frame)
    full_thresholded = np.zeros_like(frame)
    # Place the thresholded ROI in the correct position
    full_thresholded[y1:y2, x1:x2] = thresholded_color
    
    # Draw the ROI rectangle on both images
    debug_frame = frame.copy()
    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(full_thresholded, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add white percentage text with three decimal places
    cv2.putText(
        debug_frame, 
        f"White %: {white_percentage:.3f}%", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    # Create a side-by-side comparison
    comparison = np.hstack((debug_frame, full_thresholded))
    
    return comparison



