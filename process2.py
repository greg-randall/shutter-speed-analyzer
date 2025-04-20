import cv2
import numpy as np
import os
import time
import argparse
import matplotlib
# Use Agg backend which doesn't require a display server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import json
import re

def get_video_metadata(video_path):
    """Extract metadata from video file using ffprobe"""
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
        
        # Get container FPS
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
        
        # Get duration
        if 'format' in metadata and 'duration' in metadata['format']:
            info['duration'] = float(metadata['format']['duration'])
        
        # If we couldn't find real_fps in metadata, check if filename contains indicators
        if info['real_fps'] is None:
            # Check filename for common slow motion indicators (like "240fps" or "240p")
            fps_in_filename = re.search(r'(\d+)(?:fps|FPS|p)', os.path.basename(video_path))
            if fps_in_filename:
                potential_fps = float(fps_in_filename.group(1))
                if potential_fps > 60:  # Assume it's slow motion if > 60fps
                    info['real_fps'] = potential_fps
        
        return info
    
    except Exception as e:
        print(f"Warning: Could not extract metadata using ffprobe: {e}")
        return None

def analyze_shutter(video_path, roi, threshold, max_duration_seconds=None, start_time_seconds=None, end_time_seconds=None, output_visualization=True, debug=False, metadata=None, white_percentage_threshold=10.0):
    # Create output folder with timestamp
    timestamp = int(time.time())
    output_dir = f"shutter_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create debug subfolder if debug mode is enabled
    debug_dir = None
    if debug:
        debug_dir = os.path.join(output_dir, "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled: Saving thresholded frames to {debug_dir}")
    
    # Parse region of interest
    x1, y1, x2, y2 = roi
    
    # Get video metadata if not provided
    if metadata is None:
        metadata = get_video_metadata(video_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
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
        print(f"Video metadata: {metadata}")
        if metadata.get('real_fps') is not None:
            real_fps = metadata['real_fps']
            slowmo_factor = real_fps / container_fps
            print(f"Detected slow motion video: {real_fps}fps captured, played at {container_fps}fps")
            print(f"Slow motion factor: {slowmo_factor:.2f}x")
    
    # Time per frame in milliseconds (using container FPS for frame timing)
    ms_per_frame = 1000.0 / container_fps
    
    # Real-world time per frame (adjusted for slow motion if detected)
    real_ms_per_frame = ms_per_frame / slowmo_factor if slowmo_factor > 1.0 else ms_per_frame
    
    print(f"Video container FPS: {container_fps} (each frame is {ms_per_frame:.2f}ms)")
    if real_fps:
        print(f"Real capture FPS: {real_fps} (each frame represents {real_ms_per_frame:.2f}ms in real time)")
    print(f"Video resolution: {frame_width}x{frame_height}")
    print(f"ROI: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"Brightness threshold: {threshold}")
    
    # Print analysis time range
    if start_time_seconds is not None:
        print(f"Will start analysis at {start_time_seconds} seconds")
    if end_time_seconds is not None:
        print(f"Will end analysis at {end_time_seconds} seconds")
    if max_duration_seconds is not None:
        print(f"Will analyze up to {max_duration_seconds} seconds of video")
        # Calculate the frame number where we'll stop
        max_frames = int(max_duration_seconds * fps)
        if max_frames < total_frames:
            total_frames = max_frames
    if start_time_seconds is None and end_time_seconds is None and max_duration_seconds is None:
        print("Will analyze the entire video")
    
    # Calculate frame numbers for start and end times
    start_frame = 0
    if start_time_seconds is not None:
        start_frame = int(start_time_seconds * fps)
        # Seek to the start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    end_frame = total_frames
    if end_time_seconds is not None:
        end_frame = int(end_time_seconds * fps)
        if end_frame > total_frames:
            end_frame = total_frames
    
    # Calculate total frames to process
    total_frames_to_process = end_frame - start_frame
    print(f"Will process {total_frames_to_process} frames (from frame {start_frame} to {end_frame})")
    
    frame_count = start_frame
    brightness_values = []
    frame_timestamps = []
    
    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate frame timestamp in milliseconds
        frame_time_ms = frame_count * ms_per_frame
        frame_timestamps.append(frame_time_ms)
        
        # Also track real-world time if slow motion is detected
        real_time_ms = frame_count * real_ms_per_frame
        
        # Extract region of interest
        if y2 <= frame.shape[0] and x2 <= frame.shape[1]:
            roi_frame = frame[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to create binary image
            _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Calculate percentage of white pixels (brightness > threshold)
            white_pixel_count = np.count_nonzero(thresholded)
            total_pixels = thresholded.size
            white_percentage = (white_pixel_count / total_pixels) * 100
            
            # Store the white percentage
            brightness_values.append(white_percentage)
            
            # Save debug frames if debug mode is enabled
            if debug:
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
                
                # Add white percentage text
                cv2.putText(
                    debug_frame, 
                    f"White %: {white_percentage:.1f}%", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # Create a side-by-side comparison
                comparison = np.hstack((debug_frame, full_thresholded))
                
                # Save the debug frame
                debug_path = os.path.join(debug_dir, f"frame_{frame_count:06d}_{frame_time_ms:.1f}ms.jpg")
                cv2.imwrite(debug_path, comparison)
            
            # If white percentage exceeds threshold, just track it
            # (frames will be saved in event-specific folders later)
            if white_percentage > white_percentage_threshold:
                # We don't save to root folder anymore to avoid duplication
                pass
        else:
            print(f"Warning: ROI coordinates ({x1}, {y1}, {x2}, {y2}) out of frame bounds ({frame_width}x{frame_height})")
            brightness_values.append(0)
        
        frame_count += 1
        
        # Display progress periodically
        if (frame_count - start_frame) % 100 == 0 or frame_count == end_frame:
            progress = ((frame_count - start_frame) / total_frames_to_process) * 100 if total_frames_to_process > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_count - start_frame}/{total_frames_to_process})")
            
        # Check if we've reached the maximum duration to analyze
        if max_duration_seconds is not None and frame_time_ms >= max_duration_seconds * 1000:
            print(f"Reached maximum analysis duration of {max_duration_seconds} seconds")
            break
            
        # Check if we've reached the end time
        if end_time_seconds is not None and frame_time_ms >= end_time_seconds * 1000:
            print(f"Reached end time of {end_time_seconds} seconds")
            break
    
    cap.release()
    
    # Convert to numpy arrays for analysis
    brightness_array = np.array(brightness_values)
    timestamps_array = np.array(frame_timestamps)
    
    # Find shutter events (frames above white percentage threshold)
    shutter_events = brightness_array > white_percentage_threshold
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
                start_frame = event[0]
                end_frame = event[-1]
                duration_ms = (end_frame - start_frame + 1) * ms_per_frame
                start_time_ms = start_frame * ms_per_frame
                end_time_ms = end_frame * ms_per_frame
                
                shutter_intervals.append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time_ms": start_time_ms,
                    "end_time_ms": end_time_ms,
                    "duration_ms": duration_ms,
                    "max_brightness": np.max(brightness_array[start_frame:end_frame+1])
                })
    
    # Create event-specific folders and save frames with context
    if len(shutter_intervals) > 0:
        # We need to store all frames to include context frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not reopen video file {video_path}")
            return output_dir, shutter_intervals
        
        # Process each event
        for i, event in enumerate(shutter_intervals):
            # Create folder for this event
            event_dir = os.path.join(output_dir, f"shutter-event-{i+1:03d}")
            os.makedirs(event_dir, exist_ok=True)
            
            # Calculate frame range with context (5 frames before and 10 frames after)
            start_frame_with_context = max(event['start_frame'] - 5, 0)
            end_frame_with_context = min(event['end_frame'] + 10, total_frames - 1)
            
            # Set position to start frame with context
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_with_context)
            
            # Process frames for this event
            for frame_idx in range(start_frame_with_context, end_frame_with_context + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate frame timestamp in milliseconds
                frame_time_ms = frame_idx * ms_per_frame
                
                # Extract region of interest
                if y2 <= frame.shape[0] and x2 <= frame.shape[1]:
                    roi_frame = frame[y1:y2, x1:x2]
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Apply thresholding to create binary image
                    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                    
                    # Calculate percentage of white pixels
                    white_pixel_count = np.count_nonzero(thresholded)
                    total_pixels = thresholded.size
                    white_percentage = (white_pixel_count / total_pixels) * 100
                    
                    # Draw the ROI rectangle on the frame
                    marked_frame = frame.copy()
                    cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                    # Add white percentage text
                    cv2.putText(
                        marked_frame, 
                        f"White %: {white_percentage:.1f}%", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                
                    # Add indicator if this is part of the actual event (not just context)
                    is_event_frame = event['start_frame'] <= frame_idx <= event['end_frame']
                    if is_event_frame:
                        cv2.putText(
                            marked_frame, 
                            "SHUTTER EVENT", 
                            (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 0, 255), 
                            2
                        )
                
                    # Save the frame with new naming convention
                    # Add indicator in filename if this is an event frame
                    prefix = "event" if is_event_frame else "context"
                    output_path = os.path.join(event_dir, f"{prefix}-{i+1:03d}_frame_{frame_idx:06d}_{frame_time_ms:.1f}ms.jpg")
                    cv2.imwrite(output_path, marked_frame)
            
            print(f"Event {i+1}: Frames {event['start_frame']} to {event['end_frame']} (white percentage: {event['max_brightness']:.1f}%)")
            print(f"Saved frames for event {i+1} to {event_dir}")
            
            # Add event folder path to the event dictionary for reporting
            event['folder'] = event_dir
        
        cap.release()
    
    # Generate report
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
                
                # If slow motion was detected, calculate real-world duration
                if real_fps:
                    real_duration_ms = event['duration_ms'] / slowmo_factor
                    report_file.write(f"  Real-world duration: {real_duration_ms:.2f}ms\n")
                    # Convert to traditional shutter speed notation (1/x sec)
                    shutter_speed_denominator = int(1000 / real_duration_ms)
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
    
    print(f"Analysis report saved to {report_path}")
    
    # Create visualization plots
    if output_visualization and len(brightness_values) > 0:
        plt.figure(figsize=(12, 8))
        
        # Plot brightness over time
        plt.subplot(2, 1, 1)
        plt.plot(timestamps_array, brightness_array)
        plt.axhline(y=white_percentage_threshold, color='r', linestyle='--', label=f'Threshold ({white_percentage_threshold}%)')
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
            
            plt.bar(event_numbers, durations)
            plt.axhline(y=avg_duration, color='r', linestyle='--', label=f'Average ({avg_duration:.2f}ms)')
            plt.title('Shutter Event Durations')
            plt.xlabel('Event Number')
            plt.ylabel('Duration (ms)')
            plt.grid(True, axis='y')
            plt.legend()
        
        # Save the figure
        plt_path = os.path.join(output_dir, "shutter_analysis_plot.png")
        plt.tight_layout()
        plt.savefig(plt_path)
        print(f"Analysis plot saved to {plt_path}")
    
    print(f"All analysis results saved to directory: {output_dir}")
    return output_dir, shutter_intervals

def main():
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
        print("Warning: --max-duration is deprecated. Please use --start-time and --end-time instead.")
    
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
