import cv2
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_shutter(video_path, roi, threshold, max_duration_seconds=None, output_visualization=True, debug=False):
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
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Time per frame in milliseconds
    ms_per_frame = 1000.0 / fps
    
    print(f"Video FPS: {fps} (each frame is {ms_per_frame:.2f}ms)")
    print(f"Video resolution: {frame_width}x{frame_height}")
    print(f"ROI: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"Brightness threshold: {threshold}")
    
    if max_duration_seconds is not None:
        print(f"Will analyze up to {max_duration_seconds} seconds of video")
        # Calculate the frame number where we'll stop
        max_frames = int(max_duration_seconds * fps)
        if max_frames < total_frames:
            total_frames = max_frames
    else:
        print("Will analyze the entire video")
    
    frame_count = 0
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
        
        # Extract region of interest
        if y2 <= frame.shape[0] and x2 <= frame.shape[1]:
            roi_frame = frame[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate the average brightness in the ROI
            avg_brightness = np.mean(gray)
            brightness_values.append(avg_brightness)
            
            # Save debug frames if debug mode is enabled
            if debug:
                # Create a thresholded binary image
                _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                
                # Convert back to BGR for visualization
                thresholded_color = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
                
                # Draw the ROI rectangle on the original frame
                debug_frame = frame.copy()
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add brightness text
                cv2.putText(
                    debug_frame, 
                    f"Brightness: {avg_brightness:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # Create a side-by-side comparison
                comparison = np.hstack((debug_frame, thresholded_color))
                
                # Save the debug frame
                debug_path = os.path.join(debug_dir, f"frame_{frame_count:06d}_{frame_time_ms:.1f}ms.jpg")
                cv2.imwrite(debug_path, comparison)
            
            # If brightness exceeds threshold, save the frame
            if avg_brightness > threshold:
                # Draw the ROI rectangle on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add brightness text
                cv2.putText(
                    frame, 
                    f"Brightness: {avg_brightness:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # Save the frame
                output_path = os.path.join(output_dir, f"frame_{frame_count:06d}_{frame_time_ms:.1f}ms.jpg")
                cv2.imwrite(output_path, frame)
        else:
            print(f"Warning: ROI coordinates ({x1}, {y1}, {x2}, {y2}) out of frame bounds ({frame_width}x{frame_height})")
            brightness_values.append(0)
        
        frame_count += 1
        
        # Display progress periodically
        if frame_count % 100 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
        # Check if we've reached the maximum duration to analyze
        if max_duration_seconds is not None and frame_time_ms >= max_duration_seconds * 1000:
            print(f"Reached maximum analysis duration of {max_duration_seconds} seconds")
            break
    
    cap.release()
    
    # Convert to numpy arrays for analysis
    brightness_array = np.array(brightness_values)
    timestamps_array = np.array(frame_timestamps)
    
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
    
    # Generate report
    report_path = os.path.join(output_dir, "shutter_analysis_report.txt")
    with open(report_path, "w") as report_file:
        report_file.write(f"Shutter Speed Analysis Report\n")
        report_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report_file.write(f"Video: {video_path}\n")
        report_file.write(f"FPS: {fps} (each frame is {ms_per_frame:.2f}ms)\n")
        report_file.write(f"Total frames analyzed: {frame_count}\n")
        report_file.write(f"ROI: ({x1}, {y1}) to ({x2}, {y2})\n")
        report_file.write(f"Brightness threshold: {threshold}\n")
        if max_duration_seconds is not None:
            report_file.write(f"Maximum duration analyzed: {max_duration_seconds} seconds\n")
        report_file.write("\n")
        
        if len(shutter_intervals) > 0:
            report_file.write(f"Detected {len(shutter_intervals)} shutter events:\n\n")
            
            for i, event in enumerate(shutter_intervals):
                report_file.write(f"Event {i+1}:\n")
                report_file.write(f"  Frames: {event['start_frame']} to {event['end_frame']}\n")
                report_file.write(f"  Time: {event['start_time_ms']:.2f}ms to {event['end_time_ms']:.2f}ms\n")
                report_file.write(f"  Duration: {event['duration_ms']:.2f}ms\n")
                report_file.write(f"  Max brightness: {event['max_brightness']:.1f}\n")
                
                # Convert to traditional shutter speed notation (1/x sec)
                shutter_speed_denominator = int(1000 / event['duration_ms'])
                report_file.write(f"  Approximate shutter speed: 1/{shutter_speed_denominator} sec\n\n")
            
            # Calculate average shutter duration
            durations = [event['duration_ms'] for event in shutter_intervals]
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            
            report_file.write(f"Average shutter duration: {avg_duration:.2f}ms ± {std_duration:.2f}ms\n")
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
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        plt.title('Brightness in ROI over Time')
        plt.xlabel('Time (ms)')
        plt.ylabel('Average Brightness')
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
                        help='Brightness threshold (0-255, default: 100)')
    parser.add_argument('--max-duration', type=float, 
                        help='Maximum duration to analyze in seconds')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating visualization plots')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to save thresholded frames')
    
    args = parser.parse_args()
    
    analyze_shutter(
        args.video_path, 
        args.roi, 
        args.threshold,
        max_duration_seconds=args.max_duration,
        output_visualization=not args.no_plot,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
