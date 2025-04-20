import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import argparse
import os
import time

def analyze_shutter_with_circle_detection(video_path, roi=None, max_time=None, extract_events=False, 
                                         min_radius=60, max_radius=75, debug=False, threshold=None):
    # Standard shutter speeds in seconds
    standard_speeds = [
        1/8000, 1/4000, 1/2000, 1/1000, 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 
        1/8, 1/4, 1/2, 1, 2, 4, 8, 15, 30
    ]
    
    # Create debug directory if needed
    if debug:
        debug_dir = 'debug'
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled. Processed frames will be saved to {debug_dir}/")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return [], [], 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video FPS: {fps}")
    print(f"Video dimensions: {frame_width}x{frame_height}")
    print(f"Looking for circles with radius between {min_radius} and {max_radius} pixels")
    
    # If ROI not specified, use the entire frame
    if roi is None:
        # Use the entire frame
        x = 0
        y = 0
        w = frame_width
        h = frame_height
        roi = (x, y, w, h)
    
    if roi[0] == 0 and roi[1] == 0 and roi[2] == frame_width and roi[3] == frame_height:
        print("Using full frame for circle detection")
    else:
        print(f"Using ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")
    
    # List to track circle presence in each frame
    circle_detected = []
    circle_centers = []  # Store detected circle centers for visualization
    circle_radii = []    # Store detected circle radii for visualization
    frames = []          # Store frames for extraction if needed
    frame_count = 0
    
    # Create CLAHE object for enhanced contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # Process video frame by frame for circle detection
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate current timestamp in seconds
        current_time = frame_count / fps
        
        # Stop processing if we've reached the maximum time
        if max_time is not None and current_time >= max_time:
            print(f"Reached maximum time of {max_time} seconds, stopping processing")
            break
            
        # Store frame if extraction is enabled
        if extract_events:
            frames.append(frame.copy())
            
        # Extract region of interest
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply different processing based on whether thresholding is used
        if threshold is not None:
            # For thresholding, we only need basic blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            _, processed_img = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        else:
            # Apply full contrast enhancement pipeline when not using threshold
            # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe_img = clahe.apply(gray)
            
            # 2. Additional contrast stretching
            # Normalize to 0-255 (full range)
            norm_img = cv2.normalize(clahe_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
            # 3. Apply blur to reduce noise
            processed_img = cv2.GaussianBlur(norm_img, (9, 9), 2)
        
        # Debug: Save processing steps
        if debug and frame_count % 5 == 0:  # Save every 5th frame to reduce disk usage
            if threshold is not None:
                # Create a debug frame for thresholded workflow
                debug_frame = np.zeros((h*2, w*2), dtype=np.uint8)
                debug_frame[0:h, 0:w] = gray
                debug_frame[0:h, w:w*2] = blurred
                debug_frame[h:h*2, 0:w] = processed_img  # Show thresholded image
                debug_frame[h:h*2, w:w*2] = processed_img  # Duplicate for consistency
                
                # Add labels
                cv2.putText(debug_frame, "Original Gray", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                cv2.putText(debug_frame, "Blurred", (w+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                cv2.putText(debug_frame, "Thresholded", (10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                cv2.putText(debug_frame, "Thresholded", (w+10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            else:
                # Original debug frame with full enhancement pipeline
                debug_frame = np.zeros((h*2, w*2), dtype=np.uint8)
                debug_frame[0:h, 0:w] = gray
                debug_frame[0:h, w:w*2] = clahe_img
                debug_frame[h:h*2, 0:w] = norm_img
                debug_frame[h:h*2, w:w*2] = processed_img
                
                # Add labels
                cv2.putText(debug_frame, "Original Gray", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                cv2.putText(debug_frame, "CLAHE", (w+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                cv2.putText(debug_frame, "Normalized", (10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                cv2.putText(debug_frame, "Blurred", (w+10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            
            cv2.imwrite(os.path.join('debug', f'process_frame_{frame_count:06d}.jpg'), debug_frame)
        
        # Apply circle detection
        detected = False
        center_x, center_y, radius = None, None, None
        
        circles = cv2.HoughCircles(
            processed_img,            # Use the processed image (thresholded or blurred)
            cv2.HOUGH_GRADIENT, 
            dp=1,                     # Resolution ratio
            minDist=100,              # Min distance between circles (large as we expect only one)
            param1=70,                # Edge detector upper threshold (increased for contrast enhanced image)
            param2=20,                # Circle detection threshold (reduced to be more sensitive)
            minRadius=min_radius,     # Min radius based on known diameter (~134px)
            maxRadius=max_radius      # Max radius
        )
        
        if circles is not None:
            detected = True
            # Get the strongest circle match
            circles = np.uint16(np.around(circles))
            strongest_circle = circles[0, 0]
            center_x, center_y, radius = strongest_circle[0], strongest_circle[1], strongest_circle[2]
            
            # Debug: Save detected circle visualization
            if debug and frame_count % 5 == 0:
                circle_vis = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
                cv2.circle(circle_vis, (center_x, center_y), radius, (0, 255, 0), 2)
                cv2.circle(circle_vis, (center_x, center_y), 2, (0, 0, 255), 3)
                cv2.putText(circle_vis, f"Frame {frame_count}: Circle r={radius}", (10, h-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imwrite(os.path.join('debug', f'circle_frame_{frame_count:06d}.jpg'), circle_vis)
        elif debug and frame_count % 5 == 0:
            # Debug: Save frames where no circle was detected
            no_circle = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
            cv2.putText(no_circle, f"Frame {frame_count}: No Circle Detected", (10, h-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(os.path.join('debug', f'no_circle_frame_{frame_count:06d}.jpg'), no_circle)
            
        circle_detected.append(detected)
        circle_centers.append((center_x, center_y) if detected else None)
        circle_radii.append(radius if detected else None)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames... (Time: {current_time:.2f}s)")
    
    cap.release()
    print(f"Total frames processed: {frame_count}")
    
    # Find nearest standard shutter speed
    def find_nearest_standard_speed(exposure_time):
        idx = min(range(len(standard_speeds)), 
                  key=lambda i: abs(standard_speeds[i] - exposure_time))
        return standard_speeds[idx]
    
    # Format standard speed as fraction
    def format_speed(speed):
        if speed >= 1:
            return f"{speed:.1f}s"
        else:
            return f"1/{int(1/speed + 0.5)}s"
    
    # Analyze circle detection results to find shutter events
    shutter_events = []
    in_exposure = False
    start_frame = 0
    
    # Apply smoothing to reduce false detections
    window_size = 3
    smoothed_detection = np.convolve(
        np.array(circle_detected, dtype=int), 
        np.ones(window_size)/window_size, 
        mode='same'
    )
    
    # Threshold for considering a circle detected after smoothing
    detection_threshold = 0.5
    
    for i, detected in enumerate(smoothed_detection):
        if not in_exposure and detected > detection_threshold:
            # Shutter just opened (circle appeared)
            in_exposure = True
            start_frame = i
        elif in_exposure and detected <= detection_threshold:
            # Shutter just closed (circle disappeared)
            in_exposure = False
            exposure_frames = i - start_frame
            exposure_time = exposure_frames / fps
            
            # Only include if duration is reasonable (avoid noise)
            if exposure_frames > 2:
                nearest_standard = find_nearest_standard_speed(exposure_time)
                shutter_events.append({
                    'start_frame': start_frame,
                    'end_frame': i,
                    'frame_range': f"Frame {start_frame} to {i}",
                    'frames': exposure_frames,
                    'time_range': f"{start_frame/fps:.2f}s to {i/fps:.2f}s",
                    'exposure_time': exposure_time,
                    'measured_speed': f"1/{1/exposure_time:.1f}s",
                    'standard_speed': format_speed(nearest_standard),
                    'error_percent': abs(nearest_standard - exposure_time) / exposure_time * 100
                })
    
    # Plot circle detection results
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(circle_detected, dtype=int), label='Raw detection')
    plt.plot(smoothed_detection, label='Smoothed detection')
    plt.axhline(y=detection_threshold, color='r', linestyle='-', label='Detection threshold')
    plt.xlabel('Frame')
    plt.ylabel('Circle Detected')
    plt.title('Circle Detection over time')
    plt.legend()
    
    # Mark shutter events on plot
    for event in shutter_events:
        plt.axvspan(event['start_frame'], event['end_frame'], alpha=0.3, color='green')
        # Add text label with frame range and standard speed
        mid_frame = (event['start_frame'] + event['end_frame']) // 2
        plt.text(mid_frame, 0.8, 
                 f"{event['standard_speed']}\n{event['frame_range']}", 
                 horizontalalignment='center')
    
    plt.savefig('circle_detection_analysis.png')
    print("Saved analysis plot to circle_detection_analysis.png")
    
    # Extract frames for each event if requested
    if extract_events and shutter_events and len(frames) > 0:
        print("\nExtracting frames for each shutter event...")
        
        # Create timestamp directory
        timestamp = int(time.time())
        base_dir = str(timestamp)
        os.makedirs(base_dir, exist_ok=True)
        
        for event_idx, event in enumerate(shutter_events):
            # Create event directory
            event_dir = os.path.join(base_dir, f"event-{event_idx+1}")
            os.makedirs(event_dir, exist_ok=True)
            
            # Determine frame range to extract (with padding)
            start_extract = max(0, event['start_frame'] - 10)
            end_extract = min(len(frames) - 1, event['end_frame'] + 10)
            
            # Create event info file
            with open(os.path.join(event_dir, "info.txt"), 'w') as f:
                f.write(f"Event {event_idx+1}\n")
                f.write(f"Original frame range: {event['frame_range']}\n")
                f.write(f"Extracted frame range: {start_extract} to {end_extract}\n")
                f.write(f"Time range: {event['time_range']}\n")
                f.write(f"Measured speed: {event['measured_speed']}\n")
                f.write(f"Standard speed: {event['standard_speed']}\n")
                f.write(f"Exposure frames: {event['frames']}\n")
                f.write(f"Exposure time: {event['exposure_time']*1000:.2f} ms\n")
                f.write(f"Deviation: {event['error_percent']:.1f}%\n")
            
            # Extract and save frames
            for i, frame_idx in enumerate(range(start_extract, end_extract + 1)):
                frame = frames[frame_idx].copy()
                
                # Mark frames that are part of the actual event
                if frame_idx >= event['start_frame'] and frame_idx <= event['end_frame']:
                    # Draw a red border for event frames
                    frame = cv2.rectangle(frame, (0, 0), (frame_width-1, frame_height-1), (0, 0, 255), 5)
                
                # For debugging, draw ROI and detected circles
                x, y, w, h = roi
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Draw corner markers
                corner_length = 20
                # Top-left corner
                cv2.line(frame, (x, y), (x + corner_length, y), (0, 255, 0), 3)
                cv2.line(frame, (x, y), (x, y + corner_length), (0, 255, 0), 3)
                # Top-right corner
                cv2.line(frame, (x+w, y), (x+w - corner_length, y), (0, 255, 0), 3)
                cv2.line(frame, (x+w, y), (x+w, y + corner_length), (0, 255, 0), 3)
                # Bottom-left corner
                cv2.line(frame, (x, y+h), (x + corner_length, y+h), (0, 255, 0), 3)
                cv2.line(frame, (x, y+h), (x, y+h - corner_length), (0, 255, 0), 3)
                # Bottom-right corner
                cv2.line(frame, (x+w, y+h), (x+w - corner_length, y+h), (0, 255, 0), 3)
                cv2.line(frame, (x+w, y+h), (x+w, y+h - corner_length), (0, 255, 0), 3)
                # Add text labels for the corners
                cv2.putText(frame, f"({x},{y})", (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"({x+w},{y+h})", (x+w-80, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # If a circle was detected in this frame, draw it
                if frame_idx < len(circle_centers) and circle_centers[frame_idx] is not None:
                    cx, cy = circle_centers[frame_idx]
                    r = circle_radii[frame_idx]
                    # Adjust coordinates from ROI to full frame
                    cv2.circle(frame, (x + cx, y + cy), r, (0, 255, 255), 2)
                
                # Save frame
                frame_filename = os.path.join(event_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
            
            print(f"  Extracted {end_extract - start_extract + 1} frames for event {event_idx+1}")
        
        print(f"All events extracted to directory: {base_dir}/")
    
    return shutter_events, circle_detected, fps

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze camera shutter speed using circle detection.')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--stop-at', type=float, 
                        help='Stop processing at this timestamp in seconds')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
                        help='Region of interest (x, y, width, height)')
    parser.add_argument('--roi-corners', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Region of interest as corners (top-left x, top-left y, bottom-right x, bottom-right y)')
    parser.add_argument('--min-radius', type=int, default=60,
                        help='Minimum circle radius to detect (default: 60)')
    parser.add_argument('--max-radius', type=int, default=75,
                        help='Maximum circle radius to detect (default: 75)')
    parser.add_argument('--extract', action='store_true',
                        help='Extract frames for each detected shutter event')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to save processing steps to debug folder')
    parser.add_argument('--threshold', type=int, choices=range(1, 256), metavar="[1-255]",
                        help='Threshold value (1-255) to separate black shutter from white circle')
    args = parser.parse_args()
    
    # Set ROI if provided
    roi = None
    if args.roi:
        roi = args.roi
    elif args.roi_corners:
        x1, y1, x2, y2 = args.roi_corners
        # Convert to x, y, width, height format
        roi = (x1, y1, x2-x1, y2-y1)
    
    # Run analysis
    shutter_events, circle_detected, fps = analyze_shutter_with_circle_detection(
        args.video_path, 
        roi=roi, 
        max_time=args.stop_at,
        extract_events=args.extract,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        debug=args.debug,
        threshold=args.threshold
    )

    print("\nShutter Event Results:")
    print(f"Total events detected: {len(shutter_events)}")
    for i, event in enumerate(shutter_events):
        print(f"\nShutter event {i+1}:")
        print(f"  {event['frame_range']} ({event['time_range']})")
        print(f"  Frames: {event['frames']}")
        print(f"  Exposure time: {event['exposure_time']*1000:.2f} ms")
        print(f"  Measured as: {event['measured_speed']}")
        print(f"  Closest standard speed: {event['standard_speed']}")
        print(f"  Deviation: {event['error_percent']:.1f}%")

    # Write results to file
    with open('shutter_results.txt', 'w') as f:
        f.write(f"Video FPS: {fps}\n")
        f.write(f"Total events detected: {len(shutter_events)}\n\n")
        for i, event in enumerate(shutter_events):
            f.write(f"Shutter event {i+1}:\n")
            f.write(f"  {event['frame_range']} ({event['time_range']})\n")
            f.write(f"  Frames: {event['frames']}\n")
            f.write(f"  Exposure time: {event['exposure_time']*1000:.2f} ms\n")
            f.write(f"  Measured as: {event['measured_speed']}\n")
            f.write(f"  Closest standard speed: {event['standard_speed']}\n")
            f.write(f"  Deviation: {event['error_percent']:.1f}%\n\n")
    print("Saved results to shutter_results.txt")

if __name__ == "__main__":
    main()
