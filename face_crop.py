import cv2
import numpy as np
import face_recognition
import argparse

def estimate_average_face_location(video_capture):
    # Estimate the average face location for all frames
    print("Estimating the average face location...")
    face_locations = []
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(total_frames * 0.15)
    if step >= total_frames:
        print("Step is greater than or equal to total frames. Exiting...")
        return
    for frame_num in range(0, total_frames, step):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video_capture.read()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_face_locations = face_recognition.face_locations(image_rgb)
        if frame_face_locations:
            avg_top = 0
            avg_right = 0
            avg_bottom = 0
            avg_left = 0
            for face_location in frame_face_locations:
                top, right, bottom, left = face_location
                avg_top += top
                avg_right += right
                avg_bottom += bottom
                avg_left += left
            avg_top //= len(frame_face_locations)
            avg_right //= len(frame_face_locations)
            avg_bottom //= len(frame_face_locations)
            avg_left //= len(frame_face_locations)
            face_locations.append((avg_top, avg_right, avg_bottom, avg_left))
        # print the progress in the console
        progress = (frame_num / total_frames) * 100
        print(f"Processed {frame_num} frames ({progress:.2f}% complete)")
    print("Average face location estimation completed.")
    return face_locations

def crop_video(input, output, crop_ratio):
    video_capture = cv2.VideoCapture(input)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output, fourcc, fps, (256, 256))
    frame_count = 0

    # First pass: Detect face locations
    print("Starting first pass to detect face location...")
    #prints the total number of video frames
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    face_locations_1Pass = estimate_average_face_location(video_capture)

    # Reset video capture to the beginning
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    # Second pass: Create the new video     
    print("Starting to process the video...")
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Padding percentage (5% by default)
        padding_percentage = crop_ratio
        
        # Uses the face location from 1st pass
        face_locations = face_locations_1Pass
        top, right, bottom, left = face_locations_1Pass[0]

        # Calculate face perimeter (sum of the sides of the bounding box)
        face_perimeter = 2 * ((right - left) + (bottom - top))
        #print(f"Estimated face perimeter: {face_perimeter} pixels")
        

        # Extract the face region
        padding = int(face_perimeter * padding_percentage)
        top -= padding
        right += padding
        bottom += padding
        left -= padding
        face_region = frame[max(0, top):min(frame.shape[0], bottom), max(0, left):min(frame.shape[1], right)]
        
        
        # Resize the face region to fit into a 256x256 frame while maintaining aspect ratio
        face_height, face_width = face_region.shape[:2]
        scale = min(256 / face_width, 256 / face_height)
        resized_face = cv2.resize(face_region, (int(face_width * scale), int(face_height * scale)))
        
        # Create a 256x256 black frame
        output_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        # Calculate top-left corner to center the face in the 256x256 frame
        start_x = (256 - resized_face.shape[1]) // 2
        start_y = (256 - resized_face.shape[0]) // 2
        # Place the resized face in the output frame
        output_frame[start_y:start_y + resized_face.shape[0], start_x:start_x + resized_face.shape[1]] = resized_face

        # Write the output frame to the video file
        output_video.write(output_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count} frames ({progress:.2f}% complete)")

    print("Processing completed.")
    # Release the video files
    video_capture.release()
    output_video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop video to 256x256 pixels centered on the average face position.")
    parser.add_argument("--input",  type=str,   help="Path to the input video file")
    parser.add_argument("--output", type=str,   help="Path to the output video file")
    parser.add_argument("--crop_ratio", type=float, default=0.10, help="Aspect ratio of the crop (default: 10%)")
    
    args = parser.parse_args()
    crop_video(args.input, args.output, args.crop_ratio)