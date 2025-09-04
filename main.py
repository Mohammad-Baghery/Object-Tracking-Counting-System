
import argparse
import cv2
import time
import sys
from pathlib import Path
import numpy as np

# Import custom modules
from object_tracker import ObjectTracker
from counter import ObjectCounter
from utils import (
    draw_bounding_box, draw_counting_line, draw_statistics,
    resize_frame, calculate_fps, format_time
)
import config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Object Tracking & Counting System")

    parser.add_argument(
        '--source', type=str, default='0',
        help='Input source: 0 for webcam, path for video file'
    )
    parser.add_argument(
        '--model', type=str, default=config.YOLO_MODEL,
        help='YOLOv8 model path'
    )
    parser.add_argument(
        '--confidence', type=float, default=config.CONFIDENCE_THRESHOLD,
        help='Detection confidence threshold'
    )
    parser.add_argument(
        '--line_position', type=int, default=config.COUNTING_LINE_POSITION,
        help='Y coordinate of counting line'
    )
    parser.add_argument(
        '--classes', nargs='+', type=int, default=config.TRACK_CLASSES,
        help='Object classes to track (COCO class IDs)'
    )
    parser.add_argument(
        '--save_output', action='store_true',
        help='Save output video'
    )
    parser.add_argument(
        '--output_path', type=str, default=config.OUTPUT_PATH,
        help='Output video path'
    )
    parser.add_argument(
        '--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
        help='Resize frame to specified dimensions'
    )
    parser.add_argument(
        '--fps_limit', type=int, default=None,
        help='Limit processing FPS'
    )
    parser.add_argument(
        '--no_display', action='store_true',
        help='Run without display (headless mode)'
    )

    return parser.parse_args()


def setup_video_source(source: str):
    """Setup video capture source"""
    # Check if source is webcam (integer) or file path
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        source_type = "webcam"
    else:
        if not Path(source).exists():
            print(f"❌ Error: Video file not found: {source}")
            return None, None
        cap = cv2.VideoCapture(source)
        source_type = "video"

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video source: {source}")
        return None, None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video source: {source} ({source_type})")
    print(f"📏 Resolution: {width}x{height}")
    print(f"🎬 FPS: {fps:.2f}")
    if source_type == "video":
        print(f"⏱️  Duration: {format_time(total_frames / fps)}")

    return cap, {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'source_type': source_type
    }


def setup_video_writer(output_path: str, fps: float, width: int, height: int):
    """Setup video writer for output"""
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"❌ Error: Cannot create video writer: {output_path}")
        return None

    print(f"💾 Output will be saved to: {output_path}")
    return out


def main():
    """Main function"""
    print("🚀 Starting Object Tracking & Counting System")
    print("=" * 50)

    # Parse arguments
    args = parse_arguments()

    # Setup video source
    cap, video_info = setup_video_source(args.source)
    if cap is None:
        return

    # Initialize tracker
    print("\n🔧 Initializing components...")
    try:
        tracker = ObjectTracker(
            model_path=args.model,
            device=config.DEVICE
        )
        tracker.set_confidence_threshold(args.confidence)
        tracker.set_track_classes(args.classes)
    except Exception as e:
        print(f"❌ Error initializing tracker: {e}")
        return

    # Initialize counter
    counter = ObjectCounter(counting_line_position=args.line_position)

    # Setup video writer if needed
    video_writer = None
    if args.save_output:
        output_width = video_info['width']
        output_height = video_info['height']

        if args.resize:
            output_width, output_height = args.resize

        video_writer = setup_video_writer(
            args.output_path,
            video_info['fps'],
            output_width,
            output_height
        )

    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps_limit = args.fps_limit
    frame_time_target = 1.0 / fps_limit if fps_limit else 0

    print("\n▶️  Starting processing...")
    print("Press 'q' to quit, 'r' to reset counter, 's' to save screenshot")
    print("=" * 50)

    try:
        while True:
            frame_start_time = time.time()

            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("📹 End of video or cannot read frame")
                break

            # Resize frame if requested
            if args.resize:
                frame = resize_frame(frame, tuple(args.resize))

            # Process frame
            tracking_results, processed_frame = tracker.process_frame(frame)

            # Update counter
            count_stats = counter.update(tracking_results)

            # Draw visualizations
            for result in tracking_results:
                processed_frame = draw_bounding_box(
                    processed_frame,
                    result['bbox'],
                    result['track_id'],
                    result['class_name'],
                    result['confidence']
                )

            # Draw counting line
            processed_frame = draw_counting_line(processed_frame, args.line_position)

            # Draw statistics
            processed_frame = draw_statistics(processed_frame, count_stats)

            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            current_fps = calculate_fps(frame_count, elapsed_time)

            cv2.putText(
                processed_frame,
                f"FPS: {current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Save frame if video writer is setup
            if video_writer is not None:
                video_writer.write(processed_frame)

            # Display frame (if not in headless mode)
            if not args.no_display:
                cv2.imshow('Object Tracking & Counting', processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Exiting...")
                    break
                elif key == ord('r'):
                    counter.reset_counter()
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")

            # FPS limiting
            if fps_limit:
                frame_process_time = time.time() - frame_start_time
                if frame_process_time < frame_time_target:
                    time.sleep(frame_time_target - frame_process_time)

            # Progress update for video files
            if video_info['source_type'] == 'video' and frame_count % 30 == 0:
                progress = (frame_count / video_info['total_frames']) * 100
                print(f"📊 Progress: {progress:.1f}% | FPS: {current_fps:.1f} | "
                      f"Total Count: {counter.total_count}")

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")

    except Exception as e:
        print(f"❌ Error during processing: {e}")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        cap.release()

        if video_writer is not None:
            video_writer.release()

        if not args.no_display:
            cv2.destroyAllWindows()

        # Print final statistics
        print("\n" + "=" * 50)
        print("📊 FINAL STATISTICS")
        print("=" * 50)
        print(counter.export_statistics())

        # Performance summary
        total_time = time.time() - start_time
        print(f"⏱️  Total processing time: {format_time(total_time)}")
        print(f"🎬 Frames processed: {frame_count}")
        print(f"📈 Average FPS: {calculate_fps(frame_count, total_time):.2f}")

        print("\n✅ Processing completed successfully!")


if __name__ == "__main__":
    main()
