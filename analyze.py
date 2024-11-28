import numpy as np
import cv2
import matplotlib.pyplot as plt
import video_stabilization as vs

print("hello")

def analyze_video_metrics_compare(original_video_path, stabilized_video_path):
    """
    Analyze video metrics (RMSE, jerkiness, and motion plots) for the original and stabilized videos.

    Args:
        original_video_path (str): Path to the original (unstabilized) video.
        stabilized_video_path (str): Path to the stabilized video.

    Returns:
        None
    """
    def process_video(video_path):
        """
        Process a video to extract motion transforms, trajectory, and smoothed trajectory.

        Args:
            video_path (str): Path to the video.

        Returns:
            tuple: (transforms, trajectory, smoothed_trajectory)
        """
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pre-define transformation-store array
        transforms = np.zeros((n_frames - 1, 3), np.float32)

        # Read the first frame
        _, prev = cap.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        for i in range(n_frames - 2):
            # Detect feature points in the previous frame
            prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                               maxCorners=200,
                                               qualityLevel=0.01,
                                               minDistance=30,
                                               blockSize=3)

            # Read the next frame
            success, curr = cap.read()
            if not success:
                break

            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

            # Filter valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Find transformation matrix
            m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

            if m is not None:
                # Extract translation and rotation
                dx = m[0, 2]
                dy = m[1, 2]
                da = np.arctan2(m[1, 0], m[0, 0])
                transforms[i] = [dx, dy, da]
            else:
                transforms[i] = [0, 0, 0]

            prev_gray = curr_gray

        cap.release()

        # Compute trajectory and smoothed trajectory
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = vs.smooth(trajectory)

        return transforms, trajectory, smoothed_trajectory

    # Process both videos
    transforms_original, trajectory_original, smoothed_original = process_video(original_video_path)
    transforms_stabilized, trajectory_stabilized, smoothed_stabilized = process_video(stabilized_video_path)

    # Compute metrics for both videos
    rmse_original = vs.compute_rmse(trajectory_original, smoothed_original)
    rmse_stabilized = vs.compute_rmse(trajectory_stabilized, smoothed_stabilized)
    jerkiness_original = vs.compute_jerkiness(transforms_original)
    jerkiness_stabilized = vs.compute_jerkiness(transforms_stabilized)

    # Print metrics
    print("Original Video Metrics:")
    print(f"RMSE of motion vectors (dx, dy, da): {rmse_original}")
    print(f"Jerkiness of motion vectors (dx, dy, da): {jerkiness_original}")
    print("\nStabilized Video Metrics:")
    print(f"RMSE of motion vectors (dx, dy, da): {rmse_stabilized}")
    print(f"Jerkiness of motion vectors (dx, dy, da): {jerkiness_stabilized}")

    # Plot trajectories for comparison
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(trajectory_original[:, 0], label='Original X')
    plt.plot(smoothed_original[:, 0], label='Smoothed X', linestyle='dashed')
    plt.plot(trajectory_original[:, 1], label='Original Y')
    plt.plot(smoothed_original[:, 1], label='Smoothed Y', linestyle='dashed')
    plt.title("Original Video Trajectory")
    plt.xlabel("Frame")
    plt.ylabel("Motion")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(trajectory_stabilized[:, 0], label='Stabilized X')
    plt.plot(smoothed_stabilized[:, 0], label='Smoothed X', linestyle='dashed')
    plt.plot(trajectory_stabilized[:, 1], label='Stabilized Y')
    plt.plot(smoothed_stabilized[:, 1], label='Smoothed Y', linestyle='dashed')
    plt.title("Stabilized Video Trajectory")
    plt.xlabel("Frame")
    plt.ylabel("Motion")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()



# analyze_video_metrics_compare('./videos/unstabilized/dog_unstabilized_movement.mp4',
#                       './videos/stabilized/video_out.mp4')