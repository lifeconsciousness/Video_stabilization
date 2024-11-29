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

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(trajectory_original[:, 0], label='Original X', color='blue')
    plt.plot(trajectory_original[:, 1], label='Original Y', color='green')
    plt.plot(trajectory_stabilized[:, 0], label='Stabilized X', color='orange', linestyle='dashed')
    plt.plot(trajectory_stabilized[:, 1], label='Stabilized Y', color='red', linestyle='dashed')
    plt.title("Motion Trajectory Comparison")
    plt.xlabel("Frame")
    plt.ylabel("Motion (X, Y)")
    plt.legend()
    plt.grid()
    plt.show()

    # # Plot trajectories for comparison
    # plt.figure(figsize=(12, 8))
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(trajectory_original[:, 0], label='Original X')
    # plt.plot(smoothed_original[:, 0], label='Smoothed X', linestyle='dashed')
    # plt.plot(trajectory_original[:, 1], label='Original Y')
    # plt.plot(smoothed_original[:, 1], label='Smoothed Y', linestyle='dashed')
    # plt.title("Original Video Trajectory")
    # plt.xlabel("Frame")
    # plt.ylabel("Motion")
    # plt.legend()
    # plt.grid()
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(trajectory_stabilized[:, 0], label='Stabilized X')
    # plt.plot(smoothed_stabilized[:, 0], label='Smoothed X', linestyle='dashed')
    # plt.plot(trajectory_stabilized[:, 1], label='Stabilized Y')
    # plt.plot(smoothed_stabilized[:, 1], label='Smoothed Y', linestyle='dashed')
    # plt.title("Stabilized Video Trajectory")
    # plt.xlabel("Frame")
    # plt.ylabel("Motion")
    # plt.legend()
    # plt.grid()
    #
    # plt.tight_layout()
    # plt.show()



# analyze_video_metrics_compare('./videos/unstabilized/dog_unstabilized_movement.mp4',
#                       './videos/stabilized/video_out.mp4')


def analyze_video_metrics(original_video_path, stabilized_video_path):
    """
    Analyze motion metrics (RMSE and jerkiness) for the original and stabilized videos.

    Args:
        original_video_path (str): Path to the original video.
        stabilized_video_path (str): Path to the stabilized video.

    Returns:
        None
    """
    def compute_trajectory(video_path):
        """
        Compute motion trajectory for a video.

        Args:
            video_path (str): Path to the video.

        Returns:
            np.ndarray: Motion trajectory as a numpy array of shape (n_frames-1, 3).
        """
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        return np.cumsum(transforms, axis=0)

    def compute_rmse(original_trajectory, stabilized_trajectory):
        """
        Compute RMSE for each motion component (dx, dy, da).

        Args:
            original_trajectory (np.ndarray): Original motion trajectory.
            stabilized_trajectory (np.ndarray): Stabilized motion trajectory.

        Returns:
            np.ndarray: RMSE for each motion component (dx, dy, da).
        """
        min_length = min(len(original_trajectory), len(stabilized_trajectory))
        original_trajectory = original_trajectory[:min_length]
        stabilized_trajectory = stabilized_trajectory[:min_length]
        return np.sqrt(np.mean((original_trajectory - stabilized_trajectory) ** 2, axis=0))

    def compute_jerkiness(transforms_diff):
        """
        Compute jerkiness for each motion component (dx, dy, da).

        Args:
            transforms_diff (np.ndarray): Frame-to-frame differences in motion.

        Returns:
            np.ndarray: Jerkiness for each motion component (dx, dy, da).
        """
        return np.sum(np.abs(transforms_diff), axis=0)

    # Compute trajectories
    original_trajectory = compute_trajectory(original_video_path)
    stabilized_trajectory = compute_trajectory(stabilized_video_path)

    # Compute RMSE
    rmse_motion = compute_rmse(original_trajectory, stabilized_trajectory)

    # Compute jerkiness
    original_jerkiness = compute_jerkiness(np.diff(original_trajectory, axis=0))
    stabilized_jerkiness = compute_jerkiness(np.diff(stabilized_trajectory, axis=0))

    # Print overall metrics
    print(f"RMSE between original and stabilized motion trajectories: {np.mean(rmse_motion):.4f}")
    print(f"Jerkiness of original video: {np.sum(original_jerkiness):.4f}")
    print(f"Jerkiness of stabilized video: {np.sum(stabilized_jerkiness):.4f}")

    # Print detailed metrics
    print("\nOriginal Video Metrics:")
    print(f"RMSE of motion vectors (dx, dy, da): {rmse_motion}")
    print(f"Jerkiness of motion vectors (dx, dy, da): {original_jerkiness}")

    print("\nStabilized Video Metrics:")
    print(f"RMSE of motion vectors (dx, dy, da): {rmse_motion}")
    print(f"Jerkiness of motion vectors (dx, dy, da): {stabilized_jerkiness}")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(original_trajectory[:, 0], label='Original X', color='blue')
    plt.plot(original_trajectory[:, 1], label='Original Y', color='green')
    plt.plot(stabilized_trajectory[:, 0], label='Stabilized X', color='orange', linestyle='dashed')
    plt.plot(stabilized_trajectory[:, 1], label='Stabilized Y', color='red', linestyle='dashed')
    plt.title("Motion Trajectory Comparison")
    plt.xlabel("Frame")
    plt.ylabel("Motion (X, Y)")
    plt.legend()
    plt.grid()
    plt.show()



def compute_rmse_old(original_trajectory, stabilized_trajectory):
    """
    Compute RMSE between original and stabilized trajectories.

    Args:
        original_trajectory (np.ndarray): Original motion trajectory.
        stabilized_trajectory (np.ndarray): Stabilized motion trajectory.

    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.mean((original_trajectory - stabilized_trajectory) ** 2))


def compute_rmse(original_trajectory, stabilized_trajectory):
    """
    Compute RMSE between original and stabilized trajectories.

    Args:
        original_trajectory (np.ndarray): Original motion trajectory.
        stabilized_trajectory (np.ndarray): Stabilized motion trajectory.

    Returns:
        float: RMSE value.
    """
    # Find the minimum length between the two trajectories
    min_length = min(len(original_trajectory), len(stabilized_trajectory))

    # Truncate both trajectories
    original_trajectory = original_trajectory[:min_length]
    stabilized_trajectory = stabilized_trajectory[:min_length]

    # Compute RMSE
    return np.sqrt(np.mean((original_trajectory - stabilized_trajectory) ** 2))


def compute_jerkiness(transforms_diff):
    """
    Compute jerkiness as the sum of frame-to-frame differences.

    Args:
        transforms_diff (np.ndarray): Frame-to-frame differences in motion.

    Returns:
        float: Jerkiness value.
    """
    return np.sum(np.linalg.norm(transforms_diff, axis=1))


# analyze_video_metrics_compare('./videos/unstabilized/dog_unstabilized_movement.mp4',
#                       './videos/stabilized/dog_movement_cv.mp4')

# analyze_video_metrics_compare('./videos/unstabilized/dog_unstabilized_static.mp4',
#                       './videos/stabilized/dog_static_cv.mp4')


analyze_video_metrics('./videos/unstabilized/dog_unstabilized_static.mp4',
                      './videos/stabilized/dog_static_cv.mp4')


