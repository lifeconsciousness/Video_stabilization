import numpy as np
import cv2
import matplotlib.pyplot as plt

SMOOTHING_RADIUS = 50


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # Return smoothed curve
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

# Compute Root mean square error for motion vectors
def compute_rmse(original, smoothed):
    diff = original - smoothed
    # diff = smoothed - original
    rmse = np.sqrt(np.mean(diff ** 2, axis=0))  # RMSE for dx, dy, da separately
    return rmse

# Compute jerkiness (std deviation of frame-to-frame differences)
def compute_jerkiness(transforms):
    frame_differences = np.diff(transforms, axis=0)
    jerkiness = np.std(frame_differences, axis=0)
    return jerkiness




def stabilize_video(path):
    # The larger the SMOOTHING_RADIUS, the more stable the video but less reactive to sudden panning

    # Read input video
    cap = cv2.VideoCapture('videos/unstabilized/dog_unstabilized_movement.mp4')

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Set up output video to only save the stabilized frame
    out = cv2.VideoWriter('./videos/stabilized/video_out.mp4', fourcc, fps, (w, h))

    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix using estimateAffinePartial2D
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

        # If transformation matrix is found
        if m is not None:
            # Extract translation
            dx = m[0, 2]
            dy = m[1, 2]

            # Extract rotation angle
            da = np.arctan2(m[1, 0], m[0, 0])

            # Store transformation
            transforms[i] = [dx, dy, da]
        else:
            # If no matrix found, skip this frame
            print("Transformation matrix not found for frame:", i)
            transforms[i] = [0, 0, 0]

        # Move to next frame
        prev_gray = curr_gray

        print("Frame: " + str(i) + "/" + str(n_frames) + " - Tracked points: " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference





    # # To estimate the effectiveness of stabilization, calculate root-mean-square error (less is better)
    # rmse_motion = compute_rmse(trajectory, smoothed_trajectory)
    # print(f"RMSE of motion vectors (dx, dy, da): {rmse_motion}")
    #
    # #Estimate jerkiness
    # jerkiness_original = compute_jerkiness(transforms)
    # jerkiness_smoothed = compute_jerkiness(transforms_smooth)
    #
    # print(f"Jerkiness before stabilization: {jerkiness_original}")
    # print(f"Jerkiness after stabilization: {jerkiness_smoothed}")
    #
    # # Plot the motion trajectories before and after stabilization
    # plt.figure(figsize=(10, 6))
    # plt.plot(trajectory[:, 0], label='Original X')
    # plt.plot(smoothed_trajectory[:, 0], label='Smoothed X', linestyle='dashed')
    # plt.plot(trajectory[:, 1], label='Original Y')
    # plt.plot(smoothed_trajectory[:, 1], label='Smoothed Y', linestyle='dashed')
    # plt.title("Trajectory Comparison")
    # plt.xlabel("Frame")
    # plt.ylabel("Motion")
    # plt.legend()
    # plt.grid()
    # plt.show()





    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write only the stabilized frame to the output video
        out.write(frame_stabilized)

        # cv2.imshow("Stabilized Frame", frame_stabilized)
        cv2.waitKey(10)

    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()


def analyze_video_metrics(video_path):
    """
    Analyze video metrics (RMSE, jerkiness, and motion plots) for the given video.

    Args:
        video_path (str): Path to the input video.

    Returns:
        None
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

    # Compute trajectory
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)

    # Compute RMSE
    rmse_motion = compute_rmse(trajectory, smoothed_trajectory)

    # Compute jerkiness using frame-to-frame differences
    jerkiness = compute_jerkiness(np.diff(transforms, axis=0))

    # Print metrics
    print(f"RMSE of motion vectors (dx, dy, da): {rmse_motion}")
    print(f"Jerkiness of motion vectors (dx, dy, da): {jerkiness}")

    # Plot trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], label='Original X')
    plt.plot(smoothed_trajectory[:, 0], label='Smoothed X', linestyle='dashed')
    plt.plot(trajectory[:, 1], label='Original Y')
    plt.plot(smoothed_trajectory[:, 1], label='Smoothed Y', linestyle='dashed')
    plt.plot(trajectory[:, 2], label='Original Rotation')
    plt.plot(smoothed_trajectory[:, 2], label='Smoothed Rotation', linestyle='dashed')
    plt.title("Trajectory Comparison")
    plt.xlabel("Frame")
    plt.ylabel("Motion")
    plt.legend()
    plt.grid()
    plt.show()


# analyze_video_metrics("./videos/unstabilized/dog_unstabilized_movement.mp4")
# analyze_video_metrics("./videos/stabilized/video_out.mp4")
































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
        smoothed_trajectory = smooth(trajectory)

        return transforms, trajectory, smoothed_trajectory

    # Process both videos
    transforms_original, trajectory_original, smoothed_original = process_video(original_video_path)
    transforms_stabilized, trajectory_stabilized, smoothed_stabilized = process_video(stabilized_video_path)

    # Compute metrics for both videos
    rmse_original = compute_rmse(trajectory_original, smoothed_original)
    rmse_stabilized = compute_rmse(trajectory_stabilized, smoothed_stabilized)
    jerkiness_original = compute_jerkiness(transforms_original)
    jerkiness_stabilized = compute_jerkiness(transforms_stabilized)

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



analyze_video_metrics_compare('./videos/unstabilized/dog_unstabilized_movement.mp4',
                      './videos/stabilized/video_out.mp4')







