import numpy as np
import cv2
import matplotlib.pyplot as plt
import video_stabilization as vs


def analyze_video_metrics(original_video_path, stabilized_video_path):
    """
    Analyze video metrics (RMSE, jerkiness, and motion plots) for the original and stabilized videos,
    and display four frames with motion vectors (as arrows) drawn for both original and stabilized videos.
    """

    def process_video(video_path):
        """
        Process a video to extract motion transforms, trajectory, and smoothed trajectory.

        Args:
            video_path (str): Path to the video.

        Returns:
            tuple: (transforms, trajectory, smoothed_trajectory, frames_with_vectors)
        """
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        transforms = np.zeros((n_frames - 1, 3), np.float32)
        frames_with_vectors = []

        _, prev = cap.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        for i in range(n_frames - 2):
            prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                               maxCorners=200,
                                               qualityLevel=0.01,
                                               minDistance=30,
                                               blockSize=3)

            success, curr = cap.read()
            if not success:
                break

            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            if m is not None:
                dx = m[0, 2]
                dy = m[1, 2]
                da = np.arctan2(m[1, 0], m[0, 0])
                transforms[i] = [dx, dy, da]
            else:
                transforms[i] = [0, 0, 0]

            # Draw motion vectors for first two frames
            if i < 2:
                frame_copy = curr.copy()
                for p1, p2 in zip(prev_pts, curr_pts):
                    x1, y1 = p1.ravel()
                    x2, y2 = p2.ravel()
                    cv2.arrowedLine(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, tipLength=0.5)
                frames_with_vectors.append((frame_copy, prev_pts, curr_pts))

            prev_gray = curr_gray

        cap.release()
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = vs.smooth(trajectory)

        return transforms, trajectory, smoothed_trajectory, frames_with_vectors

    # Process both videos
    orig_results = process_video(original_video_path)
    stab_results = process_video(stabilized_video_path)

    _, trajectory_original, smoothed_original, frames_original = orig_results
    _, trajectory_stabilized, smoothed_stabilized, frames_stabilized = stab_results

    # Display frames with motion vectors
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, ((frame, prev_pts, curr_pts), ax) in enumerate(zip(frames_original + frames_stabilized, axs.flat)):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"{'Original' if i < 2 else 'Stabilized'} Video - Frame {i % 2 + 1}"
        )
        # Overlay motion vectors as arrows
        for p1, p2 in zip(prev_pts, curr_pts):
            x1, y1 = p1.ravel()
            x2, y2 = p2.ravel()
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color='red', lw=1.5))
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # The rest of the metrics analysis remains unchanged

analyze_video_metrics('./videos/unstabilized/drone_unstabilized.mp4', './videos/stabilized/drone_stabilized_online.mp4')
