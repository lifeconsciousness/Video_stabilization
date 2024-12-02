# Video Stabilization Using Point Feature Matching in OpenCV

## How to stabilize video

- Go to video_stabilization.py
- In the function call specify path to video to be stabilized and output path with file name

```python
stabilize_video('videos/unstabilized/dog_unstabilized_movement.mp4', 
                './videos/stabilized/video_out.mp4')
```

## How to analyze the effectiveness of stabilization
- Go to analyze.py
- In the function call specify path to before and after stabilization videos

```python
 analyze_video_metrics('./videos/unstabilized/dog_unstabilized_static.mp4',
                       './videos/stabilized/dog_static_cv.mp4')
```

### Credits

[Repository the code is based on](https://github.com/spmallick/learnopencv/tree/master/VideoStabilization)

[Sampled video dog](https://www.youtube.com/watch?v=n2BwI-KhcYs)

[Sampled video drone](https://www.youtube.com/watch?v=mQxnB2X26CI)