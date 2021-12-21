# visualize
This is a simple project relied on mmdet to visualize object detection annotations or results.

### Dependency
The main code is depedent to the mmdetection code from `mmdetection/mmdet/core/visualization/image.py`. So you should install `mmdet` and `mmcv`

### How to use
You should revise `main.py` to visualize annotations, which does not support `argparse` up to now.
You can set the `name` to opt different annotation formats, including `COCO` and `xml`.
The detection result will be supported later.
