### `rpn` module overview

##### `generate_anchors.py`

Generates a regular grid of multi-scale, multi-aspect anchor boxes.

##### `proposal_layer.py`

Converts RPN outputs (per-anchor scores and bbox regression estimates) into object proposals.

##### `anchor_target_layer.py` 

Generates training targets/labels for each anchor. Classification labels are 1 (object), 0 (not object) or -1 (ignore).
Bbox regression targets are specified when the classification label is > 0.

##### `proposal_target_layer.py`

Generates training targets/labels for each object proposal: classification labels 0 - K (bg or object class 1, ... , K)
and bbox regression targets in that case that the label is > 0.

##### `generate.py`

Generate object detection proposals from an imdb using an RPN.
