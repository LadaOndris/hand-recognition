Available from: [url link]

The dataset was captured using an SR305 camera. The resolution of images is
640×480 pixels, and the maximum depth is 150 cm. The dataset consists of sequences,
each targeting a specific gesture and conditions. The dataset consists
of three gesture, Gesture 1 (opened palm with fingers stretched and apart), Gesture
2 (number two), and Non-gesture.

Parameters:
principal point = (313.683, 242.755)
focal length = 476.007
depth unit = 0.125 mm

Each line in annotations.txt file consists of the following descriptors:
directory_name gesture_label contains_left_hand contains_right_hand subject

gesture_label is one of three values: 0 (non-gesture), 1 (gesture 1), 2 (gesture 2)
contains_left_hand and contains_right_hand: 1 means True, 0 means False
subject: subject's index starting at 0
