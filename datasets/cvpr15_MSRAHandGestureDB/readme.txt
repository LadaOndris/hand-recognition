MSRA Hand Gesture database is described in the paper
Cascaded Hand Pose Regression, Xiao Sun, Yichen Wei, Shuang Liang, Xiaoou Tang and Jian Sun, CVPR 2015.
Please cite the paper if you use this database.

In total 9 subjects' right hands are captured using Intel's Creative Interactive Gesture Camera. Each subject has 17 gestures captured and there are about 500 frames for each gesture. For subject X and gesture G, the depth images and ground truth 3D hand joints are under .\PX\G\.

The camera intrinsic parameters are: principle point = image center(160, 120), focal length = 241.42.

While the depth image is 320x240, the valid hand region is usually much smaller. To save space, each *.bin file only stores the bounding box of the hand region. Specifically, each bin file starts with 6 unsigned int: img_width img_height left top right bottom. [left, right) and [top, bottom) is the bounding box coordinate in this depth image. The bin file then stores all the depth pixel values in the bounding box in row scanning order, which are  (right - left) * (bottom - top) floats. The unit is millimeters. The bin file is binary and needs to be opened with std::ios::binary flag.

joint.txt file stores 500 frames x 21 hand joints per frame. Each line has 3 * 21 = 63 floats for 21 3D points in (x, y, z) coordinates. The 21 hand joints are: wrist, index_mcp, index_pip, index_dip, index_tip, middle_mcp, middle_pip, middle_dip, middle_tip, ring_mcp, ring_pip, ring_dip, ring_tip, little_mcp, little_pip, little_dip, little_tip, thumb_mcp, thumb_pip, thumb_dip, thumb_tip.

The corresponding *.jpg file is just for visualization of depth and ground truth joints.

For any questiones, please send email to xias@microsoft.com