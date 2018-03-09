### Scale invariant features or adaptive window selection.
1) For our adaptive scale invariant detector, I compute the descriptor for each Harris Keypoint at a number of scales. I experimented with a few values for the downscaling fraction and the number of total scales. I finally used the following configurations - number_of_scales = 4 and with a scale down fraction of ½ at each scale.
2) For each Harris feature I compute the descriptor at each of the scales. To compute the descriptor at a scale, I first map the location of the coordinates at the original scale to the scaled down images. Then, at each scale, we compute the descriptor in the standard manner and these are then added to the total number of descriptors.
3) Hence, in the configuration described above, the total_number_of_descriptors = number_of_scale * number_of_keypoints. This approach introduces many outliers which can then be addressed using RANSAC.

### Reducing number of descriptors
1) Then, I experimented with techniques to find the best descriptor for a single keypoint among the different scales. In the first variation, I use the standard deviation of the descriptor to choose the descriptor which has the highest std. This was based on the intuition that this descriptor describes a richer surrounding. Results for this setup are reported in the report.
2) In the second variation, I chose the descriptor from the scales randomly! Results for this setup are reported in the report.
3) In the 2nd and 3rd variations described above, the total_number_of_descriptors = number_of_keypoints.

### Sanity Checks
We add some sanity checks in the code. One being that when subsampling the image, the images should never reduce a shape in which the height or the width is less than 40 pixels.

### Misc Notes -
Before we subsample an image at any scale, we use a gaussian blur with a sigma_pyramid == 1.0. This was inspired by the values used in the MOPS paper, which is a variant of the method I have used and described above.
