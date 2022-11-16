import cv2

def disparity_mapper(calibrated=False):
    # stereo disparity solver configuration
    if calibrated:
        block_size = 5
        min_disp = -64
        max_disp = 64
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 5
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 0
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 1
        disp12MaxDiff = 0
    else:
        block_size = 7
        min_disp = -128
        max_disp = 128
        num_disp = max_disp - min_disp
        uniquenessRatio = 5
        speckleWindowSize = 200
        speckleRange = 1
        disp12MaxDiff = 0

    mapper = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
    )
    return mapper

if __name__ == "__main__":
    disparity_mapper()