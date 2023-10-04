
# Load and convert datasets
    # Load image pairs
    # Load IMU data
    # Synchronise image data and IMU data

# Calibration
    # Extrinsic and intrinsic parameters

# Visual odometry
    # Camera motion model
        # Remove marine snow (if present?)
        # Perform color correction based on ROV depth (barometer?)
        # Extract features using e.g. SIFT or SURF (For now, later we can use SVO)
        # Compute disparity from stereo cameras to find depth
            # (Image rectification)
            # Correspondance search along epipolar line
        # Obtain the feature world coordinates  (Triangulate)
        # Compute relative motion of camera using PnP (3D-2D)
            # Determine feature correspondences
            # Triangulate to determine relative motion
        # (Bundle adjustment. Scaramuzza recommends it)
    # IMU motion model
        # Compute relative motion using integration
        # Combine IMU estimates from both cameras
    # Perform state prediction by fusing the models

# Mapping
    # Map trajectory
    # Map feature points