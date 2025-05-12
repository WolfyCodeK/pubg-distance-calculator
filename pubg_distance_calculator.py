import cv2
import numpy as np

def load_image(image_path):
    """Loads an image from the specified path."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        print("Please ensure the image path is correct and the file exists.")
    return img

def display_image(window_name, image):
    """Displays an image in a window."""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

def merge_line_segments(lines, is_horizontal, primary_axis_tolerance, secondary_axis_max_gap):
    """
    Merges collinear and overlapping/nearby line segments.
    For horizontal lines: primary_axis is y, secondary_axis is x.
    For vertical lines: primary_axis is x, secondary_axis is y.
    """
    if not lines:
        return []

    # Sort lines:
    # Horizontal: by y1, then x1
    # Vertical: by x1, then y1
    if is_horizontal:
        lines.sort(key=lambda line: (line[1], line[0])) 
    else: # Vertical
        lines.sort(key=lambda line: (line[0], line[1]))

    merged_lines = []
    used = [False] * len(lines)

    for i in range(len(lines)):
        if used[i]:
            continue

        current_segment = list(lines[i]) # Make a mutable copy
        used[i] = True
        
        # Accumulate points for averaging primary coordinate if needed (e.g. y for horizontal)
        primary_coords_sum = current_segment[1] if is_horizontal else current_segment[0]
        count_in_merge = 1

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue

            next_segment = lines[j]
            
            # Primary axis check (are they on the same "level"?)
            # Horizontal: y1 vs y1; Vertical: x1 vs x1
            primary_coord_current = current_segment[1] if is_horizontal else current_segment[0]
            primary_coord_next = next_segment[1] if is_horizontal else next_segment[0]

            if abs(primary_coord_next - primary_coord_current) > primary_axis_tolerance:
                # If sorted primarily by this axis, lines further down won't match either if current group is small
                # This simple break might not be optimal if primary_axis_tolerance is large.
                # A more robust way would be to average the primary_coord_current of the growing merged line.
                # For now, using the first segment's primary coord for comparison with others.
                # Or, average the primary coordinates of all segments in the current merge.
                # Let's refine this: compare next_segment to the *average* primary coord of the current merged group
                avg_primary_coord_of_current_merged_group = primary_coords_sum / count_in_merge
                if abs(primary_coord_next - avg_primary_coord_of_current_merged_group) > primary_axis_tolerance:
                    continue # Not on the same primary axis level


            # Secondary axis check (do they overlap or are they close enough to bridge a gap?)
            # Horizontal: x-coords; Vertical: y-coords
            s1_current_start = current_segment[0] if is_horizontal else current_segment[1]
            s1_current_end = current_segment[2] if is_horizontal else current_segment[3]
            s2_next_start = next_segment[0] if is_horizontal else next_segment[1]
            s2_next_end = next_segment[2] if is_horizontal else next_segment[3]

            # Check for overlap or if gap is bridgeable
            # They are considered mergeable if:
            # max_start_coord <= min_end_coord + secondary_axis_max_gap
            mergeable_on_secondary = max(s1_current_start, s2_next_start) <= \
                                     min(s1_current_end, s2_next_end) + secondary_axis_max_gap

            if mergeable_on_secondary:
                # Merge: update extents of current_segment and mark next_segment as used
                if is_horizontal:
                    current_segment[0] = min(s1_current_start, s2_next_start) # New x1
                    current_segment[2] = max(s1_current_end, s2_next_end)   # New x2
                    # Update average y
                    primary_coords_sum += next_segment[1] 
                    count_in_merge += 1
                    current_segment[1] = current_segment[3] = int(round(primary_coords_sum / count_in_merge)) # new y1, y2
                else: # Vertical
                    current_segment[1] = min(s1_current_start, s2_next_start) # New y1
                    current_segment[3] = max(s1_current_end, s2_next_end)   # New y2
                    # Update average x
                    primary_coords_sum += next_segment[0]
                    count_in_merge += 1
                    current_segment[0] = current_segment[2] = int(round(primary_coords_sum / count_in_merge)) # new x1, x2
                used[j] = True
        
        merged_lines.append(tuple(current_segment))
    
    return merged_lines


def detect_grid_and_calculate_scale(image, original_map_image_for_drawing):
    """
    Detects grid lines in the image and calculates the pixel-to-meter scale.
    """
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)

    # Apply Canny edge detection directly to grayscale
    canny_threshold1 = 20 
    canny_threshold2 = 60 
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
    cv2.imshow("Canny Edges on Grayscale (Raw)", edges)
    cv2.waitKey(0)

    # --- First Hough Pass to find all potential short straight segments --- 
    hough_threshold_prune = 10
    min_line_length_prune = 25 # Slightly less than user's 20px target
    max_line_gap_prune = 5
    
    pruning_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold_prune, 
                                    minLineLength=min_line_length_prune, 
                                    maxLineGap=max_line_gap_prune)
    
    # Create a mask from these pruning lines
    line_mask = np.zeros_like(edges)
    if pruning_lines is not None:
        for line in pruning_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1) # Draw these lines on the mask
    cv2.imshow("Mask of Short Straight Segments", line_mask)
    cv2.waitKey(0)

    # Prune the original Canny edges using this mask
    pruned_edges = cv2.bitwise_and(edges, edges, mask=line_mask)
    cv2.imshow("Canny Edges (Pruned by Short Lines)", pruned_edges)
    cv2.waitKey(0)

    # --- Second Hough Line Transform (Grid Detection) on PRUNED edges --- 
    # Use the pruned_edges as input now
    current_edges_input_for_grid_hough = pruned_edges

    hough_lines_display_image = original_map_image_for_drawing.copy() 
    all_hough_lines_image = original_map_image_for_drawing.copy() 
    
    initial_h_lines = []
    initial_v_lines = []
    
    hough_found_any = False
    pink_lines_drawn = False

    if current_edges_input_for_grid_hough is not None and current_edges_input_for_grid_hough.any():
        rho = 1  
        theta = np.pi / 180  
        hough_threshold_grid = 15     # Main grid detection threshold 
        min_line_length_grid = 20 # Main grid detection min length
        max_line_gap_grid = 7    # Main grid detection max gap
        
        detected_lines = cv2.HoughLinesP(current_edges_input_for_grid_hough, rho, theta, hough_threshold_grid,   
                                        minLineLength=min_line_length_grid, 
                                        maxLineGap=max_line_gap_grid)
        
        if detected_lines is not None:
            hough_found_any = True
            # Draw ALL detected lines in BLUE on the separate image
            # Ensure all_hough_lines_image is initialized before this loop if it might be empty
            if 'all_hough_lines_image' not in locals() and 'all_hough_lines_image' not in globals():
                 all_hough_lines_image = original_map_image_for_drawing.copy()

            for line_segment in detected_lines:
                x1_h, y1_h, x2_h, y2_h = line_segment[0]
                cv2.line(all_hough_lines_image, (x1_h, y1_h), (x2_h, y2_h), (255, 0, 0), 1) # Blue, thickness 1

            # Filter for horizontal/vertical for the PINK lines image & collect them
            angle_tolerance_pixels = 7 

            for line_segment in detected_lines: # Iterate again for clarity, or combine loops
                x1, y1, x2, y2 = line_segment[0]
                
                is_horizontal = abs(y1 - y2) <= angle_tolerance_pixels
                is_vertical = abs(x1 - x2) <= angle_tolerance_pixels

                if is_horizontal:
                    # Make it truly horizontal for merging
                    avg_y = int(round((y1 + y2) / 2))
                    initial_h_lines.append((min(x1,x2), avg_y, max(x1,x2), avg_y))
                    cv2.line(hough_lines_display_image, (min(x1,x2), avg_y), (max(x1,x2), avg_y), (255, 0, 255), 2)
                    pink_lines_drawn = True
                elif is_vertical:
                    # Make it truly vertical
                    avg_x = int(round((x1 + x2) / 2))
                    initial_v_lines.append((avg_x, min(y1,y2), avg_x, max(y1,y2)))
                    cv2.line(hough_lines_display_image, (avg_x, min(y1,y2)), (avg_x, max(y1,y2)), (255, 0, 255), 2)
                    pink_lines_drawn = True
            
            if pink_lines_drawn:
                print(f"Collected {len(initial_h_lines)} initial H lines and {len(initial_v_lines)} initial V lines.")
            else:
                print("Detected lines by Hough, but none met H/V criteria with tolerance.")
        else:
            print("HoughLinesP did not detect any lines.")
    else:
        print("Edge image is empty, skipping Hough Line Transform.")

    if hough_found_any:
        cv2.imshow("ALL Detected Hough Lines (Blue)", all_hough_lines_image)
        cv2.waitKey(0)

    if pink_lines_drawn:
        cv2.imshow("Detected Grid Lines (Pink H-V Initial Filtered)", hough_lines_display_image)
        cv2.waitKey(0)
    
    # --- Line Merging Re-enabled ---
    # print("Line merging is currently disabled to focus on initial H/V detection.") # Re-enable this section
    y_coord_tolerance_for_horizontal_merge = 10
    max_gap_for_horizontal_merge = 25 
    merged_h_lines = merge_line_segments(initial_h_lines, True, 
                                         y_coord_tolerance_for_horizontal_merge, 
                                         max_gap_for_horizontal_merge)
    x_coord_tolerance_for_vertical_merge = 10
    max_gap_for_vertical_merge = 25
    merged_v_lines = merge_line_segments(initial_v_lines, False,
                                         x_coord_tolerance_for_vertical_merge,
                                         max_gap_for_vertical_merge)
    merged_lines_image = original_map_image_for_drawing.copy()
    merged_lines_drawn = False
    if merged_h_lines:
        print(f"Number of merged horizontal lines: {len(merged_h_lines)}")
        for x1, y1, x2, y2 in merged_h_lines:
            cv2.line(merged_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green
            merged_lines_drawn = True
    if merged_v_lines:
        print(f"Number of merged vertical lines: {len(merged_v_lines)}")
        for x1, y1, x2, y2 in merged_v_lines:
            cv2.line(merged_lines_image, (x1, y1), (x2, y2), (255, 255, 0), 2) # Cyan
            merged_lines_drawn = True
    if merged_lines_drawn:
        cv2.imshow("Merged Grid Lines (Green H, Cyan V)", merged_lines_image)
        cv2.waitKey(0)
    else:
        print("No lines were merged or no initial H/V lines to merge.")
        

    # --- Find Intersections and Calculate Scale --- 
    intersection_points = []
    if merged_h_lines and merged_v_lines:
        for h_line in merged_h_lines:
            x1h, yh, x2h, _ = h_line # y1h is same as y2h
            for v_line in merged_v_lines:
                xv, y1v, _, y2v = v_line # x1v is same as x2v
                
                # Check for intersection
                if (min(x1h, x2h) <= xv <= max(x1h, x2h) and 
                    min(y1v, y2v) <= yh <= max(y1v, y2v)):
                    intersection_points.append((int(round(xv)), int(round(yh))))

    intersections_image = original_map_image_for_drawing.copy()
    if intersection_points:
        print(f"Found {len(intersection_points)} intersection points.")
        for pt in intersection_points:
            cv2.circle(intersections_image, pt, 5, (0, 0, 255), -1) # Draw red circles at intersections
        cv2.imshow("Intersection Points", intersections_image)
        cv2.waitKey(0)
    else:
        print("No intersection points found between merged H and V lines.")

    # Calculate average spacing
    pixel_distances_h = []
    if len(merged_h_lines) >= 2:
        # Get unique y-coordinates of horizontal lines, sorted
        unique_y_coords = sorted(list(set([line[1] for line in merged_h_lines])))
        for i in range(len(unique_y_coords) - 1):
            pixel_distances_h.append(unique_y_coords[i+1] - unique_y_coords[i])
        if pixel_distances_h:
            print(f"Horizontal line spacings (pixels): {pixel_distances_h}")

    pixel_distances_v = []
    if len(merged_v_lines) >= 2:
        # Get unique x-coordinates of vertical lines, sorted
        unique_x_coords = sorted(list(set([line[0] for line in merged_v_lines])))
        for i in range(len(unique_x_coords) - 1):
            pixel_distances_v.append(unique_x_coords[i+1] - unique_x_coords[i])
        if pixel_distances_v:
            print(f"Vertical line spacings (pixels): {pixel_distances_v}")

    all_spacings = pixel_distances_h + pixel_distances_v
    pixels_per_100m = 0
    
    min_expected_spacing = 30 
    max_expected_spacing = 300 
    min_samples_for_median = 2 # Need at least 2 spacings to compute a meaningful median/average

    robust_avg_h_spacing = 0
    if len(pixel_distances_h) >= min_samples_for_median:
        filtered_h_spacings = [s for s in pixel_distances_h if min_expected_spacing <= s <= max_expected_spacing]
        if len(filtered_h_spacings) >= min_samples_for_median:
            robust_avg_h_spacing = np.median(filtered_h_spacings)
            print(f"Robust average H spacing: {robust_avg_h_spacing:.2f} (from {filtered_h_spacings})")
        else:
            print("Not enough valid horizontal spacings for robust average.")

    robust_avg_v_spacing = 0
    if len(pixel_distances_v) >= min_samples_for_median:
        filtered_v_spacings = [s for s in pixel_distances_v if min_expected_spacing <= s <= max_expected_spacing]
        if len(filtered_v_spacings) >= min_samples_for_median:
            robust_avg_v_spacing = np.median(filtered_v_spacings)
            print(f"Robust average V spacing: {robust_avg_v_spacing:.2f} (from {filtered_v_spacings})")
        else:
            print("Not enough valid vertical spacings for robust average.")

    valid_spacings_found = 0
    total_spacing_sum = 0
    if robust_avg_h_spacing > 0:
        total_spacing_sum += robust_avg_h_spacing
        valid_spacings_found += 1
    if robust_avg_v_spacing > 0:
        total_spacing_sum += robust_avg_v_spacing
        valid_spacings_found += 1

    if valid_spacings_found > 0:
        pixels_per_100m = total_spacing_sum / valid_spacings_found
        print(f"Calculated average pixels_per_100m (from H/V medians): {pixels_per_100m:.2f}")
    else:
        print("No robust H or V spacings found, cannot calculate overall scale.")
        # pixels_per_100m remains 0

    # The rest of the function (drawing verification line, returning scale) uses this pixels_per_100m
    
    if pixels_per_100m > 0:
        scale_meters_per_pixel = 100.0 / pixels_per_100m
        print(f"Final calculated scale: {scale_meters_per_pixel:.4f} meters/pixel")
        # --- Draw a reference 100m line on the map for visual verification ---
        verification_image = original_map_image_for_drawing.copy()
        # Define a starting point for the reference line (e.g., 10% from top-left)
        img_h, img_w = verification_image.shape[:2]
        start_x = int(img_w * 0.1) - 72
        start_y = int(img_h * 0.1) - 50
        
        # Ensure the line end points are within image bounds
        # Horizontal line
        end_x_h = min(start_x + int(round(pixels_per_100m)), img_w - 1)
        cv2.line(verification_image, (start_x, start_y), (end_x_h, start_y), (0, 255, 255), 3) # Yellow
        # Vertical line
        end_y_v = min(start_y + int(round(pixels_per_100m)), img_h - 1)
        cv2.line(verification_image, (start_x, start_y), (start_x, end_y_v), (0, 255, 255), 3) # Yellow
        
        cv2.putText(verification_image, f"100m ref (calc: {pixels_per_100m:.0f}px)", (start_x + 5, start_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("100m Reference Line Verification", verification_image)
        cv2.waitKey(0)
        # --- End of reference line drawing ---

        return scale_meters_per_pixel
    else:
        print("Could not determine a valid pixels_per_100m scale.")
        return None # Or return the dummy scale if you want to proceed with player icon detection

    # REMOVE DUMMY Placeholder for scale calculation
    # print("Placeholder: Grid detection and scale calculation logic to be implemented here using MERGED lines.")
    # pixels_per_100m = 100 
    # if pixels_per_100m == 0:
    #     print("Error: Could not determine scale from grid lines.")
    #     return None
    # scale_meters_per_pixel = 100.0 / pixels_per_100m
    # return scale_meters_per_pixel

def check_intersection_pattern(window, brightness_threshold):
    """
    Checks if a 4x4 window matches the intersection pattern.
    Pattern: The 2x2 pixels at the center of the 4x4 are all brighter 
             than the average of the 4 corner pixels of the 4x4.
    Window indexing for 4x4:
    P00 P01 P02 P03
    P10 P11 P12 P13  (Center 2x2: P11, P12)
    P20 P21 P22 P23  (Center 2x2: P21, P22)
    P30 P31 P32 P33
    Corners: P00, P03, P30, P33
    """
    if window.shape != (4,4):
        return False
        
    # Corner pixels
    p00, p03 = window[0,0], window[0,3]
    p30, p33 = window[3,0], window[3,3]
    avg_corners = (int(p00) + int(p03) + int(p30) + int(p33)) / 4.0

    # Center 2x2 pixels
    p11, p12 = window[1,1], window[1,2]
    p21, p22 = window[2,1], window[2,2]

    # Condition: All 4 center pixels are brighter than avg_corners by brightness_threshold
    if (p11 > avg_corners + brightness_threshold and
        p12 > avg_corners + brightness_threshold and
        p21 > avg_corners + brightness_threshold and
        p22 > avg_corners + brightness_threshold):
        return True
    return False

def detect_intersections_by_pattern(map_image_cv, original_map_image_for_drawing):
    """Detects grid intersections by looking for a local 4x4 pixel pattern."""
    if map_image_cv is None:
        print("Error: Map image not loaded.")
        return None

    gray_map = cv2.cvtColor(map_image_cv, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Map for Pattern Detection", gray_map)
    cv2.waitKey(0)

    intersection_points = []
    height, width = gray_map.shape
    brightness_delta = 10 # How much brighter the 2x2 center should be than corner average

    # Iterate, leaving a 3-pixel border on right/bottom for 4x4 window
    # (x,y) will be the top-left corner of the 4x4 window
    for y in range(height - 3): # e.g. if height is 100, y goes 0..96. y+3 is 99 (last row index)
        for x in range(width - 3):  # e.g. if width is 100, x goes 0..96. x+3 is 99 (last col index)
            window = gray_map[y:y+4, x:x+4] # Extract 4x4 window
            if check_intersection_pattern(window, brightness_delta):
                # Report center of the 2x2 bright area, relative to map coords
                center_of_pattern_x = x + 1 
                center_of_pattern_y = y + 1 
                # Or, to be more precise for a 2x2 block, the point between the pixels, 
                # but cv2.circle needs integers. (x+1, y+1) is top-left of 2x2. 
                # Let's use (x+1, y+1) as the representative point for now.
                # A better visual might be (x+1.5, y+1.5) if we could draw subpixel, 
                # so (x+1, y+1) or (x+2,y+2) are approximations. We draw at (x+1, y+1)
                intersection_points.append((center_of_pattern_x, center_of_pattern_y)) 

    intersections_display_image = original_map_image_for_drawing.copy()
    if intersection_points:
        print(f"Found {len(intersection_points)} potential intersection patterns.")
        for pt in intersection_points:
            cv2.circle(intersections_display_image, pt, 5, (0, 0, 255), 1) # Red circles, thickness 1
        cv2.imshow("Detected Intersection Patterns", intersections_display_image)
        cv2.waitKey(0)
    else:
        print("No intersection patterns found.")

    # --- Reconstruct Grid Lines from Clustered Intersection Points ---
    reconstructed_h_lines = []
    reconstructed_v_lines = []
    
    if intersection_points:
        y_cluster_tolerance = 1
        x_cluster_tolerance = 1
        min_points_for_line = 50 # As per user suggestion

        # Process for horizontal lines (group by Y)
        # Sort points by Y primarily to help with grouping
        sorted_points_for_h = sorted(intersection_points, key=lambda p: p[1])
        
        visited_for_h = [False] * len(sorted_points_for_h)
        for i in range(len(sorted_points_for_h)):
            if visited_for_h[i]:
                continue
            current_y_group = [sorted_points_for_h[i]]
            visited_for_h[i] = True
            for j in range(i + 1, len(sorted_points_for_h)):
                if visited_for_h[j]:
                    continue
                # If next point is close in Y to the *first* point of this group
                if abs(sorted_points_for_h[j][1] - sorted_points_for_h[i][1]) <= y_cluster_tolerance:
                    current_y_group.append(sorted_points_for_h[j])
                    visited_for_h[j] = True
            
            if len(current_y_group) >= min_points_for_line:
                avg_y = int(round(np.mean([p[1] for p in current_y_group])))
                min_x = min([p[0] for p in current_y_group])
                max_x = max([p[0] for p in current_y_group])
                if max_x > min_x: # Ensure it has some length
                    reconstructed_h_lines.append((min_x, avg_y, max_x, avg_y))

        # Process for vertical lines (group by X)
        sorted_points_for_v = sorted(intersection_points, key=lambda p: p[0])
        visited_for_v = [False] * len(sorted_points_for_v)
        for i in range(len(sorted_points_for_v)):
            if visited_for_v[i]:
                continue
            current_x_group = [sorted_points_for_v[i]]
            visited_for_v[i] = True
            for j in range(i + 1, len(sorted_points_for_v)):
                if visited_for_v[j]:
                    continue
                if abs(sorted_points_for_v[j][0] - sorted_points_for_v[i][0]) <= x_cluster_tolerance:
                    current_x_group.append(sorted_points_for_v[j])
                    visited_for_v[j] = True
            
            if len(current_x_group) >= min_points_for_line:
                avg_x = int(round(np.mean([p[0] for p in current_x_group])))
                min_y = min([p[1] for p in current_x_group])
                max_y = max([p[1] for p in current_x_group])
                if max_y > min_y: # Ensure it has some length
                    reconstructed_v_lines.append((avg_x, min_y, avg_x, max_y))

    reconstructed_lines_image = original_map_image_for_drawing.copy()
    drawn_reconstructed = False
    if reconstructed_h_lines:
        print(f"Reconstructed {len(reconstructed_h_lines)} horizontal lines from point clusters.")
        for x1, y1, x2, y2 in reconstructed_h_lines:
            cv2.line(reconstructed_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green
            drawn_reconstructed = True
    if reconstructed_v_lines:
        print(f"Reconstructed {len(reconstructed_v_lines)} vertical lines from point clusters.")
        for x1, y1, x2, y2 in reconstructed_v_lines:
            cv2.line(reconstructed_lines_image, (x1, y1), (x2, y2), (255, 255, 0), 2) # Cyan
            drawn_reconstructed = True
    
    if drawn_reconstructed:
        cv2.imshow("Reconstructed Grid Lines (from Patterns)", reconstructed_lines_image)
        cv2.waitKey(0)
    else:
        print("Could not reconstruct significant H or V lines from intersection patterns.")

    # --- Calculate Scale using these reconstructed lines ---
    pixels_per_100m = 0
    min_expected_spacing = 30 
    max_expected_spacing = 300 
    min_samples_for_median = 2 

    robust_avg_h_spacing = 0
    if len(reconstructed_h_lines) >= min_samples_for_median: # Check reconstructed lines
        unique_y_coords = sorted(list(set([line[1] for line in reconstructed_h_lines])))
        pixel_distances_h = []
        if len(unique_y_coords) >= 2:
            for i in range(len(unique_y_coords) - 1):
                pixel_distances_h.append(unique_y_coords[i+1] - unique_y_coords[i])
        
        if pixel_distances_h: # Check if any distances were actually computed
            filtered_h_spacings = [s for s in pixel_distances_h if min_expected_spacing <= s <= max_expected_spacing]
            if len(filtered_h_spacings) >= min_samples_for_median:
                robust_avg_h_spacing = np.median(filtered_h_spacings)
                print(f"Robust average H spacing (from reconstructed): {robust_avg_h_spacing:.2f} (from {filtered_h_spacings})")
            # else: print("Not enough valid H spacings after filtering reconstructed lines.")
        # else: print("Not enough unique Ys for H spacing from reconstructed lines.")
    # else: print("Not enough reconstructed H lines for spacing.")

    robust_avg_v_spacing = 0
    if len(reconstructed_v_lines) >= min_samples_for_median: # Check reconstructed lines
        unique_x_coords = sorted(list(set([line[0] for line in reconstructed_v_lines])))
        pixel_distances_v = []
        if len(unique_x_coords) >= 2:
            for i in range(len(unique_x_coords) - 1):
                pixel_distances_v.append(unique_x_coords[i+1] - unique_x_coords[i])

        if pixel_distances_v: # Check if any distances were actually computed
            filtered_v_spacings = [s for s in pixel_distances_v if min_expected_spacing <= s <= max_expected_spacing]
            if len(filtered_v_spacings) >= min_samples_for_median:
                robust_avg_v_spacing = np.median(filtered_v_spacings)
                print(f"Robust average V spacing (from reconstructed): {robust_avg_v_spacing:.2f} (from {filtered_v_spacings})")
            # else: print("Not enough valid V spacings after filtering reconstructed lines.")
        # else: print("Not enough unique Xs for V spacing from reconstructed lines.")
    # else: print("Not enough reconstructed V lines for spacing.")

    valid_spacings_found = 0
    total_spacing_sum = 0
    if robust_avg_h_spacing > 0:
        total_spacing_sum += robust_avg_h_spacing
        valid_spacings_found += 1
    if robust_avg_v_spacing > 0:
        total_spacing_sum += robust_avg_v_spacing
        valid_spacings_found += 1

    if valid_spacings_found > 0:
        pixels_per_100m = total_spacing_sum / valid_spacings_found
        print(f"Calculated average pixels_per_100m (from reconstructed lines): {pixels_per_100m:.2f}")
    else:
        print("No robust H or V spacings found, cannot calculate overall scale.")
        # pixels_per_100m remains 0

    # The rest of the function (drawing verification line, returning scale) uses this pixels_per_100m
    
    if pixels_per_100m > 0:
        scale_meters_per_pixel = 100.0 / pixels_per_100m
        print(f"Final calculated scale: {scale_meters_per_pixel:.4f} meters/pixel")
        # --- Draw a reference 100m line on the map for visual verification ---
        verification_image = original_map_image_for_drawing.copy()
        # Define a starting point for the reference line (e.g., 10% from top-left)
        img_h, img_w = verification_image.shape[:2]
        start_x = int(img_w * 0.1) - 72
        start_y = int(img_h * 0.1) - 50
        
        # Ensure the line end points are within image bounds
        # Horizontal line
        end_x_h = min(start_x + int(round(pixels_per_100m)), img_w - 1)
        cv2.line(verification_image, (start_x, start_y), (end_x_h, start_y), (0, 255, 255), 3) # Yellow
        # Vertical line
        end_y_v = min(start_y + int(round(pixels_per_100m)), img_h - 1)
        cv2.line(verification_image, (start_x, start_y), (start_x, end_y_v), (0, 255, 255), 3) # Yellow
        
        cv2.putText(verification_image, f"100m ref (calc: {pixels_per_100m:.0f}px)", (start_x + 5, start_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("100m Reference Line Verification", verification_image)
        cv2.waitKey(0)
        # --- End of reference line drawing ---

        return scale_meters_per_pixel
    else:
        print("Could not determine a valid pixels_per_100m scale.")
        return None

if __name__ == "__main__":
    map_image_path = "map-lines.png"
    original_map_image = load_image(map_image_path)
    
    if original_map_image is not None:
        scale = detect_intersections_by_pattern(original_map_image.copy(), 
                                                original_map_image)
        if scale is not None:
            print(f"Calculated scale: {scale:.4f} meters/pixel")
        else:
            print("Could not calculate scale from the image.")
            
        cv2.destroyAllWindows()
    else:
        print(f"Could not load map image: {map_image_path}") 