import cv2
import numpy as np

def load_image(image_path):
    """Loads an image from the specified path."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        print("Please ensure the image path is correct and the file exists.")
    return img

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

def detect_object_template_matching(main_image_bgr, template_path, object_name, threshold=0.7):
    """
    Detects an object in the main image using template matching.
    Draws a rectangle around the detected object and returns its center coordinates.
    """
    print(f"Attempting to detect {object_name} using template: {template_path}")
    template_bgr = cv2.imread(template_path)
    if template_bgr is None:
        print(f"Error: Could not load template image from {template_path}")
        return None, None

    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    main_gray = cv2.cvtColor(main_image_bgr, cv2.COLOR_BGR2GRAY)
    
    w, h = template_gray.shape[::-1] # width, height of template

    # Apply template Matching
    # TM_CCOEFF_NORMED is good for handling variations in lighting/color to some extent
    res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    print(f"Max match value for {object_name}: {max_val:.2f} (Threshold: {threshold})")

    detection_image = main_image_bgr.copy()
    center_coords = None

    if max_val >= threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(detection_image, top_left, bottom_right, (0, 255, 0), 2) # Green rectangle
        cv2.putText(detection_image, object_name, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2
        center_coords = (center_x, center_y)
        print(f"{object_name} detected at {top_left}, center: {center_coords}")
        cv2.imshow(f"Detected {object_name}", detection_image)
        cv2.waitKey(0)
    else:
        print(f"{object_name} not found with sufficient confidence.")
        cv2.imshow(f"No {object_name} Detected (or below threshold)", detection_image) # Show image even if not found
        cv2.waitKey(0)
        
    return center_coords, detection_image # Return image for further drawing if needed

def calculate_and_display_distance(image_to_draw_on, p1_coords, p2_coords, scale_meters_per_pixel, 
                                   p1_name="Player", p2_name="Ping"):
    """
    Calculates the distance between two points in meters and displays it on the image.
    """
    if p1_coords is None or p2_coords is None or scale_meters_per_pixel is None or scale_meters_per_pixel == 0:
        print("Error: Missing coordinates or scale for distance calculation.")
        return

    # Calculate pixel distance
    dx = p1_coords[0] - p2_coords[0]
    dy = p1_coords[1] - p2_coords[1]
    pixel_distance = np.sqrt(dx**2 + dy**2)

    # Convert to meters
    real_world_distance_m = pixel_distance * scale_meters_per_pixel

    print(f"Pixel distance between {p1_name} and {p2_name}: {pixel_distance:.2f} pixels.")
    print(f"Real-world distance: {real_world_distance_m:.2f} meters.")

    # Draw line and text on the image
    cv2.line(image_to_draw_on, p1_coords, p2_coords, (255, 0, 0), 2) # Blue line

    # Position text midway along the line
    text_x = (p1_coords[0] + p2_coords[0]) // 2
    text_y = (p1_coords[1] + p2_coords[1]) // 2
    
    # Add offsets to avoid drawing text directly on the line
    text_offset_x = 10
    text_offset_y = -10

    cv2.putText(image_to_draw_on, f"{real_world_distance_m:.1f} m", 
                (text_x + text_offset_x, text_y + text_offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA) # Black outline
    cv2.putText(image_to_draw_on, f"{real_world_distance_m:.1f} m", 
                (text_x + text_offset_x, text_y + text_offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA) # White text
    
    cv2.imshow("Map with Distance", image_to_draw_on)
    cv2.waitKey(0)

def check_horizontal_line_pattern_4x4(window, brightness_delta):
    """
    Checks if a 4x4 window matches the horizontal line pattern:
    Top and bottom rows are all darker than the average of the center 2x4 rows.
    """
    if window.shape != (4, 4):
        return False

    center_pixels_h = window[1:3, :]  # Rows 1 and 2 (0-indexed), all columns
    if center_pixels_h.size == 0: return False # Should not happen with 4x4 window
    avg_center_h = np.mean(center_pixels_h)

    # Check top row
    for j in range(4):
        if not (window[0, j] < avg_center_h - brightness_delta):
            return False
    # Check bottom row
    for j in range(4):
        if not (window[3, j] < avg_center_h - brightness_delta):
            return False
    return True

def check_vertical_line_pattern_4x4(window, brightness_delta):
    """
    Checks if a 4x4 window matches the vertical line pattern:
    Left and right columns are all darker than the average of the center 4x2 columns.
    """
    if window.shape != (4, 4):
        return False

    center_pixels_v = window[:, 1:3]  # All rows, columns 1 and 2 (0-indexed)
    if center_pixels_v.size == 0: return False
    avg_center_v = np.mean(center_pixels_v)

    # Check left column
    for i in range(4):
        if not (window[i, 0] < avg_center_v - brightness_delta):
            return False
    # Check right column
    for i in range(4):
        if not (window[i, 3] < avg_center_v - brightness_delta):
            return False
    return True

def detect_grid_lines_via_local_patterns_and_calculate_scale(map_image_cv, original_map_image_for_drawing, min_points_for_line_threshold=200):
    """
    Detects grid lines by looking for local 4x4 horizontal/vertical line patterns,
    reconstructs lines, and calculates the pixel-to-meter scale.
    """
    if map_image_cv is None:
        print("Error: Map image not loaded for pattern-based line detection.")
        return None

    gray_map = cv2.cvtColor(map_image_cv, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Map for New Pattern Detection", gray_map)
    cv2.waitKey(0)

    horizontal_pattern_windows = [] # Stores (x,y) of top-left of 4x4 window matching H pattern
    vertical_pattern_windows = []   # Stores (x,y) of top-left of 4x4 window matching V pattern
    
    brightness_delta = 5 # REDUCED from 10 to increase sensitivity
    height, width = gray_map.shape

    for y in range(height - 3): # Window top-left y
        for x in range(width - 3):  # Window top-left x
            window = gray_map[y:y+4, x:x+4]
            
            if check_horizontal_line_pattern_4x4(window, brightness_delta):
                horizontal_pattern_windows.append((x, y))

            if check_vertical_line_pattern_4x4(window, brightness_delta):
                vertical_pattern_windows.append((x, y))

    print(f"Found {len(horizontal_pattern_windows)} H-pattern windows, {len(vertical_pattern_windows)} V-pattern windows.")

    # Optional: Visualize detected pattern hits
    pattern_hits_image = original_map_image_for_drawing.copy()
    for (x,y) in horizontal_pattern_windows: # Draw a small mark at window's top-left or center
        cv2.circle(pattern_hits_image, (x+1, y+1), 1, (0, 255, 255), -1) # Yellow for H
    for (x,y) in vertical_pattern_windows:
        cv2.circle(pattern_hits_image, (x+1, y+1), 1, (255, 0, 255), -1) # Pink for V
    cv2.imshow("Detected H/V Pattern Window Locations", pattern_hits_image)
    cv2.waitKey(0)


    # --- Reconstruct Grid Lines from Clustered Pattern Window Locations ---
    reconstructed_h_lines = []
    reconstructed_v_lines = []
    
    y_cluster_tolerance = 2 # Allow a bit more play for y-coords of horizontal lines
    x_cluster_tolerance = 2 # Allow a bit more play for x-coords of vertical lines
    
    # Process for horizontal lines
    if horizontal_pattern_windows:
        # Sort by y-coordinate of window to help grouping
        sorted_h_windows = sorted(horizontal_pattern_windows, key=lambda p: p[1])
        visited_h = [False] * len(sorted_h_windows)

        for i in range(len(sorted_h_windows)):
            if visited_h[i]:
                continue
            
            current_y_group_windows = [sorted_h_windows[i]]
            visited_h[i] = True
            base_y_for_group = sorted_h_windows[i][1] # y of the window's top-left

            for j in range(i + 1, len(sorted_h_windows)):
                if visited_h[j]:
                    continue
                if abs(sorted_h_windows[j][1] - base_y_for_group) <= y_cluster_tolerance:
                    current_y_group_windows.append(sorted_h_windows[j])
                    visited_h[j] = True
            
            if len(current_y_group_windows) >= min_points_for_line_threshold:
                # The actual horizontal line is centered at y_window + 1.5
                avg_line_y = int(round(np.mean([p[1] for p in current_y_group_windows]) + 1.5))
                
                # The x-extent of the line is from the leftmost x of a window to the rightmost x+3 of a window
                min_x_coord = min([p[0] for p in current_y_group_windows])
                max_x_coord = max([p[0] + 3 for p in current_y_group_windows]) # Window is 4px wide
                
                if max_x_coord > min_x_coord: # Ensure it has some length
                    reconstructed_h_lines.append((min_x_coord, avg_line_y, max_x_coord, avg_line_y))

    # Process for vertical lines
    if vertical_pattern_windows:
        # Sort by x-coordinate of window
        sorted_v_windows = sorted(vertical_pattern_windows, key=lambda p: p[0])
        visited_v = [False] * len(sorted_v_windows)

        for i in range(len(sorted_v_windows)):
            if visited_v[i]:
                continue

            current_x_group_windows = [sorted_v_windows[i]]
            visited_v[i] = True
            base_x_for_group = sorted_v_windows[i][0] # x of the window's top-left

            for j in range(i + 1, len(sorted_v_windows)):
                if visited_v[j]:
                    continue
                if abs(sorted_v_windows[j][0] - base_x_for_group) <= x_cluster_tolerance:
                    current_x_group_windows.append(sorted_v_windows[j])
                    visited_v[j] = True
            
            if len(current_x_group_windows) >= min_points_for_line_threshold:
                # The actual vertical line is centered at x_window + 1.5
                avg_line_x = int(round(np.mean([p[0] for p in current_x_group_windows]) + 1.5))
                
                min_y_coord = min([p[1] for p in current_x_group_windows])
                max_y_coord = max([p[1] + 3 for p in current_x_group_windows]) # Window is 4px tall
                
                if max_y_coord > min_y_coord:
                    reconstructed_v_lines.append((avg_line_x, min_y_coord, avg_line_x, max_y_coord))

    reconstructed_lines_image = original_map_image_for_drawing.copy()
    drawn_reconstructed = False
    if reconstructed_h_lines:
        print(f"Reconstructed {len(reconstructed_h_lines)} horizontal lines from new patterns.")
        for x1, y1, x2, y2 in reconstructed_h_lines:
            cv2.line(reconstructed_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green
            drawn_reconstructed = True
    if reconstructed_v_lines:
        print(f"Reconstructed {len(reconstructed_v_lines)} vertical lines from new patterns.")
        for x1, y1, x2, y2 in reconstructed_v_lines:
            cv2.line(reconstructed_lines_image, (x1, y1), (x2, y2), (255, 255, 0), 2) # Cyan
            drawn_reconstructed = True
    
    if drawn_reconstructed:
        cv2.imshow("Reconstructed Grid Lines (New Patterns)", reconstructed_lines_image)
        cv2.waitKey(0)
    else:
        print("Could not reconstruct significant H or V lines from new patterns.")

    # --- Calculate Scale using these reconstructed lines (copied from detect_intersections_by_pattern) ---
    pixels_per_100m = 0
    min_expected_spacing = 30 
    max_expected_spacing = 300 
    min_samples_for_median = 2 

    robust_avg_h_spacing = 0
    if len(reconstructed_h_lines) >= 2: # Need at least 2 lines for spacing
        unique_y_coords = sorted(list(set([line[1] for line in reconstructed_h_lines])))
        pixel_distances_h = []
        if len(unique_y_coords) >= 2:
            for i in range(len(unique_y_coords) - 1):
                pixel_distances_h.append(unique_y_coords[i+1] - unique_y_coords[i])
        
        if pixel_distances_h: 
            filtered_h_spacings = [s for s in pixel_distances_h if min_expected_spacing <= s <= max_expected_spacing]
            if len(filtered_h_spacings) >= min_samples_for_median: # Check after filtering
                robust_avg_h_spacing = np.median(filtered_h_spacings)
                print(f"Robust average H spacing (New Patterns): {robust_avg_h_spacing:.2f} (from {filtered_h_spacings})")
            else: print("Not enough valid H spacings after filtering (New Patterns).")
        else: print("Not enough unique Ys for H spacing (New Patterns).")
    else: print("Not enough reconstructed H lines for spacing (New Patterns).")

    robust_avg_v_spacing = 0
    if len(reconstructed_v_lines) >= 2: # Need at least 2 lines
        unique_x_coords = sorted(list(set([line[0] for line in reconstructed_v_lines])))
        pixel_distances_v = []
        if len(unique_x_coords) >= 2:
            for i in range(len(unique_x_coords) - 1):
                pixel_distances_v.append(unique_x_coords[i+1] - unique_x_coords[i])

        if pixel_distances_v:
            filtered_v_spacings = [s for s in pixel_distances_v if min_expected_spacing <= s <= max_expected_spacing]
            if len(filtered_v_spacings) >= min_samples_for_median: # Check after filtering
                robust_avg_v_spacing = np.median(filtered_v_spacings)
                print(f"Robust average V spacing (New Patterns): {robust_avg_v_spacing:.2f} (from {filtered_v_spacings})")
            else: print("Not enough valid V spacings after filtering (New Patterns).")
        else: print("Not enough unique Xs for V spacing (New Patterns).")
    else: print("Not enough reconstructed V lines for spacing (New Patterns).")

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
        print(f"Calculated average pixels_per_100m (New Patterns): {pixels_per_100m:.2f}")
    else:
        print("No robust H or V spacings found (New Patterns), cannot calculate overall scale.")
        pixels_per_100m = 0 # Ensure it's zero if no scale found
    
    if pixels_per_100m > 0:
        scale_meters_per_pixel = 100.0 / pixels_per_100m
        print(f"Final calculated scale (New Patterns): {scale_meters_per_pixel:.4f} meters/pixel")
        verification_image = original_map_image_for_drawing.copy()
        img_h, img_w = verification_image.shape[:2]
        start_x = int(img_w * 0.1) - 72
        start_y = int(img_h * 0.1) - 50
        end_x_h = min(start_x + int(round(pixels_per_100m)), img_w - 1)
        cv2.line(verification_image, (start_x, start_y), (end_x_h, start_y), (0, 255, 255), 3)
        end_y_v = min(start_y + int(round(pixels_per_100m)), img_h - 1)
        cv2.line(verification_image, (start_x, start_y), (start_x, end_y_v), (0, 255, 255), 3)
        cv2.putText(verification_image, f"100m ref (calc: {pixels_per_100m:.0f}px)", (start_x + 5, start_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("100m Reference Line Verification (New Patterns)", verification_image)
        cv2.waitKey(0)
        return scale_meters_per_pixel
    else:
        print("Could not determine a valid pixels_per_100m scale (New Patterns).")
        return None

if __name__ == "__main__":
    map_image_path = "map-test.png" # As per user's last change
    player_icon_template_path = "player-icon.png" 
    ping_marker_template_path = "player-ping.png" 
    template_matching_threshold = 0.7 
    current_min_points_for_line = 100 # DRASTICALLY REDUCED from 200 for testing

    original_map_image = load_image(map_image_path)
    
    if original_map_image is not None:
        map_for_scale_detection = original_map_image.copy()
        
        # Call the new function for scale detection
        scale_meters_per_pixel = detect_grid_lines_via_local_patterns_and_calculate_scale(
            map_for_scale_detection, 
            original_map_image.copy(), # Pass another copy for drawing within function
            min_points_for_line_threshold=current_min_points_for_line
        )
        
        if scale_meters_per_pixel is not None:
            print(f"Calculated scale using new patterns: {scale_meters_per_pixel:.4f} meters/pixel")

            player_coords, _ = detect_object_template_matching(original_map_image.copy(),
                                                               player_icon_template_path, 
                                                               "Player Icon", 
                                                               template_matching_threshold)
            
            ping_coords, _ = detect_object_template_matching(original_map_image.copy(),
                                                             ping_marker_template_path, 
                                                             "Ping Marker", 
                                                             template_matching_threshold)

            if player_coords and ping_coords:
                final_display_image = original_map_image.copy()
                calculate_and_display_distance(final_display_image, 
                                               player_coords, 
                                               ping_coords, 
                                               scale_meters_per_pixel)
            else:
                print("Could not detect both player icon and ping marker. Cannot calculate distance.")
        else:
            print("Could not calculate map scale using new patterns. Aborting further processing.")
            
        cv2.destroyAllWindows()
    else:
        print(f"Could not load map image: {map_image_path}") 