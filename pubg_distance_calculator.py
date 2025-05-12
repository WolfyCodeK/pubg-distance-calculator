import tkinter as tk
import tkinter.messagebox
import keyboard
import time

# Added imports for image processing
import cv2
import numpy as np
import mss
import mss.tools # For mss.exception.ScreenShotError if needed
from PIL import Image

# --- Global OSD Variables ---
osd_root = None
osd_label = None
osd_dismiss_key_handler = None # For the keyboard.on_press handler

# --- Global Image Processing & Template Variables ---
player_template_cv = None
ping_template_cv = None

# Tunable Parameters for image processing (can be adjusted here)
BRIGHTNESS_DELTA_GRID = 5
MIN_POINTS_FOR_LINE_GRID = 100
TEMPLATE_MATCHING_THRESHOLD_PLAYER = 0.7
TEMPLATE_MATCHING_THRESHOLD_PING = 0.7

# --- Screen Capture ---
def capture_screen():
    """Captures the primary monitor and returns it as an OpenCV BGR image."""
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            sct_img = sct.grab(monitor)
            img_pil = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb, 'raw', 'RGB')
            img_np = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            return img_bgr
    except mss.exception.ScreenShotError as e_mss:
        print(f"MSS ScreenShotError: {e_mss}. This can happen if the screen is locked or changing resolution.")
        return None
    except Exception as e:
        print(f"General error during screen capture: {e}")
        return None

# --- OSD Functions (from minimal test - largely unchanged) ---
def setup_osd():
    global osd_root, osd_label
    if osd_root is not None: return
    osd_root = tk.Tk()
    osd_root.withdraw()
    osd_root.overrideredirect(True)
    osd_root.wm_attributes("-topmost", True)
    osd_root.attributes('-alpha', 0.85)
    osd_label = tk.Label(osd_root, text="", font=("Arial", 22, "bold"),
                         bg="black", fg="white", wraplength=600, justify="center")
    osd_label.pack(padx=15, pady=10)
    print("OSD Setup: Tkinter <Any-KeyPress> binding REMOVED. Dismissal will be handled by 'keyboard' library.")

def show_osd_message(message_to_display):
    global osd_root, osd_label, osd_dismiss_key_handler # Added osd_dismiss_key_handler
    if not osd_root or not osd_label: 
        print("OSD Error: Attempted to show message, but OSD not initialized.")
        return
    
    osd_label.config(text=message_to_display)
    
    # Calculate position (e.g., top-center)
    screen_width = osd_root.winfo_screenwidth()
    osd_root.update_idletasks()  # Allow Tkinter to calculate window size based on new text
    window_width = osd_root.winfo_width()
    
    x_pos = (screen_width // 2) - (window_width // 2)
    y_pos = 70  # Pixels from the top of the screen

    osd_root.geometry(f"+{x_pos}+{y_pos}")
    
    # Make the OSD window visible and on top
    osd_root.deiconify()
    osd_root.lift()
    osd_root.wm_attributes("-topmost", True)
    print("OSD: Window deiconified, lifted, and set topmost.")

    # Unhook any previous dismiss handler before registering a new one
    if osd_dismiss_key_handler:
        try:
            keyboard.unhook(osd_dismiss_key_handler)
            print("OSD Show: Unhooked previous 'any key' OSD dismiss listener.")
        except KeyError: # keyboard lib raises KeyError if hook not found
            print("OSD Show: Previous 'any key' listener already unhooked or not found (KeyError).")
        except Exception as e:
            print(f"OSD Show: Error unhooking previous 'any key' listener: {e}")
        osd_dismiss_key_handler = None

    def on_any_key_for_osd_dismiss(event):
        # This callback is active only when OSD is supposed to be visible.
        # It relies on hide_osd() to unhook it.
        if not osd_root or not osd_root.winfo_viewable():
            print(f"OSD Dismiss (keyboard lib): '{event.name}' pressed, but OSD not viewable. Attempting to hide/unhook via hide_osd().")
            hide_osd() # Attempt cleanup
            return

        print(f"OSD Dismiss (keyboard lib): Key '{event.name}' pressed. Hiding OSD.")
        hide_osd()
        # suppress=False is used, so no need to return True/False to block event.

    # Register the new listener
    osd_dismiss_key_handler = keyboard.on_press(on_any_key_for_osd_dismiss, suppress=False)
    print("OSD Show: Registered 'any key' listener for OSD dismissal using 'keyboard' library (suppress=False).")

def hide_osd():
    """Hides the OSD window."""
    global osd_root, osd_dismiss_key_handler # Added osd_dismiss_key_handler
    if osd_root:
        print("OSD: Hiding window via hide_osd().")
        osd_root.withdraw()

    # Unhook the 'any key' OSD dismiss listener when OSD is hidden
    if osd_dismiss_key_handler:
        try:
            print("OSD Hide: Unhooking 'any key' OSD dismiss listener.")
            keyboard.unhook(osd_dismiss_key_handler)
            print("OSD Hide: Listener unhooked successfully.")
        except KeyError: # keyboard lib raises KeyError if hook not found
             print("OSD Hide: Listener already unhooked or not found (KeyError).")
        except Exception as e:
            print(f"OSD Hide: Error unhooking 'any key' listener: {e}")
        finally:
            osd_dismiss_key_handler = None # Ensure it's cleared

# --- Core Image Processing Functions (Re-integrated, no cv2.imshow) ---
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None: print(f"Error loading template: {image_path}")
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
            # Compare next_segment to the *average* primary coord of the current merged group
            avg_primary_coord_of_current_merged_group = primary_coords_sum / count_in_merge
            primary_coord_next = next_segment[1] if is_horizontal else next_segment[0]
            if abs(primary_coord_next - avg_primary_coord_of_current_merged_group) > primary_axis_tolerance:
                continue # Not on the same primary axis level

            # Secondary axis check (do they overlap or are they close enough to bridge a gap?)
            # Horizontal: x-coords; Vertical: y-coords
            s1_current_start = current_segment[0] if is_horizontal else current_segment[1]
            s1_current_end = current_segment[2] if is_horizontal else current_segment[3]
            s2_next_start = next_segment[0] if is_horizontal else next_segment[1]
            s2_next_end = next_segment[2] if is_horizontal else next_segment[3]

            # Check for overlap or if gap is bridgeable
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

def detect_grid_lines_via_local_patterns_and_calculate_scale(map_image_cv, brightness_delta_param, min_points_for_line_thr):
    """
    Detects grid lines by looking for local 4x4 horizontal/vertical line patterns,
    reconstructs lines, and calculates the pixel-to-meter scale.
    This version is adapted from test.py for use in the main script (no cv2.imshow).
    """
    if map_image_cv is None:
        print("Scale Error: Map image missing for pattern-based line detection.")
        return None

    gray_map = cv2.cvtColor(map_image_cv, cv2.COLOR_BGR2GRAY)
    # No cv2.imshow here

    horizontal_pattern_windows = [] 
    vertical_pattern_windows = []   
    
    # Use the passed brightness_delta_param instead of a hardcoded value
    # brightness_delta = 5 # This was from test.py, now using param
    height, width = gray_map.shape

    for y_scan in range(height - 3): 
        for x_scan in range(width - 3):  
            window = gray_map[y_scan:y_scan+4, x_scan:x_scan+4]
            
            if check_horizontal_line_pattern_4x4(window, brightness_delta_param):
                horizontal_pattern_windows.append((x_scan, y_scan))

            if check_vertical_line_pattern_4x4(window, brightness_delta_param):
                vertical_pattern_windows.append((x_scan, y_scan))

    print(f"Scale Info: Found {len(horizontal_pattern_windows)} H-pattern windows, {len(vertical_pattern_windows)} V-pattern windows.")

    # No visualization of pattern hits (cv2.imshow removed)

    reconstructed_h_lines = []
    reconstructed_v_lines = []
    
    y_cluster_tolerance = 2 
    x_cluster_tolerance = 2 
    
    if horizontal_pattern_windows:
        sorted_h_windows = sorted(horizontal_pattern_windows, key=lambda p: p[1])
        visited_h = [False] * len(sorted_h_windows)
        for i in range(len(sorted_h_windows)):
            if visited_h[i]: continue
            current_y_group_windows = [sorted_h_windows[i]]; visited_h[i] = True
            base_y_for_group = sorted_h_windows[i][1]
            for j in range(i + 1, len(sorted_h_windows)):
                if visited_h[j]: continue
                if abs(sorted_h_windows[j][1] - base_y_for_group) <= y_cluster_tolerance:
                    current_y_group_windows.append(sorted_h_windows[j]); visited_h[j] = True
            if len(current_y_group_windows) >= min_points_for_line_thr: # Use passed parameter
                avg_line_y = int(round(np.mean([p[1] for p in current_y_group_windows]) + 1.5))
                min_x_coord = min([p[0] for p in current_y_group_windows])
                max_x_coord = max([p[0] + 3 for p in current_y_group_windows])
                if max_x_coord > min_x_coord: 
                    reconstructed_h_lines.append((min_x_coord, avg_line_y, max_x_coord, avg_line_y))

    if vertical_pattern_windows:
        sorted_v_windows = sorted(vertical_pattern_windows, key=lambda p: p[0])
        visited_v = [False] * len(sorted_v_windows)
        for i in range(len(sorted_v_windows)):
            if visited_v[i]: continue
            current_x_group_windows = [sorted_v_windows[i]]; visited_v[i] = True
            base_x_for_group = sorted_v_windows[i][0]
            for j in range(i + 1, len(sorted_v_windows)):
                if visited_v[j]: continue
                if abs(sorted_v_windows[j][0] - base_x_for_group) <= x_cluster_tolerance:
                    current_x_group_windows.append(sorted_v_windows[j]); visited_v[j] = True
            if len(current_x_group_windows) >= min_points_for_line_thr: # Use passed parameter
                avg_line_x = int(round(np.mean([p[0] for p in current_x_group_windows]) + 1.5))
                min_y_coord = min([p[1] for p in current_x_group_windows])
                max_y_coord = max([p[1] + 3 for p in current_x_group_windows])
                if max_y_coord > min_y_coord:
                    reconstructed_v_lines.append((avg_line_x, min_y_coord, avg_line_x, max_y_coord))

    # No visualization of reconstructed lines (cv2.imshow removed)
    if reconstructed_h_lines: print(f"Scale Info: Reconstructed {len(reconstructed_h_lines)} H-lines (New Patterns).")
    if reconstructed_v_lines: print(f"Scale Info: Reconstructed {len(reconstructed_v_lines)} V-lines (New Patterns).")
    if not reconstructed_h_lines and not reconstructed_v_lines:
        print("Scale Error: Could not reconstruct significant H or V lines (New Patterns).")
        return None # Added return based on previous logic if no lines

    pixels_per_100m = 0
    min_expected_spacing = 30 
    max_expected_spacing = 300 
    min_samples_for_median = 2 

    robust_avg_h_spacing = 0
    if len(reconstructed_h_lines) >= 2:
        unique_y_coords = sorted(list(set([line[1] for line in reconstructed_h_lines])))
        pixel_distances_h = []
        if len(unique_y_coords) >= 2:
            for i in range(len(unique_y_coords) - 1):
                pixel_distances_h.append(unique_y_coords[i+1] - unique_y_coords[i])
        if pixel_distances_h: 
            filtered_h_spacings = [s for s in pixel_distances_h if min_expected_spacing <= s <= max_expected_spacing]
            if len(filtered_h_spacings) >= min_samples_for_median:
                robust_avg_h_spacing = np.median(filtered_h_spacings)
                print(f"Scale Info: Robust average H spacing (New Patterns): {robust_avg_h_spacing:.2f} (from {filtered_h_spacings})")
            else: print("Scale Info: Not enough valid H spacings after filtering (New Patterns).")
        else: print("Scale Info: Not enough unique Ys for H spacing (New Patterns).")
    else: print("Scale Info: Not enough reconstructed H lines for spacing (New Patterns).")

    robust_avg_v_spacing = 0
    if len(reconstructed_v_lines) >= 2:
        unique_x_coords = sorted(list(set([line[0] for line in reconstructed_v_lines])))
        pixel_distances_v = []
        if len(unique_x_coords) >= 2:
            for i in range(len(unique_x_coords) - 1):
                pixel_distances_v.append(unique_x_coords[i+1] - unique_x_coords[i])
        if pixel_distances_v:
            filtered_v_spacings = [s for s in pixel_distances_v if min_expected_spacing <= s <= max_expected_spacing]
            if len(filtered_v_spacings) >= min_samples_for_median:
                robust_avg_v_spacing = np.median(filtered_v_spacings)
                print(f"Scale Info: Robust average V spacing (New Patterns): {robust_avg_v_spacing:.2f} (from {filtered_v_spacings})")
            else: print("Scale Info: Not enough valid V spacings after filtering (New Patterns).")
        else: print("Scale Info: Not enough unique Xs for V spacing (New Patterns).")
    else: print("Scale Info: Not enough reconstructed V lines for spacing (New Patterns).")

    valid_spacings_found = 0
    total_spacing_sum = 0
    if robust_avg_h_spacing > 0:
        total_spacing_sum += robust_avg_h_spacing; valid_spacings_found += 1
    if robust_avg_v_spacing > 0:
        total_spacing_sum += robust_avg_v_spacing; valid_spacings_found += 1

    if valid_spacings_found > 0:
        pixels_per_100m = total_spacing_sum / valid_spacings_found
        print(f"Scale Info: Calculated average pixels_per_100m (New Patterns): {pixels_per_100m:.2f}")
    else:
        print("Scale Error: No robust H or V spacings found (New Patterns), cannot calculate overall scale.")
        pixels_per_100m = 0 
    
    if pixels_per_100m > 0:
        scale_meters_per_pixel = 100.0 / pixels_per_100m
        print(f"Scale Info: Final calculated scale (New Patterns): {scale_meters_per_pixel:.4f} meters/pixel")
        # No verification image drawing (cv2.imshow removed)
        return scale_meters_per_pixel
    else:
        print("Scale Error: Could not determine a valid pixels_per_100m scale (New Patterns).")
        return None

def detect_object_template_matching(main_image_bgr, template_image_cv, object_name, threshold):
    if main_image_bgr is None or template_image_cv is None: print(f"Template Match Error: Missing image/template for {object_name}."); return None
    print(f"Template Match Info: Detecting {object_name}...")
    template_gray = cv2.cvtColor(template_image_cv, cv2.COLOR_BGR2GRAY) if len(template_image_cv.shape) == 3 else template_image_cv
    main_gray = cv2.cvtColor(main_image_bgr, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    print(f"Template Match Info: Max val for {object_name}: {max_val:.2f} (Threshold: {threshold})")
    if max_val >= threshold:
        center_coords = (max_loc[0] + w // 2, max_loc[1] + h // 2)
        print(f"Template Match Info: {object_name} found at center: {center_coords}"); return center_coords
    print(f"Template Match Warning: {object_name} not found (or below threshold)."); return None

def get_distance_string(p1_coords, p2_coords, scale_meters_per_pixel):
    if not all([p1_coords, p2_coords, scale_meters_per_pixel, scale_meters_per_pixel > 0]):
        print(f"Distance Error: Invalid inputs - P1:{p1_coords}, P2:{p2_coords}, Scale:{scale_meters_per_pixel}")
        return "Error: Calc data missing"
    dist_px = np.sqrt((p1_coords[0]-p2_coords[0])**2 + (p1_coords[1]-p2_coords[1])**2)
    dist_m = dist_px * scale_meters_per_pixel
    print(f"Distance Info: Pixel dist: {dist_px:.2f}, Real dist: {dist_m:.2f} m.")
    return f"{dist_m:.1f} m"

# --- Hotkey Triggered Action (Modified to include full analysis) ---
def on_hotkey_pressed():
    """This function is called when the hotkey is detected."""
    print(f"HOTKEY: Ctrl+Shift+Alt+D detected by keyboard library. Timestamp: {time.time():.2f}")
    print("HOTKEY: Processing screen analysis...")
    start_time = time.time()
    
    final_osd_message = ""

    # 1. Capture screen
    screen_image = capture_screen()
    if screen_image is None:
        final_osd_message = "Error: Screen capture failed."
    else:
        # 2. Calculate scale
        scale = detect_grid_lines_via_local_patterns_and_calculate_scale(
            screen_image, 
            BRIGHTNESS_DELTA_GRID, 
            MIN_POINTS_FOR_LINE_GRID
        )
        if scale is None:
            final_osd_message = "Error: Map scale not found."
        else:
            # 3. Detect Player Icon
            player_coords = detect_object_template_matching(
                screen_image, 
                player_template_cv, 
                "Player Icon", 
                TEMPLATE_MATCHING_THRESHOLD_PLAYER
            )
            if player_coords is None:
                final_osd_message = "Error: Player icon not found."
            else:
                # 4. Detect Ping Marker
                ping_coords = detect_object_template_matching(
                    screen_image, 
                    ping_template_cv, 
                    "Ping Marker", 
                    TEMPLATE_MATCHING_THRESHOLD_PING
                )
                if ping_coords is None:
                    final_osd_message = "Error: Ping marker not found."
                else:
                    # 5. Calculate distance string (Success case)
                    final_osd_message = get_distance_string(player_coords, ping_coords, scale)
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"HOTKEY: Analysis function complete ({processing_time:.2f}s). Message: '{final_osd_message}'. Scheduling OSD update.")

    # Schedule OSD update with the final message (either success or error)
    if osd_root:
        osd_root.after(0, show_osd_message, final_osd_message)
        print("HOTKEY: OSD update scheduled via osd_root.after().")
    else:
        print("HOTKEY: OSD root window not available. Cannot schedule OSD update.")

# --- Main Program Setup ---
if __name__ == "__main__":
    PLAYER_ICON_PATH = "player-icon.png"
    PING_MARKER_PATH = "player-ping.png"
    HOTKEY_STRING = "ctrl+shift+alt+d"

    print("PUBG Distance Calculator - Screen Version")
    print("Loading templates...")
    player_template_cv = load_image(PLAYER_ICON_PATH)
    ping_template_cv = load_image(PING_MARKER_PATH)

    if player_template_cv is None or ping_template_cv is None:
        err_msg = f"Fatal Error: Could not load template files. Check '{PLAYER_ICON_PATH}' and '{PING_MARKER_PATH}'."
        print(err_msg)
        try: tkinter.messagebox.showerror("Template Load Error", err_msg)
        except tk.TclError: print("(Tkinter not available for error dialog)")
        exit(1)
    print("Templates loaded successfully.")

    print("Initializing OSD system...")
    setup_osd()
    if not osd_root:
        print("Fatal: Could not initialize OSD. Exiting.")
        exit(1)
    print("OSD initialized.")

    print(f"Registering hotkey: {HOTKEY_STRING}")
    try:
        print(f"MAIN: Attempting to register hotkey '{HOTKEY_STRING}'...")
        keyboard.add_hotkey(HOTKEY_STRING, on_hotkey_pressed, suppress=False)
        print(f"MAIN: Hotkey '{HOTKEY_STRING}' registered successfully.")
        print(f"Press {HOTKEY_STRING} to measure distance.")
        print("Press any key on the OSD to hide it.")
    except Exception as e:
        error_message = f"Fatal: Could not register hotkey '{HOTKEY_STRING}'. Error: {e}\nTry running as administrator."
        print(error_message)
        try: tkinter.messagebox.showerror("Hotkey Registration Failed", error_message)
        except tk.TclError: print("(Tkinter not available for error dialog)")
        if osd_root: osd_root.destroy()
        exit(1)

    print("Application running. Press Ctrl+C in console to quit.")
    try:
        osd_root.mainloop()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
    except Exception as e_main:
        print(f"Main loop error: {e_main}")
    finally:
        print("Cleaning up resources...")
        print("MAIN: Unhooking all keyboard events...")
        keyboard.unhook_all()
        print("MAIN: Keyboard events unhooked.")
        if osd_root:
            print("MAIN: Destroying OSD window...")
            osd_root.destroy()
            print("MAIN: OSD window destroyed.")
        print("Shutdown complete.")