# PUBG Map Distance Calculator

## Overview

This Python script calculates the distance between two points (typically a player and a ping) on a PlayerUnknown's Battlegrounds (PUBG) map image. It achieves this by dynamically determining the map's scale (meters per pixel) from the grid lines visible on the map.

## Installation

1.  Clone this repository or download the source code.
2.  Ensure you have Python installed.
3.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure the template images `player-icon.png` and `player-ping.png` are present in the same directory as the script (`pubg_distance_calculator.py`).
2.  Run the script from your terminal:
    ```bash
    python pubg_distance_calculator.py
    ```
3.  The script will run in the background, waiting for hotkey presses. You'll see messages in the console indicating it's running (e.g., "Application running. Press Ctrl+C in console to quit.").
4.  **To Calculate Distance:**
    *   Make sure the PUBG map is visible on your primary monitor.
    *   Press the hotkey `Ctrl+Shift+Alt+D`.
    *   The script will capture the screen, show a brief "Processing..." indicator, analyze the map area (defined by borders set within the script), find the player and ping icons, calculate the distance, and display the result in an On-Screen Display (OSD) near the top-center of your screen.
    *   Press any key to dismiss the OSD result.
5.  **To View Debug Overlay:**
    *   Press the hotkey `Ctrl+Shift+Alt+K`.
    *   This performs a similar analysis but displays a full-screen overlay showing the detected grid lines, scale reference, processing border, and detected icon locations.
    *   Press any key to dismiss the debug overlay.
6.  **To Stop the Script:** Go back to the terminal window where you started the script and press `Ctrl+C`.