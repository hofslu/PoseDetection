import os
import cv2
import numpy as np
import json
from itertools import combinations

DATA_DIR = os.path.join(os.getcwd(), '../data')
POINTS_PER_EDGE = 4
EDGE_NAMES = ['Top', 'Right', 'Bottom', 'Left']

def crop_image(image):
    roi = cv2.selectROI("Crop the Desk Area", image, False)
    cv2.destroyWindow("Crop the Desk Area")
    x, y, w, h = roi
    return image[y:y+h, x:x+w], (x, y)

def collect_edge_points(image, edge_name):
    print(f"[🖱] Click {POINTS_PER_EDGE} points for the {edge_name} edge...")
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < POINTS_PER_EDGE:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow(f"Click: {edge_name}", display_img)

    display_img = image.copy()
    cv2.imshow(f"Click: {edge_name}", display_img)
    cv2.setMouseCallback(f"Click: {edge_name}", click_event)

    while len(points) < POINTS_PER_EDGE:
        cv2.waitKey(1)

    cv2.destroyWindow(f"Click: {edge_name}")
    return np.array(points, dtype=np.float32)  # ✅ No scaling here anymore


def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    if np.ptp(x) > np.ptp(y):  # horizontal-ish
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return lambda x: m * x + b, 'h'
    else:
        A = np.vstack([y, np.ones(len(y))]).T
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        return lambda y: m * y + b, 'v'

def compute_intersection(f1, f2, mode1, mode2):
    # Compute intersection between two lines defined as y=f(x) or x=f(y)
    if mode1 == 'h' and mode2 == 'h':
        # Solve f1(x) = f2(x)
        x = np.linspace(0, 1000, 1000)
        diff = np.abs(f1(x) - f2(x))
        x_best = x[np.argmin(diff)]
        return int(x_best), int(f1(x_best))

    elif mode1 == 'v' and mode2 == 'v':
        y = np.linspace(0, 1000, 1000)
        diff = np.abs(f1(y) - f2(y))
        y_best = y[np.argmin(diff)]
        return int(f1(y_best)), int(y_best)

    elif mode1 == 'h' and mode2 == 'v':
        # f1 is y(x), f2 is x(y)
        # Solve: x = f2(y), y = f1(x)
        # Use fixed-point iteration
        y = np.linspace(0, 1000, 1000)
        x = f2(y)
        diff = np.abs(y - f1(x))
        best = np.argmin(diff)
        return int(x[best]), int(y[best])

    elif mode1 == 'v' and mode2 == 'h':
        # f1 is x(y), f2 is y(x)
        x = np.linspace(0, 1000, 1000)
        y = f2(x)
        diff = np.abs(x - f1(y))
        best = np.argmin(diff)
        return int(x[best]), int(y[best])


def main():
    image_path = DATA_DIR + "/ESP32-CAM2_snapshot.jpg"
    image = cv2.imread(image_path)

    # Step 1: Crop the image
    cropped_img, crop_offset = crop_image(image)
    if cropped_img is None:
        print("[❌] No image selected. Exiting.")
        return

    # Step 1.5: Scale up cropped image for better point selection
    scale_factor = 2.0
    cropped_img = cv2.resize(cropped_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    lines = []
    modes = []
    all_edge_points = []

    for edge_name in EDGE_NAMES:
        edge_pts_scaled = collect_edge_points(cropped_img, edge_name)
        edge_pts = edge_pts_scaled / scale_factor  # 🔄 Revert to original cropped scale
        all_edge_points.append(edge_pts)
        f, mode = fit_line(edge_pts)
        lines.append(f)
        modes.append(mode)

    # Step 2: Calculate corners from intersecting lines
    corners = []
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edge_pairs:
        pt = compute_intersection(lines[i], lines[j], modes[i], modes[j])
        pt_global = (pt[0] + crop_offset[0], pt[1] + crop_offset[1])
        corners.append(pt_global)

    # Step 3: Visualize debug output
    debug_img = image.copy()

    for i in range(len(EDGE_NAMES)):
        edge_pts = all_edge_points[i] + np.array(crop_offset)

        for pt in edge_pts:
            cv2.circle(debug_img, tuple(int(v) for v in pt), 4, (0, 0, 255), -1)

        # Draw fitted lines with offset
        if modes[i] == 'h':
            x_vals = np.linspace(0, cropped_img.shape[1], 2)
            y_vals = lines[i](x_vals)
        else:
            y_vals = np.linspace(0, cropped_img.shape[0], 2)
            x_vals = lines[i](y_vals)

        pts = np.column_stack((x_vals + crop_offset[0], y_vals + crop_offset[1])).astype(int)
        cv2.line(debug_img, tuple(pts[0]), tuple(pts[1]), (255, 0, 0), 2)

    for pt in corners:
        cv2.circle(debug_img, pt, 6, (0, 255, 0), -1)

    cv2.imshow("Debug Overlay", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("debug_overlay_result.jpg", debug_img)
    print("[🖼️] Saved debug overlay as 'debug_overlay_result.jpg'")

    # Step 4: Save result
    save_path = "camera_2_corners.json"
    with open(save_path, 'w') as f:
        json.dump({"corners": corners}, f)

    print(f"[✅] Saved 4 corner points to: {save_path}")


if __name__ == "__main__":
    main()
