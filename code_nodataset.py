import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from detection import Detector
from controller import SignalController


# ------------------- LSTM MODEL -------------------
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64)
        c0 = torch.zeros(2, x.size(0), 64)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# ------------------- MAIN CONTROL FUNCTION -------------------
def run_control(image, detector, lstm_model, scaler, controller, past_seq):
    detection_result = detector.detect_from_image(image)
    detected = detection_result["detected"]

    height, width, _ = image.shape
    directions = ["North", "South", "East", "West"]

    counts = {d: {"Car": 0, "Bike": 0, "Bus": 0, "Truck": 0} for d in directions}
    occupancy = {d: 0 for d in directions}

    vehicle_classes = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}

    # Count total vehicles
    total_vehicles = {"Car": 0, "Bike": 0, "Bus": 0, "Truck": 0}
    for d in detected:
        cls_id = d["class"]
        vehicle_type = vehicle_classes.get(cls_id, "Other")
        if vehicle_type in total_vehicles:
            total_vehicles[vehicle_type] += 1

    total_vehicle_count = sum(total_vehicles.values())
    print(f"Total vehicles detected: {total_vehicle_count}")
    for v in total_vehicles:
        print(f"{v}: {total_vehicles[v]}")

    # Assign vehicles to directions
    for d in detected:
        x, y, w, h = d["bbox"]
        center_x = x + w // 2
        center_y = y + h // 2
        obj_area = w * h
        cls_id = d["class"]
        vehicle_type = vehicle_classes.get(cls_id, "Other")

        dist_top = center_y
        dist_bottom = height - center_y
        dist_left = center_x
        dist_right = width - center_x
        min_dist = min(dist_top, dist_bottom, dist_left, dist_right)

        if min_dist == dist_top:
            dir = "North"
        elif min_dist == dist_bottom:
            dir = "South"
        elif min_dist == dist_right:
            dir = "East"
        else:
            dir = "West"

        if vehicle_type in counts[dir]:
            counts[dir][vehicle_type] += 1
        occupancy[dir] += obj_area

    total_area = height * width
    occupancy = {d: occupancy[d] / total_area for d in directions}

    for d in directions:
        print(f"{d} - ", end="")
        for v in ["Car", "Bike", "Bus", "Truck"]:
            print(f"{v}: {counts[d][v]}, ", end="")
        print(f"Occupancy: {occupancy[d]:.4f}")

    # LSTM prediction
    past_seq = np.array(past_seq).reshape(-1, 1)
    past_seq_scaled = scaler.transform(past_seq)
    input_tensor = torch.tensor(
        past_seq_scaled.reshape((1, past_seq_scaled.shape[0], 1)), dtype=torch.float32
    )

    lstm_model.eval()
    with torch.no_grad():
        pred_scaled = lstm_model(input_tensor).item()
        predicted = scaler.inverse_transform([[pred_scaled]])[0][0]

    print(f"Predicted next traffic value: {predicted:.2f}")

    # Control signals
    traffic_counts = {d: sum(counts[d].values()) for d in directions}
    signals = controller.auto_control(traffic_counts)

    print("Traffic signals:", signals)
    return signals


# ------------------- MAIN EXECUTION -------------------
if __name__ == "__main__":
    image = cv2.imread("testimage1.jpg")
    if image is None:
        raise FileNotFoundError("Test image not found in the project folder.")

    detector = Detector(model_path="yolov8n.pt")
    lstm_model = TrafficLSTM()

    dummy_data = np.array([5, 7, 10, 8, 6, 9, 11, 12, 10, 7]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(dummy_data)

    controller = SignalController()
    past_seq = [5, 7, 10, 8, 6, 9, 11, 12, 10, 7]

    run_control(image, detector, lstm_model, scaler, controller, past_seq)
