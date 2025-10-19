import cv2
import numpy as np
import random

class SignalController:
    def __init__(self):
        self.signals = {"North": "RED", "South": "RED", "East": "RED", "West": "RED"}

    def auto_control(self, traffic_counts, emergency_direction=None):
        for d in self.signals:
            self.signals[d] = "RED"

        if emergency_direction:
            self.signals[emergency_direction] = "GREEN"
            return self.signals

        max_dir = max(traffic_counts, key=lambda d: traffic_counts[d])
        self.signals[max_dir] = "GREEN"
        return self.signals

class Vehicle:
    def __init__(self, direction, x, y, speed, color=(255,0,0), is_emergency=False):
        self.direction = direction
        self.x = x
        self.y = y
        self.speed = speed
        self.color = color
        self.is_emergency = is_emergency
        self.stopped = False

    def move(self, signal):
        if self.is_emergency:
            self.stopped = False
        else:
            self.stopped = (signal != "GREEN")

        if not self.stopped:
            if self.direction == "North":
                self.y -= self.speed
            elif self.direction == "South":
                self.y += self.speed
            elif self.direction == "East":
                self.x += self.speed
            elif self.direction == "West":
                self.x -= self.speed

def run_realistic_simulation():
    width, height = 600, 600
    controller = SignalController()
    vehicles = []
    directions = ["North", "South", "East", "West"]

    while True:
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add normal vehicles randomly
        if random.random() < 0.05:
            dir_choice = random.choice(directions)
            if dir_choice == "North":
                vehicles.append(Vehicle("North", width//2 - 30, height, speed=2))
            elif dir_choice == "South":
                vehicles.append(Vehicle("South", width//2 + 30, 0, speed=2))
            elif dir_choice == "East":
                vehicles.append(Vehicle("East", 0, height//2 - 30, speed=2))
            elif dir_choice == "West":
                vehicles.append(Vehicle("West", width, height//2 + 30, speed=2))

        # Add emergency vehicle (Ambulance) randomly
        if random.random() < 0.01:
            dir_choice = random.choice(directions)
            vehicles.append(
                Vehicle(dir_choice, width//2, height//2, speed=3, color=(0,0,255), is_emergency=True)
            )

        traffic_counts = {d:0 for d in directions}
        emergency_direction = None
        for v in vehicles:
            if v.is_emergency:
                emergency_direction = v.direction
            traffic_counts[v.direction] += 1

        signals = controller.auto_control(traffic_counts, emergency_direction)

        # Draw roads
        cv2.rectangle(frame, (width//2-50,0),(width//2+50,height),(50,50,50),-1)
        cv2.rectangle(frame, (0,height//2-50),(width,height//2+50),(50,50,50),-1)

        # Draw vehicles
        for v in vehicles:
            v.move(signals[v.direction])
            cv2.rectangle(frame, (int(v.x)-10,int(v.y)-10), (int(v.x)+10,int(v.y)+10), v.color, -1)

            # Label emergency vehicle as "Ambulance"
            if v.is_emergency:
                cv2.putText(frame, "Ambulance", (int(v.x)-20, int(v.y)-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # Keep vehicles within bounds
        vehicles = [v for v in vehicles if 0 <= v.x <= width and 0 <= v.y <= height]

        # Draw traffic signals
        signal_pos = {"North":(width//2-100,50), "South":(width//2+50,height-50),
                      "East":(width-50,height//2-100), "West":(50,height//2+50)}
        for d,(x,y) in signal_pos.items():
            color = (0,255,0) if signals[d]=="GREEN" else (0,0,255)
            cv2.circle(frame,(x,y),15,color,-1)
            cv2.putText(frame,d,(x-30,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        cv2.imshow("Realistic Smart Traffic Simulation", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realistic_simulation()