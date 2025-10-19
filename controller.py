class SignalController:
    def __init__(self):
        # Default all signals RED
        self.signals = {
            "North": "RED",
            "South": "RED",
            "East": "RED",
            "West": "RED"
        }

    def update_signal(self, direction, status):
        """Update the traffic signal for a given direction"""
        if direction in self.signals:
            self.signals[direction] = status

    def get_signals(self):
        """Return current signal states"""
        return self.signals

    def auto_control(self, traffic_counts):
        """
        Simple rule-based control:
        Green goes to the direction with max vehicles
        """
        # Reset all to RED
        for d in self.signals:
            self.signals[d] = "RED"

        # Pick direction with max traffic
        max_dir = max(traffic_counts, key=traffic_counts.get)
        self.signals[max_dir] = "GREEN"

        return self.signals
