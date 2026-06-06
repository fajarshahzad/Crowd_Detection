class CrowdClassifier:
    def __init__(self, sparse_limit=5, moderate_limit=20, dense_limit=50):
        self.sparse_limit = sparse_limit
        self.moderate_limit = moderate_limit
        self.dense_limit = dense_limit

    def classify(self, person_count):
        """
        Classifies crowd level based on the person count.
        """
        if person_count < self.sparse_limit:
            return {
                "level": "Sparse (Low)",
                "color": "green",
                "hex_color": "#28a745",
                "description": "The area is clear with very low density. Safe to operate under normal conditions.",
                "action": "Normal operations."
            }
        elif person_count < self.moderate_limit:
            return {
                "level": "Moderate (Medium)",
                "color": "yellow",
                "hex_color": "#ffc107",
                "description": "Noticeable crowd presence. Activity is normal but monitoring is advised.",
                "action": "Monitor situation."
            }
        elif person_count < self.dense_limit:
            return {
                "level": "Dense (High)",
                "color": "orange",
                "hex_color": "#fd7e14",
                "description": "High density crowd. Flow speed might be reduced. Social distancing/flow control recommended.",
                "action": "Activate flow management."
            }
        else:
            return {
                "level": "Crowded (Critical)",
                "color": "red",
                "hex_color": "#dc3545",
                "description": "Critically overcrowded area. High risk of stampede or bottleneck. Immediate control required.",
                "action": "Immediate crowd control / dispersion."
            }
