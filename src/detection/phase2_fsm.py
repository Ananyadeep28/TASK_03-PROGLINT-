class BagActionFSM:
    def __init__(
        self,
        touch_threshold=50,
        displacement_threshold=40,
        return_tolerance=35
    ):
        self.touch_threshold = touch_threshold
        self.displacement_threshold = displacement_threshold
        self.return_tolerance = return_tolerance

        self.state = "PLACED (INITIAL)"

        self.initial_bag_pos = None
        self.last_known_bag_pos = None

        self.event_log = []

    def update(self, bag_center, hand_distance):

        # No bag detected
        if bag_center is None:
            return self.state

        # Save the first detected bag position
        if self.initial_bag_pos is None:
            self.initial_bag_pos = bag_center

        self.last_known_bag_pos = bag_center

        # Calculate bag movement from original position
        dx = bag_center[0] - self.initial_bag_pos[0]
        dy = bag_center[1] - self.initial_bag_pos[1]

        displacement = (dx ** 2 + dy ** 2) ** 0.5

        # -------------------------
        # PLACED -> PICKING
        # -------------------------

        if self.state == "PLACED (INITIAL)":

            if (
                hand_distance is not None
                and hand_distance <= self.touch_threshold
            ):
                self.state = "PICKING"

                self.event_log.append(
                    "Hand reached bag (Picking)"
                )

        # -------------------------
        # PICKING -> PICKED
        # -------------------------

        elif self.state == "PICKING":

            if displacement > self.displacement_threshold:

                self.state = "PICKED"

                self.event_log.append(
                    "Bag lifted/moved from initial spot (Picked)"
                )

        # -------------------------
        # PICKED -> PLACING
        # -------------------------

        elif self.state == "PICKED":

            if (
                displacement <= self.return_tolerance
                and
                hand_distance is not None
                and
                hand_distance <= self.touch_threshold
            ):
                self.state = "PLACING"

                self.event_log.append(
                    "Bag returned to original area (Placing)"
                )

        # -------------------------
        # PLACING -> PLACED
        # -------------------------

        elif self.state == "PLACING":

            if (
                hand_distance is None
                or
                hand_distance > self.touch_threshold
            ):
                self.state = "PLACED (RETURNED)"

                self.event_log.append(
                    "Hand released; bag placed back at rest (Placed)"
                )

        return self.state