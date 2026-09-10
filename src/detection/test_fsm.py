from phase2_fsm import BagActionFSM


fsm = BagActionFSM()


test_frames = [
    # bag_center, hand_distance

    ((500, 400), 200),  # Bag initially placed
    ((500, 400), 100),  # Hand approaching
    ((500, 400), 40),   # Hand reaches bag -> PICKING

    ((520, 420), 30),   # Bag starts moving
    ((550, 450), 25),   # Bag moved enough -> PICKED

    ((520, 420), 30),   # Bag returning
    ((505, 405), 25),   # Bag near original position -> PLACING

    ((500, 400), 80),   # Hand moves away -> PLACED
]


for frame_number, (bag_center, hand_distance) in enumerate(
    test_frames, start=1
):

    state = fsm.update(
        bag_center,
        hand_distance
    )

    print(
        f"Frame {frame_number}: "
        f"Bag={bag_center}, "
        f"HandDistance={hand_distance}, "
        f"State={state}"
    )


print("\nEvent History:")

for event in fsm.event_log:
    print("-", event)