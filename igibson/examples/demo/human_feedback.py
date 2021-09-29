import threading as th

from pynput import keyboard


# order: torso - x, y, z, roll, pitch, yaw
#        head
#        left hand
#        right hand
#        gripper
class HumanFeedback:
    def __init__(self):
        super(HumanFeedback, self).__init__()
        self.keyboard_feedback_dictionary = {
            keyboard.KeyCode.from_char("w"): [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # +y
            keyboard.KeyCode.from_char("a"): [
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # -x
            keyboard.KeyCode.from_char("s"): [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],  # +z
            keyboard.KeyCode.from_char("d"): [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # +x
            keyboard.KeyCode.from_char("z"): [
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
            ],  # -z
            keyboard.KeyCode.from_char("x"): [
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # -y
        }
        self.human_keyboard_feedback = None
        self.run_keyboard_capture_thread()

    def keyboard_capture_thread(self):
        with keyboard.Events() as events:
            event = events.get(1e6)
            self.human_keyboard_feedback = event

    def run_keyboard_capture_thread(self):
        th.Thread(target=self.keyboard_capture_thread, args=(), name="keyboard_capture_thread", daemon=True).start()

    def return_human_keyboard_feedback(self, action_length):
        feedback = None
        if self.human_keyboard_feedback:
            if "Press" in str(self.human_keyboard_feedback):  # only use keypresses as reward signals
                if self.human_keyboard_feedback.key in self.keyboard_feedback_dictionary:
                    feedback = [0 for _ in range(action_length)]
                    feedback = self.keyboard_feedback_dictionary[self.human_keyboard_feedback.key]
                elif self.human_keyboard_feedback.key == keyboard.KeyCode.from_char("p"):  # use 'p' for pausing:
                    feedback = "Pause"

            self.run_keyboard_capture_thread()

        self.human_keyboard_feedback = None
        return feedback
