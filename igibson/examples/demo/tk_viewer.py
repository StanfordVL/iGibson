import os
import igibson
import tkinter as tk
from PIL import Image, ImageTk
from itertools import count, cycle

class ImageLabel(tk.Label):
    """
    A Label that displays images, and plays them if they are gifs
    :im: A PIL Image instance or a string filename
    """
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        frames = []

        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames[::5])

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100
            # print('err')
        self.delay = 1
        # print(self.delay)

        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)

#demo :
root = tk.Tk()
data_dir = os.path.join(os.path.dirname(os.path.dirname(igibson.ig_dataset_path)),
                "derived_data",
                "straight_chair")
print(f'loading {len(os.listdir(data_dir))} models')
for model_id in os.listdir(data_dir):
    lbl = ImageLabel(root)
    lbl.pack()
    lbl.load(os.path.join(data_dir, model_id))
    print(f'loaded {model_id}')
root.mainloop()