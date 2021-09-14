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
lbl = ImageLabel(root)
lbl.pack()
lbl.load('/home/frieda/iGibson/igibson/derived_data/straight_chair/219c603c479be977d5e0096fb2d3266a.gif')
root.mainloop()