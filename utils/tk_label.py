#!/usr/bin/env python3.6

from tkinter import *


def font_resize(env=None):
    label.config(font='Helvetica -%d bold' % scale.get())


root = Tk()
root.geometry('250x150')
label = Label(root, text='hello world')
label.pack(fill=Y, expand=1)
scale = Scale(root, from_=10, to=40, orient=HORIZONTAL, command=font_resize)
scale.set(12)
scale.pack(fill=X, expand=1)
quit = Button(root, text='quit', command=root.quit)
quit.pack(fill=X, expand=1)
mainloop()
