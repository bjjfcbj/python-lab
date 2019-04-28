#! /urs/bin/env python3.6

import os
from time import sleep
from tkinter import *


class Dirlist(object):
    def __init__(self, init_dir=None):
        self.root = Tk()
        self.label = Label(self.root, text='Directory lister 1.0')
        self.label.pack()

        self.cwd = StringVar(self.root)

        self.dirl = Label(self.root, fg='blue', font=('Helvetica', 12, 'bold'))
        self.dirl.pack()

        self.dirfm = Frame(self.root)
        self.dirsb = Scrollbar(self.dirfm)
        self.dirsb.pack(side=RIGHT, fill=Y)
        self.dirb = Listbox(self.dirfm, height=15, width=50, yscrollcommand=self.dirsb.set)
        self.dirb.bind('<Double-1>', self.setDirAndGo)
        self.dirsb.config(command=self.dirb.yview)
        self.dirb.pack(side=LEFT, fill=BOTH)
        self.dirfm.pack()

        self.dirn = Entry(self.root, width=50, textvariable=self.cwd)
        self.dirn.bind('<Return>', self.doLS)
        self.dirn.pack()

        self.bfm = Frame(self.root)
        self.clr = Button(self.bfm, text='clear', command=self.clrDir, activeforeground='white',
                          activebackground='blue')
        self.ls = Button(self.bfm, text='list', command=self.doLS, activeforeground='white', activebackground='blue')
        self.quit = Button(self.bfm, text='quit', command=self.root.quit, activeforeground='white',
                           activebackground='blue')
        self.clr.pack(side=LEFT)
        self.ls.pack(side=LEFT)
        self.quit.pack(side=LEFT)
        self.bfm.pack()

        if init_dir:
            self.cwd.set(os.curdir)
            self.doLS()

    def clrDir(self, ev=None):
        self.cwd.set('')

    def setDirAndGo(self, ev=None):
        self.last = self.cwd.get()
        self.dirb.config(selectbackground='red')
        check = self.dirb.get(self.dirb.curselection())
        if not check:
            check = os.curdir
        self.cwd.set(check)
        self.doLS()

    def doLS(self, ev=None):
        error = ''
        tdir = self.cwd.get()
        if not tdir:
            tdir = os.curdir
        if not os.path.exists(tdir):
            error = tdir + ': not such file'
        elif not os.path.isdir(tdir):
            error = tdir + ': not a directory'

        if error:
            self.cwd.set(error)
            self.root.update()
            sleep(2)
            if not (hasattr(self, 'last') and self.last):
                self.last = os.curdir
            self.cwd.set(self.last)
            self.dirb.config(selectbackground='LightSkyBlue')
            self.root.update()
            return

        self.cwd.set('FETCHING DIRECTORY CONTENTS...')
        self.root.update()
        dirlist = os.listdir(tdir)
        dirlist.sort()
        os.chdir(tdir)

        self.dirl.config(text=os.getcwd())
        self.dirb.delete(0, END)
        self.dirb.insert(END, os.curdir)
        self.dirb.insert(END, os.pardir)
        for f in dirlist:
            self.dirb.insert(END, f)
        self.cwd.set(os.curdir)
        self.dirb.config(selectbackground='LightSkyBlue')


def main():
    d = Dirlist(os.curdir)
    mainloop()


if __name__ == '__main__':
    main()
