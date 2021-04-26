#! /usr/bin/env python3.6

from tkinter import *
from tkinter import messagebox
from functools import partial as pto

WARN = 'warn'
CRIT = 'crit'
REGU = 'regu'
SIGN = {
    'do not enter': CRIT,
    'railroad crossing': WARN,
    'speed limit': REGU,
    'wrong way': CRIT,
    'merging traffic': WARN,
    'one way': REGU,
}

critCB = lambda: messagebox.showerror('ERROR', 'error button pressed!')
warnCB = lambda: messagebox.showwarning('WARNING', 'warning button pressed')
infoCB = lambda: messagebox.showinfo('INFO', 'info button pressed')

root = Tk()
root.title('road sign')
Button(root, text='quit', command=root.quit, bg='red', fg='white').pack()

Mybutton = pto(Button, root)
Critbutton = pto(Mybutton, command=critCB, bg='white', fg='red')
Warnbutton = pto(Mybutton, command=warnCB, bg='goldenrod1')
Regubutton = pto(Mybutton, command=infoCB, bg='white')

for s in SIGN:
    st = SIGN[s]
    cmd = '%sbutton(text=%r%s).pack(fill=X,expand=True)' % (st.title(), s, '.upper()' if s == CRIT else '.title()')
    eval(cmd)

root.mainloop()
