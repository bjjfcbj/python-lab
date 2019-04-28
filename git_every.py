#! /bin/env/python3.6

import os
import datetime
import random

os.chdir('/home/d/PycharmProjects/py_ex')
num = input('回溯日期数： ')
date = datetime.datetime.now()
try:
    fp = open('everyday_git_blog.txt', 'w')
finally:
    fp.close()
try:
    for i in range(int(num), 0, -1):
        timestr = str((date - datetime.timedelta(days=i)).strftime('%Y-%m-%d'))
        os.system(('sudo date -s ' + timestr))
        for j in range(random.randint(1, 2)):
            with open("everyday_git_blog.txt", "a") as fp:
                fp.write(timestr + '\n')
            os.system('git add .')
            os.system('git commit -m' + timestr)

finally:
    os.system('git push')
    fp.close()
