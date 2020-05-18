#! /usr/bin/env python3

import os


def speak(path):
    pw = (os.getcwd())
    os.system('mpg321 {}/{}'.format(pw,path))
    return
