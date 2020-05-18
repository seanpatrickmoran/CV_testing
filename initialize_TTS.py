#! /usr/bin/env python3

from gtts import gTTS


mytext = 'hello!'
language = 'en'
myobj = gTTS(text=mytext, lang=language, slow=False)

myobj.save("file.mp3")

