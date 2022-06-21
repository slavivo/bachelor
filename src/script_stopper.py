from pynput import keyboard

class ScriptStopper:
  def __init__(self):
    self.listener = keyboard.Listener(on_press=self.on_press)
    self.listener.start()
    self.stopped = False

  def on_press(self,key):
      try:
          k = key.char
          if(k == 'f'):
              self.stopped = True
      except AttributeError:
          pass

  def stop(self):
    self.listener.stop()

  def script_stopped(self):
    return self.stopped