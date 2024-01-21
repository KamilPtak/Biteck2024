import tinytuya
import mySecrets
DEVICE_ID=mySecrets.DEVICE_ID
IP_ADDR=mySecrets.IP_ADDR
LOCAL_KEY=mySecrets.LOCAL_KEY

d = tinytuya.BulbDevice(DEVICE_ID,IP_ADDR, LOCAL_KEY)
d.set_version(3.3)
# data = d.status()
# print('Device status: %r' % data)
# resp = d.set_colour(200,120,50, False)
# print(resp)
# d.set_value(20, False)

def setLightsColor(rgb):
    d.set_colour(rgb['r'], rgb['g'], rgb['b'], False)

def setLightOn(on=True):
    d.set_value(20, on)