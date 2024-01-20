import tinytuya

DEVICE_ID='-1'
IP_ADDR='10.0.0.1'
LOCAL_KEY="0"

d = tinytuya.BulbDevice(DEVICE_ID,IP_ADDR, LOCAL_KEY)
d.set_version(3.3)
data = d.status()
print('Device status: %r' % data)
resp = d.set_colour(200,120,50, False)
print(resp)
d.set_value(20, False)
