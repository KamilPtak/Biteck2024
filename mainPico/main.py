import machine, utime

p = machine.Pin("LED", machine.Pin.OUT)

while True:
    p.value(1-p.value())
    utime.sleep_ms(500)