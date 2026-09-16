import time
import subprocess
import digitalio
import board
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.st7789 as st7789

# Configuration for CS and DC pins (these are FeatherWing defaults on M0/M4):
cs_pin = digitalio.DigitalInOut(board.D5) 
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = None

# Config for display baudrate (default max is 24mhz):
BAUDRATE = 64000000

# Setup SPI bus using hardware SPI:
spi = board.SPI()

# Create the ST7789 display:
disp = st7789.ST7789(
    spi,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
    width=135,
    height=240,
    x_offset=53,
    y_offset=40,
)

# Create blank image for drawing.
# Make sure to create image with mode 'RGB' for full color.
height = disp.width  # we swap height/width to rotate it to landscape!
width = disp.height
image = Image.new("RGB", (width, height))
rotation = 90

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a black filled box to clear the image.
draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))
disp.image(image, rotation)
# Draw some shapes.
# First define some constants to allow easy resizing of shapes.
padding = -2
top = padding
bottom = height - padding
# Move left to right keeping track of the current x position for drawing shapes.
x = 0

# Alternatively load a TTF font.  Make sure the .ttf font file is in the
# same directory as the python script!
# Some other nice fonts to try: http://www.dafont.com/bitmap.php
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)

# Turn on the backlight
backlight = digitalio.DigitalInOut(board.D22)
backlight.switch_to_output()
backlight.value = True

# 1 second -> 1 step; km rolls over every 1400 steps; marathons roll over every 42 km
STEPS_PER_KM = 1400
KM_PER_MARATHON = 42
MESSAGE_DURATION = 3  # seconds to keep a milestone message on screen
start_time = time.time()
prev_total_km = 0
prev_marathons = 0
message = ""
message_until = 0

while True:
    # Draw a black filled box to clear the image.
    draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))

    total_steps = int(time.time() - start_time)
    total_km = total_steps // STEPS_PER_KM
    steps = total_steps % STEPS_PER_KM
    marathons = total_km // KM_PER_MARATHON
    km = total_km % KM_PER_MARATHON

    if total_km != prev_total_km:
        message = "Marathon reached!" if marathons != prev_marathons else "Km reached!"
        message_until = time.time() + MESSAGE_DURATION
    prev_total_km = total_km
    prev_marathons = marathons

    distance_string = "{} Mar : {} Km : {} Step(s)".format(marathons, km, steps)
    date_string = time.strftime("%m/%d/%Y")
    draw.text((x, top), date_string, font=font, fill="#FFFFFF")
    draw.text((x, top + 20), distance_string, font=font, fill="#FFFFFF")
    if time.time() < message_until:
        draw.text((x, top + 40), message, font=font, fill="#FFFFFF")

    # Display image.
    disp.image(image, rotation)
    time.sleep(1)
