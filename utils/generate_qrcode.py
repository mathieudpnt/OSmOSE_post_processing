import os
import qrcode
from PIL import Image

# path = r'L:\acoustock\Bioacoustique\COMMUNICATION\LOGO'
# Logo_link = os.path.join(path, "logo_osmose.png")
# logo = Image.open(Logo_link).convert("RGBA")

# taking base width
basewidth = 100

# adjust image size
# wpercent = (basewidth / float(logo.size[0]))
# hsize = int((float(logo.size[1]) * float(wpercent)))
# logo = logo.resize((basewidth, hsize), Image.LANCZOS)

QRcode = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)

url = "https://osmose.ifremer.fr/"

QRcode.add_data(url)
QRcode.make()

# QRcolor = '#2683c6'
QRcolor = "white"

# Back_color = 'white'
Back_color = None

# Generate QR code with a transparent background
QRimg = QRcode.make_image(fill_color=QRcolor, back_color=Back_color).convert("RGBA")

# Combine QR code with the logo
pos = ((QRimg.size[0] - logo.size[0]) // 2, (QRimg.size[1] - logo.size[1]) // 2)
QRimg.paste(logo, pos, logo)

# Save the QR code with a transparent background
QRimg.save(os.path.join(path, "qrcode_transparent.png"))

print("QR code generated with transparent background!")

# %%
import os
import qrcode
from PIL import Image

path = r"L:\acoustock\Bioacoustique\COMMUNICATION\Site web"

# Create QR Code instance
QRcode = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
url = "https://osmose.ifremer.fr/"
QRcode.add_data(url)
QRcode.make()

# Generate QR code with a transparent background
QRimg = QRcode.make_image(fill_color="white", back_color="transparent").convert("RGBA")

# Save the QR code
QRimg.save(os.path.join(path, "qrcode_transparent.png"))

print("QR code generated with optional logo!")
