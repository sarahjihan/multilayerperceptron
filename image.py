from PIL import Image

class ImageReader:
	def read(self, url):
		img = Image.open(url).convert('L')  # convert image to 8-bit grayscale
		WIDTH, HEIGHT = img.size

		data = list(img.getdata()) # convert image data to a list of integers
		# convert that to 2D list (list of lists of integers)
		data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

		# At this point the image's pixels are all in memory and can be accessed
		# individually using data[row][col].

		# For example:
		out = []
		for row in data:
		    out += row

		return out
# # Here's another more compact representation.
# chars = '@%#*+=-:. '  # Change as desired.
# scale = (len(chars)-1)/255.
# print()
# for row in data:
#     print(' '.join(chars[int(value*scale)] for value in row))
