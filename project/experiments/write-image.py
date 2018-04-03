import numpy as np

from PIL import Image

image = np.ones((128, 128), dtype=np.int8)

background = Image.new("RGB", image.shape, (255, 0, 0))
background.save('foo.png', 'PNG', quality=80)
