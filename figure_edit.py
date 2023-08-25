import math

from PIL import Image
import numpy as np
import time

file_path = 'D:\Documents-SSD\Interaction Technology\Thesis\Thesis-NLP_Alignment_Agreement_Sentiment\Results\Clustering\examples'
image = Image.open(file_path + '\\alignment_bin_4_class_5.png')

hsv_image = image.convert('HSV')

h, s, v = hsv_image.split()

#%%
destaturated = np.array(s)
destaturated[True] = destaturated[True] * 0.5
lowval = np.array(v)
lowval[True] = np.sqrt(lowval[True]/255)*255

des = Image.fromarray(destaturated)
viets = Image.fromarray(lowval)

new_image = Image.merge('HSV', (h, des, viets))
new_image.convert('RGB').save(f'{file_path}\\alignment_bin_4_class_5_lighter_{time.time()}.png')