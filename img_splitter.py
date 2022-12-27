import os
import numpy as np
import cv2
import PIL
from PIL import Image
from tqdm import tqdm


# fix seed
np.random.seed(0)

def randomize_bg(img: np.array) -> np.array:
    """Randomize the background of the image"""
    color = np.random.randint(0, 255, 3)
    color = color.reshape(1, 3)
    # color where alpha channel is 0
    idxes = (img[:, :, 3] == 0)
    img = img[:, :, :3]
    img[idxes] = color
    return img

path = "punks.png"
# img = cv2.imread(path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.open(path)
img = np.array(img)
# BGR to RGB
img = img[:, :, [2, 1, 0, 3]]

# #show image
# cv2.imshow("image", img)
# cv2.waitKey(0)

#slice image
num_rows = 100
num_col = 100


rows = np.vsplit(img, num_rows)
cells = []
for row in rows:
    row_cells = np.hsplit(row, num_col)
    for cell in row_cells:
        cell = randomize_bg(cell)
        cells.append(cell)

        # cell = cv2.resize(cell, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        # cell = cv2.resize(cell, (128, 128), interpolation=cv2.INTER_AREA)
        # cell = cv2.resize(cell, (128, 128), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow("cell", cell)
        # cv2.waitKey(0)

        # cell = cv2.resize(cell, (24, 24), interpolation=cv2.INTER_LANCZOS4)
        # cv2.imshow("cell", cell)
        # cv2.waitKey(0)

#save cells as images
if not os.path.exists("images"):
    os.mkdir("images")
for i, cell in enumerate(tqdm(cells)):
    path = os.path.join("images", str(i) + ".png")
    cv2.imwrite(path, cell)