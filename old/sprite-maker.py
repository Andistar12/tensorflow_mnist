import numpy as np
from PIL import Image
from tensorflow import keras

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage

# Load in our data
X_OURS = []
Y_OURS = []
for image_index in range(10):
    image = Image.open(str(image_index) + ".jpg").convert("L")
    array = np.array(image)
    array = 1 - np.resize(array, (28,28)) / 255
    X_OURS.append(array)
    Y_OURS.append(image_index)
X_OURS = np.array(X_OURS).astype("float32")
Y_OURS = np.array(Y_OURS)

sprite_image = create_sprite_image(X_OURS)

result = Image.fromarray((sprite_image * 255).astype(np.uint8))
result.save('out.jpg')
