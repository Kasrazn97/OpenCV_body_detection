import numpy as np
import matplotlib.pyplot as plt

df = np.load("output.npy")


# Subtract the initial position from all the others to have a reference
# for i in range(df.shape[0]):
    # c = df[i].copy()
    # c = np.roll(c, 1, axis=0)
    # c[0, :] = 0
    #df[i, :, :] -= c

figure, axis = plt.subplots(2, 2)

# Plotting X of RIGHT_FOREARM (greater is further to the right of the image)
axis[0, 0].plot(range(11), df[4, :, 0])
axis[0, 0].set_title("X of right_forearm")

# Plotting Y of RIGHT_FOREARM (greater is further up north of the image)
axis[0, 1].plot(range(11), df[4, :, 1])
axis[0, 1].set_title("Y of right_forearm")

# Plotting Z of RIGHT_FOREARM (greater is further from the camera)
axis[1, 0].plot(range(11), df[4, :, 2])
axis[1, 0].set_title("Z of right_forearm")

plt.show()