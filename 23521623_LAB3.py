# import library
import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Enhancing image", layout='wide')

st.title('Image enhancement')

uploaded = st.file_uploader('choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    file_bytes = np.asanyarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols, _ = img.shape

    # noisy
    noise = np.random.normal(0,15, (rows, cols, 3)).astype(np.uint8)
    noisy_img = img + noise

    #denoising
    denoiseMean = cv2.blur(noisy_img, (3,3))
    denoisedMedian = cv2.medianBlur(noisy_img, 3)
    smoothGauss = cv2.GaussianBlur(noisy_img, (5,5), sigmaX=4, sigmaY=4)

    #sharpening
    kernel = np.array ([[-1,-1,-1], 
                        [-1, 9,-1],
                        [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)

    #edge detection - Sobel
    gray = cv2.cvtColor(smoothGauss, cv2.COLOR_BGR2GRAY)

    ddepth = cv2.CV_16S
    grad_x =cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # prewitt
    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])
    kernely = np.array([[1,1,1],
                        [0,0,0],
                        [-1,-1,-1]])
    prx = cv2.filter2D(gray, -1, kernelx)
    pry = cv2.filter2D(gray, -1, kernely)
    prewitt = cv2.magnitude(prx.astype(np.float32), pry.astype(np.float32))
    prewitt = np.uint8(np.clip(prewitt, 0, 255))

    # canny using otsu thres
    otsu_thres, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny = cv2.Canny(gray, 0.5 * otsu_thres, 1.5 * otsu_thres)

    # plot
    titles = ['Origin', 'Noisy', 'Denoised(Mean)', 'Denoised(Median)', "Gaussian smooth", 'Sharpened', 'Sobel edge', 'Prewitt edge', 'Canny edge']
    imgs = [img, noisy_img, denoiseMean, denoisedMedian, smoothGauss, sharpened, sobel, prewitt, canny]

    fig, axes = plt.subplots(3,3,figsize=(16,14))
    for i, ax in enumerate(axes.flat):
        c_map = 'gray' if len(imgs[i].shape) == 2 else None
        ax.imshow(imgs[i], cmap=c_map)
        ax.set_title(titles[i], fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')

    plt.tight_layout(pad=3.0,h_pad=4.0, w_pad=4.0)
    st.pyplot(fig)
else:
    st.info("please upload an image to start")


