# ATCS_vision_tutorial

My project discusses the behavior of two different blob detection schemes:
- Laplacian of Gaussian 
- Difference of Gaussians 

These filters are all aimed at detecting sections (“blobs”) of an image which have some sort of generally constant property (like brightness or color) that differs from the background. They are concerned primarily with approximating multivariable second derivatives.

<ins> Laplacian of Gaussian (LoG) </ins>: The motivation behind this scheme is to analyze regions of rapid intensity change in an image. This is done by approximating the 2D second derivative (hence, the Laplacian operator) of an image using a convolution filter. 

Since the LoG is quite sensitive to noise, it is important to use a Gaussian filter. Interestingly enough, this opens us up to a shortcut. We can make use of the associativity of the convolution operator to apply the Laplacian operator to the Gaussian filter and get a function which we can then discretize into a standalone filter. Using that single filter is often more computationally efficient than doing both runs.

<ins> Difference of Gaussians (DoG)</ins>: This is essentially an approximation of the LoG approach, which comes from an interesting mathematical result. Essentially, when you subtract two different Gaussian blurs, you get a function that resembles the laplacian of gaussian.

I worked with three different images: one of Lena (a classic CV image), one a butterfly, and one of Salvador Dali’s “The Persistence of Memory.”
