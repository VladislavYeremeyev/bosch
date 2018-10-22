from pdfTransformStyle import *

f1, f2 = ('data/cur/2.jpg', 'data/cur/3.jpg')
alice, reality = ims(f1,f2)
a = reality
b = alice
b = b[:770,800:,:]
imshow0(a)
imshow0(b)

c = np.copy(b)

#c0 = colorPdfTransform(a,b)
c0 = colorPdfTransform(incrColorful(a, 10),incrColorful(b,10))
#c0 = colorPdfTransform(b,a)
imshow(c0)

