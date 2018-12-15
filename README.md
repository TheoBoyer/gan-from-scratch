# gan-from-scratch
Implementation of flat mnist Generative Adversiarial Neural Network using low level features of tfjs

This is a simple implementation of gan using only very basics features. The neural nets are not convolutionnals ones, they are just flat. 
Obviously the result is not very amazing but is pretty nice. The images are a little bit noisy but on most of it we can clearly recognize the digits.

Installing :
- Clone the repo

- Install dependencies : "npm install"

- Download mnist dataset and copy all 4 files into the root of the project here : http://yann.lecun.com/exdb/mnist/

- Launch the server : "node server.js"

- Go to : http://localhost:1000

- You should see some images produced by the gan at each epoch.

- At the end of training (200 epochs by default) the trained GAN generate random images each second

- To save an image just right click on it, it's a HTML canvas so most popular browser should give you possibility to download it.
