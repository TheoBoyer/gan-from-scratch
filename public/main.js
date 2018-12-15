const IMG_SIZE=28,
IMG_PIXELS = IMG_SIZE**2,
NOISE_SIZE = 100,
BATCH_SIZE = 150,
LEARNING_RATE = 0.0001;

/// Maths functions

function noise(shape) {
	return tf.randomNormal(shape)
}

function xavier_init(shape) {
	let stddev = 1 / Math.sqrt(shape.length==1 ? shape[0] : shape[1])
	return tf.randomUniform(shape, -stddev, stddev)
}

function vector_to_images(imgs) {
	let n_imgs = []
	let img, k
	for (let i = 0; i < imgs.length; i++) {
		img=[]
		k = 0;
		img[k] = []
		for (let j = 0; j < imgs[i].length; j++) {
			if(j!=0&&j%IMG_SIZE===0){
				k++
				img[k] = []
			}
			img[k][j%IMG_SIZE] = imgs[i][j]
		}
		n_imgs[i] = img
	}
	return n_imgs
}

function image_to_vectors(vectors) {
	let n_vectors = []
	let vector
	for (let i = 0; i < vectors.length; i++) {
		vector = []
		for (let j = 0; j < vectors[i].length; j++) {
			Array.prototype.push.apply(vector, vectors[i][j]);
		}
		n_vectors[i] = vector
	}
	return n_vectors
}

/// Data functions

function getBatch(s, f) {
	const req = new XMLHttpRequest();
	req.onreadystatechange = function(event) {
		// XMLHttpRequest.DONE === 4
		if (this.readyState === XMLHttpRequest.DONE) {
			if (this.status === 200) {
				return f(JSON.parse(this.responseText))
			} else {
				console.log("Status de la réponse: %d (%s)", this.status, this.statusText);
			}
		}
	};

	req.open('GET', '/data-train?size='+s, true);
	req.send(null);
}
/// Model

// Discriminator

// Layer 1 Variables
const D_W1 = tf.variable(xavier_init([IMG_PIXELS, 1024]))
const D_B1 = tf.variable(xavier_init([1024]))
// Layer 2 Variables
const D_W2 = tf.variable(xavier_init([1024, 512]))
const D_B2 = tf.variable(xavier_init([512]))
// Layer 3 Variables
const D_W3 = tf.variable(xavier_init([512, 256]))
const D_B3 = tf.variable(xavier_init([256]))
// Out layer Variables
const D_W4 = tf.variable(xavier_init([256, 1]))
const D_B4 = tf.variable(xavier_init([1]))
//List of variables
const D_var_list = [D_W1, D_B1, D_W2, D_B2, D_W3, D_B3, D_W4, D_B4]

// Generator

// Layer 1 Variables
const G_W1 = tf.variable(xavier_init([NOISE_SIZE, 256]))
const G_B1 = tf.variable(xavier_init([256]))
// Layer 2 Variables
const G_W2 = tf.variable(xavier_init([256, 512]))
const G_B2 = tf.variable(xavier_init([512]))
// Layer 3 Variables
const G_W3 = tf.variable(xavier_init([512, 1024]))
const G_B3 = tf.variable(xavier_init([1024]))
// Out layer Variables
const G_W4 = tf.variable(xavier_init([1024, IMG_PIXELS]))
const G_B4 = tf.variable(xavier_init([IMG_PIXELS]))
//List of variables
const G_var_list = [G_W1, G_B1, G_W2, G_B2, G_W3, G_B3, G_W4, G_B4]

/// Model Computing

function discriminator(x) {
	return tf.tidy(() => {
		let l1 = x.matMul(D_W1).add(D_B1).leakyRelu(0.2)
		let l1_d = tf.layers.dropout({rate: 0.3}).apply(l1)
		let l2 = l1_d.matMul(D_W2).add(D_B2).leakyRelu(0.2)
		let l2_d = tf.layers.dropout({rate: 0.3}).apply(l2)
		let l3 = l2_d.matMul(D_W3).add(D_B3).leakyRelu(0.2)
		let l3_d = tf.layers.dropout({rate: 0.3}).apply(l3)
		return l3_d.matMul(D_W4).add(D_B4)
	})
}

function generator(z) {
	return tf.tidy(() => {
		let l1 = z.matMul(G_W1).add(G_B1).leakyRelu(0.2)
		let l2 = l1.matMul(G_W2).add(G_B2).leakyRelu(0.2)
		let l3 = l2.matMul(G_W3).add(G_B3).leakyRelu(0.2)
		return l3.matMul(G_W4).add(G_B4).tanh()
	})
}

const D_optimizer = tf.train.adam(LEARNING_RATE)
const G_optimizer = tf.train.adam(LEARNING_RATE)

/// Training

const num_epochs = 200

train(num_epochs)

let pause = false, loss = 1

function onEnd() {
	console.log('finished')
	setInterval(function(){
		if(!pause) 
			draw()
	}, 1000);
}

function draw() {
	let content = '<div class="ecran">'
	let img = generator(noise([1, NOISE_SIZE])).mul(tf.scalar(-1)).add(tf.scalar(1)).mul(tf.scalar(255/2)).floor().dataSync()
	for (let i = 0; i < img.length; i++) {
		if(i>0&&i%IMG_SIZE===0)
			content += '<br>'
		content+='<div class="pixel" style="background-color: rgb('+img[i]+','+img[i]+','+img[i]+')"></div>'
	}
	content+='</div>'
	content+='<div id="loss">Loss: '+loss+'</div>'
	document.body.innerHTML = content
}

function train(n) {
	if(n==0) {
		return onEnd()
	}
	getBatch(BATCH_SIZE, function(c){
		let inputs = c.inputs
		for (let i = 0; i < inputs.length; i++) {
			D_optimizer.minimize(function(){
				let D_real = discriminator(tf.tensor2d([inputs[i]]).mul(tf.scalar(2/255)).sub(tf.scalar(1)))
				let D_fake = discriminator(generator(noise([1, NOISE_SIZE])))
				let D_loss_real = tf.sigmoidCrossEntropyWithLogits(tf.onesLike(D_real), D_real).mean()
				let D_loss_fake = tf.sigmoidCrossEntropyWithLogits(tf.zerosLike(D_fake), D_fake).mean()
				return D_loss_real.add(D_loss_fake)
			}, false, D_var_list)
			G_optimizer.minimize(function(){
				let D_fake = discriminator(generator(noise([1, NOISE_SIZE])))
				let out = tf.sigmoidCrossEntropyWithLogits(tf.onesLike(D_fake), D_fake).mean()
				loss = out.dataSync()[0]
				return out
			}, false, G_var_list)
		}
		console.log(n+' epochs remaining')
		draw()
		train(n-1)
	})
}

//console.log(discriminator(noise([1,IMG_PIXELS])).dataSync())

window.onload=function() {
	draw()
}