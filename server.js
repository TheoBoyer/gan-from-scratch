const fs = require('fs');

const IMAGE_NUMBER = 59999;
const IMAGE_TEST_NUMBER = 9999;

let dataFileBuffer  = fs.readFileSync(__dirname + '/train-images.idx3-ubyte');
let labelFileBuffer = fs.readFileSync(__dirname + '/train-labels.idx1-ubyte');
let pixelValues     = [];

// It would be nice with a checker instead of a hard coded 60000 limit here
for (let image = 0; image <= IMAGE_NUMBER; image++) { 
    let pixels = [];

    for (let y = 0; y <= 27; y++) {
        for (let x = 0; x <= 27; x++) {
            pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16]);
        }
    }

    let imageData  = {
    	label: parseInt(JSON.stringify(labelFileBuffer[image + 8])),
    	pixels: pixels
    };

    pixelValues.push(imageData);
}

dataFileBuffer  = fs.readFileSync(__dirname + '/t10k-images.idx3-ubyte');
labelFileBuffer = fs.readFileSync(__dirname + '/t10k-labels.idx1-ubyte');
let pixelTestValues     = [];

// It would be nice with a checker instead of a hard coded 60000 limit here
for (let image = 0; image <= IMAGE_TEST_NUMBER; image++) { 
    let pixels = [];

    for (let y = 0; y <= 27; y++) {
        for (let x = 0; x <= 27; x++) {
            pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16]);
        }
    }

    let imageData  = {
    	label: parseInt(JSON.stringify(labelFileBuffer[image + 8])),
    	pixels: pixels
    };

    pixelTestValues.push(imageData);
}

console.log('Images loaded !');

function getData() {
	return pixelValues[Math.round(Math.random()*IMAGE_NUMBER)];
}

function getTestData() {
	return pixelTestValues[Math.round(Math.random()*IMAGE_TEST_NUMBER)];
}

function pickBatch(s) {
	let i = [];
	let l = [];
	for(let k=0; k<s; k++){
		let d = getData();
		i[k] = d.pixels;
		l[k] = getOutputs(d.label);
	}
	return {
		inputs: i,
		labels: l
	}
}

function pickTestBatch(s) {
	let i = [];
	let l = [];
	for(let k=0; k<s; k++){
		let d = getTestData();
		i[k] = d.pixels;
		l[k] = getOutputs(d.label);
	}
	return {
		inputs: i,
		labels: l
	}
}

function getOutputs(n) {
	let r = [
		0,0,0,0,0,0,0,0,0,0
	];
	r[n] = 1;
	return r;
}

const express = require('express')
const app = express()

app.get('/data-train', function (req, res) {
	res.send(JSON.stringify(pickBatch(req.query.size)))
})

app.get('/data-test', function (req, res) {
	res.send(JSON.stringify(pickTestBatch(req.query.size)))
})

app.use(express.static('public'))

app.listen(1000, function () {
  console.log('Example app listening on port 1000!')
})