let data = {
	nextTrainBatch: function(s, f) {
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
	},
	nextTestBatch: function(s, f) {
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

		req.open('GET', '/data-test?size='+s, true);
		req.send(null);
	}
}