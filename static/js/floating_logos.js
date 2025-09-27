document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("bubbleCanvas");
    const ctx = canvas.getContext("2d");

    // Set canvas size dynamically
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    const logoImage = new Image();
    logoImage.src = "/static/WeCareheart.png"; // Update image path as needed

    const bubbles = [];
    const numBubbles = 15; // Adjust number of floating logos

    class Bubble {
        constructor() {
            this.radius = Math.random() * 40 + 30; // Random size between 30-70px
            this.x = Math.random() * (canvas.width - this.radius * 2) + this.radius;
            this.y = Math.random() * (canvas.height - this.radius * 2) + this.radius;
            this.dx = (Math.random() - 0.5) * 2.5; // Random horizontal speed
            this.dy = (Math.random() - 0.5) * 2.5; // Random vertical speed
        }

        draw() {
            ctx.drawImage(logoImage, this.x - this.radius, this.y - this.radius, this.radius * 2, this.radius * 2);
        }

        update() {
            this.x += this.dx;
            this.y += this.dy;

            // Collision detection for canvas edges
            if (this.x - this.radius <= 0 || this.x + this.radius >= canvas.width) {
                this.dx *= -1; // Reverse direction on collision
            }
            if (this.y - this.radius <= 0 || this.y + this.radius >= canvas.height) {
                this.dy *= -1; // Reverse direction on collision
            }

            this.draw();
        }
    }

    function init() {
        for (let i = 0; i < numBubbles; i++) {
            bubbles.push(new Bubble());
        }
        animate();
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        bubbles.forEach(bubble => bubble.update());
        requestAnimationFrame(animate);
    }

    logoImage.onload = init;
});
