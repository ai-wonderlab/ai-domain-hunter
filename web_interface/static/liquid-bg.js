// Liquid Marble Background with Three.js
const liquidBackground = {
    init() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        
        this.renderer = new THREE.WebGLRenderer({ 
            alpha: true,
            antialias: true,
            powerPreference: "high-performance",
            precision: "highp",
            stencil: false,
            depth: false,
            preserveDrawingBuffer: false,
            logarithmicDepthBuffer: false,
            failIfMajorPerformanceCaveat: false
        });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio * 2, 4));
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.domElement.style.position = 'fixed';
        this.renderer.domElement.style.top = '0';
        this.renderer.domElement.style.left = '0';
        this.renderer.domElement.style.zIndex = '-1';
        this.renderer.domElement.id = 'liquid-canvas';
        document.body.insertBefore(this.renderer.domElement, document.body.firstChild);
        
        // Shader for liquid marble
        const vertexShader = `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = vec4(position, 1.0);
            }
        `;
        
        const fragmentShader = `
    uniform float time;
    uniform float theme;
    varying vec2 vUv;
    
    float noise(vec2 p) {
        return sin(p.x) * sin(p.y);
    }
    
    float fbm(vec2 p) {
        float value = 0.0;
        float amplitude = 0.5;
        
        for (int i = 0; i < 3; i++) {
            value += amplitude * noise(p);
            p *= 1.5;
            amplitude *= 0.6;
        }
        return value;
    }
    
    void main() {
        vec2 p = vUv * 2.5 - 1.25;
        
        // Smooth continuous breathing - NO random
        float breath = sin(time * 0.8) * 0.3 + cos(time * 0.5) * 0.2;
        float pulse = sin(time * 1.5) * 0.1;
        
        // Smooth continuous waves - NO morphing
        float t = time * 0.4;
        float wave1 = sin(p.x * (1.5 + breath) + fbm(p + t * 0.15) * 2.0);
        float wave2 = cos(p.y * (1.5 - breath) + fbm(p - t * 0.2) * 2.0);
        float wave3 = sin(length(p + vec2(sin(t * 0.3), cos(t * 0.2))) * (2.0 + pulse) - t);
        
        // Smooth combination - NO random
        float pattern = wave1 * wave2 + wave3 * (0.3 + pulse);
        pattern += fbm(p * (0.5 + breath * 0.2) + t * 0.08);
        
        // Smooth transitions
        float smooth1 = smoothstep(-1.0, 1.0, pattern);
        float smooth2 = smoothstep(-0.5, 0.5, pattern);  
        float smooth3 = smoothstep(-0.1, 0.1, pattern);
        
        // Continuous blending - NO sudden changes
        float final = smooth1 * 0.2 + smooth2 * 0.3 + smooth3 * 0.5;
        
        // Final smoothing
        final = smoothstep(0.45, 0.55, final);
        
        vec3 color = mix(
            vec3(1.0 - theme),  
            vec3(theme),        
            final
        );
        
        gl_FragColor = vec4(color, 1.0);
    }
`;
        
        this.material = new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms: {
                time: { value: 0 },
                theme: { value: 0 } // 0 for light, 1 for dark
            }
        });
        
        const geometry = new THREE.PlaneGeometry(2, 2);
        const mesh = new THREE.Mesh(geometry, this.material);
        this.scene.add(mesh);
        
        this.animate();
        this.handleResize();
        window.addEventListener('resize', () => this.handleResize());
        
        // Initialize animation speed
        this.animSpeed = 0.004;
    },
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Variable speed για πιο organic movement
        this.animSpeed = 0.0008 + Math.sin(Date.now() * 0.0008) * 0.001;
        this.material.uniforms.time.value += this.animSpeed;
        
        this.renderer.render(this.scene, this.camera);
    },
    
    handleResize() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    },
    
    setTheme(isDark) {
        if (this.material) {
            this.material.uniforms.theme.value = isDark ? 1 : 0;
        }
    }
};

// Initialize after DOM loads
document.addEventListener('DOMContentLoaded', () => {
    liquidBackground.init();
    
    // Connect to theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            liquidBackground.setTheme(!isDark);
        });
    }
});