/**
 * SportBackground - Elegant, professional sports-themed background
 * 
 * Design Philosophy:
 * - Very subtle, low opacity (0.03-0.08)
 * - Monochromatic/milky white tones that blend with dark theme
 * - Abstract representations of sports (no emojis, no flashy colors)
 * - Slow, barely noticeable movement
 * - Premium, atmospheric feel
 * - Stadium architecture vibes without literal depictions
 */

import { useEffect, useRef } from 'react';

interface FloatingShape {
  id: number;
  x: number;
  y: number;
  size: number;
  opacity: number;
  speedX: number;
  speedY: number;
  rotation: number;
  rotationSpeed: number;
  type: 'hexagon' | 'circle' | 'line' | 'arc';
}

export function SportBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shapesRef = useRef<FloatingShape[]>([]);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Initialize subtle shapes - very few, very slow
    const initShapes = () => {
      const shapes: FloatingShape[] = [];
      // Max 8 shapes for minimal, clean look
      const count = Math.min(8, Math.floor(window.innerWidth / 250));

      for (let i = 0; i < count; i++) {
        const types: FloatingShape['type'][] = ['hexagon', 'circle', 'line', 'arc'];
        shapes.push({
          id: i,
          type: types[Math.floor(Math.random() * types.length)],
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          size: 80 + Math.random() * 150,
          opacity: 0.05 + Math.random() * 0.05, // Visible but subtle
          speedX: (Math.random() - 0.5) * 0.05,
          speedY: (Math.random() - 0.5) * 0.05,
          rotation: Math.random() * Math.PI * 2,
          rotationSpeed: (Math.random() - 0.5) * 0.0005,
        });
      }
      shapesRef.current = shapes;
    };
    initShapes();

    const drawHexagon = (ctx: CanvasRenderingContext2D, shape: FloatingShape) => {
      const { x, y, size, rotation, opacity } = shape;
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.globalAlpha = opacity;
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const angle = (i * Math.PI) / 3;
        const px = Math.cos(angle) * size;
        const py = Math.sin(angle) * size;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    };

    const drawCircle = (ctx: CanvasRenderingContext2D, shape: FloatingShape) => {
      const { x, y, size, opacity } = shape;
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(x, y, size * 0.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = opacity * 0.5;
      ctx.beginPath();
      ctx.arc(x, y, size * 0.3, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };

    const drawLine = (ctx: CanvasRenderingContext2D, shape: FloatingShape) => {
      const { x, y, size, rotation, opacity } = shape;
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.globalAlpha = opacity;
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(-size, 0);
      ctx.lineTo(size, 0);
      ctx.stroke();
      ctx.restore();
    };

    const drawArc = (ctx: CanvasRenderingContext2D, shape: FloatingShape) => {
      const { x, y, size, rotation, opacity } = shape;
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.globalAlpha = opacity;
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.5, 0, Math.PI, false);
      ctx.stroke();
      ctx.restore();
    };

    const animate = () => {
      ctx.fillStyle = '#0a0e1a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, 'rgba(10, 14, 26, 0.8)');
      gradient.addColorStop(0.5, 'rgba(15, 22, 35, 0.4)');
      gradient.addColorStop(1, 'rgba(10, 14, 26, 0.8)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      shapesRef.current.forEach(shape => {
        shape.x += shape.speedX;
        shape.y += shape.speedY;
        shape.rotation += shape.rotationSpeed;

        if (shape.x < -200) shape.x = canvas.width + 200;
        if (shape.x > canvas.width + 200) shape.x = -200;
        if (shape.y < -200) shape.y = canvas.height + 200;
        if (shape.y > canvas.height + 200) shape.y = -200;

        switch (shape.type) {
          case 'hexagon':
            drawHexagon(ctx, shape);
            break;
          case 'circle':
            drawCircle(ctx, shape);
            break;
          case 'line':
            drawLine(ctx, shape);
            break;
          case 'arc':
            drawArc(ctx, shape);
            break;
        }
      });

      // Subtle vignette
      const vignette = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 0,
        canvas.width / 2, canvas.height / 2, canvas.width * 0.7
      );
      vignette.addColorStop(0, 'rgba(10, 14, 26, 0)');
      vignette.addColorStop(1, 'rgba(10, 14, 26, 0.4)');
      ctx.fillStyle = vignette;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationRef.current);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-0"
      style={{ background: '#0a0e1a' }}
    />
  );
}

export default SportBackground;
