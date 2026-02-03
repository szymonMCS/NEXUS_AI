/**
 * FloatingBackground - Animated sports betting background
 * 
 * Features:
 * - Floating coins (tokens) drifting in space
 * - Trophies floating around
 * - Subtle parallax effect
 * - Neon glow effects
 * - Stadium lights in background
 */

import { useEffect, useRef } from 'react';

interface FloatingItem {
  id: number;
  type: 'coin' | 'trophy' | 'chip' | 'ball';
  x: number;
  y: number;
  size: number;
  speedX: number;
  speedY: number;
  rotation: number;
  rotationSpeed: number;
  opacity: number;
  glow: string;
}

export function FloatingBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const itemsRef = useRef<FloatingItem[]>([]);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Initialize floating items
    const initItems = () => {
      const items: FloatingItem[] = [];
      const count = Math.min(25, Math.floor(window.innerWidth / 60));

      for (let i = 0; i < count; i++) {
        const types: FloatingItem['type'][] = ['coin', 'trophy', 'chip', 'ball'];
        items.push({
          id: i,
          type: types[Math.floor(Math.random() * types.length)],
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          size: 30 + Math.random() * 50,
          speedX: (Math.random() - 0.5) * 0.3,
          speedY: (Math.random() - 0.5) * 0.3 - 0.1, // Slight upward drift
          rotation: Math.random() * Math.PI * 2,
          rotationSpeed: (Math.random() - 0.5) * 0.01,
          opacity: 0.1 + Math.random() * 0.25,
          glow: ['#00d4ff', '#00ff88', '#ff9500', '#b829dd'][Math.floor(Math.random() * 4)],
        });
      }
      itemsRef.current = items;
    };
    initItems();

    // Draw functions
    const drawCoin = (ctx: CanvasRenderingContext2D, item: FloatingItem) => {
      const { x, y, size, rotation, glow, opacity } = item;
      
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.globalAlpha = opacity;
      
      // Outer glow
      const gradient = ctx.createRadialGradient(0, 0, size * 0.3, 0, 0, size);
      gradient.addColorStop(0, glow);
      gradient.addColorStop(0.5, glow + '40');
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(0, 0, size, 0, Math.PI * 2);
      ctx.fill();
      
      // Coin body
      ctx.strokeStyle = glow;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.6, 0, Math.PI * 2);
      ctx.stroke();
      
      // Inner detail
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.4, 0, Math.PI * 2);
      ctx.strokeStyle = glow + '60';
      ctx.lineWidth = 1;
      ctx.stroke();
      
      // Dollar sign or symbol
      ctx.fillStyle = glow;
      ctx.font = `${size * 0.5}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('$', 0, 0);
      
      ctx.restore();
    };

    const drawTrophy = (ctx: CanvasRenderingContext2D, item: FloatingItem) => {
      const { x, y, size, rotation, glow, opacity } = item;
      
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.globalAlpha = opacity;
      
      // Glow
      const gradient = ctx.createRadialGradient(0, 0, size * 0.5, 0, 0, size * 1.5);
      gradient.addColorStop(0, glow + '30');
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(0, 0, size * 1.5, 0, Math.PI * 2);
      ctx.fill();
      
      // Trophy cup
      ctx.strokeStyle = glow;
      ctx.lineWidth = 2;
      
      // Cup body
      ctx.beginPath();
      ctx.moveTo(-size * 0.3, -size * 0.2);
      ctx.quadraticCurveTo(-size * 0.5, size * 0.1, -size * 0.2, size * 0.3);
      ctx.lineTo(size * 0.2, size * 0.3);
      ctx.quadraticCurveTo(size * 0.5, size * 0.1, size * 0.3, -size * 0.2);
      ctx.closePath();
      ctx.stroke();
      
      // Handles
      ctx.beginPath();
      ctx.arc(-size * 0.4, 0, size * 0.2, -Math.PI / 2, Math.PI / 2, true);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(size * 0.4, 0, size * 0.2, -Math.PI / 2, Math.PI / 2, false);
      ctx.stroke();
      
      // Stem
      ctx.beginPath();
      ctx.moveTo(0, size * 0.3);
      ctx.lineTo(0, size * 0.5);
      ctx.stroke();
      
      // Base
      ctx.beginPath();
      ctx.moveTo(-size * 0.2, size * 0.5);
      ctx.lineTo(size * 0.2, size * 0.5);
      ctx.lineTo(size * 0.25, size * 0.6);
      ctx.lineTo(-size * 0.25, size * 0.6);
      ctx.closePath();
      ctx.stroke();
      
      ctx.restore();
    };

    const drawChip = (ctx: CanvasRenderingContext2D, item: FloatingItem) => {
      const { x, y, size, rotation, glow, opacity } = item;
      
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.globalAlpha = opacity;
      
      // Glow
      const gradient = ctx.createRadialGradient(0, 0, size * 0.4, 0, 0, size);
      gradient.addColorStop(0, glow + '40');
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(0, 0, size, 0, Math.PI * 2);
      ctx.fill();
      
      // Chip body
      ctx.strokeStyle = glow;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.7, 0, Math.PI * 2);
      ctx.stroke();
      
      // Dotted edge
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2;
        const dotX = Math.cos(angle) * size * 0.55;
        const dotY = Math.sin(angle) * size * 0.55;
        ctx.beginPath();
        ctx.arc(dotX, dotY, size * 0.08, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
      }
      
      ctx.restore();
    };

    const drawBall = (ctx: CanvasRenderingContext2D, item: FloatingItem) => {
      const { x, y, size, rotation, glow, opacity } = item;
      
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.globalAlpha = opacity;
      
      // Glow
      const gradient = ctx.createRadialGradient(0, 0, size * 0.3, 0, 0, size);
      gradient.addColorStop(0, glow + '50');
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(0, 0, size, 0, Math.PI * 2);
      ctx.fill();
      
      // Ball
      ctx.strokeStyle = glow;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.5, 0, Math.PI * 2);
      ctx.stroke();
      
      // Pattern lines
      ctx.beginPath();
      ctx.moveTo(-size * 0.5, 0);
      ctx.lineTo(size * 0.5, 0);
      ctx.moveTo(0, -size * 0.5);
      ctx.lineTo(0, size * 0.5);
      ctx.stroke();
      
      ctx.restore();
    };

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw gradient background
      const bgGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      bgGradient.addColorStop(0, '#0a0e1a');
      bgGradient.addColorStop(0.5, '#0f1525');
      bgGradient.addColorStop(1, '#0a0e1a');
      ctx.fillStyle = bgGradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Update and draw items
      itemsRef.current.forEach(item => {
        // Update position
        item.x += item.speedX;
        item.y += item.speedY;
        item.rotation += item.rotationSpeed;
        
        // Wrap around edges
        if (item.x < -100) item.x = canvas.width + 100;
        if (item.x > canvas.width + 100) item.x = -100;
        if (item.y < -100) item.y = canvas.height + 100;
        if (item.y > canvas.height + 100) item.y = -100;
        
        // Draw based on type
        switch (item.type) {
          case 'coin':
            drawCoin(ctx, item);
            break;
          case 'trophy':
            drawTrophy(ctx, item);
            break;
          case 'chip':
            drawChip(ctx, item);
            break;
          case 'ball':
            drawBall(ctx, item);
            break;
        }
      });
      
      // Draw stadium lights effect (subtle)
      const lightGradient = ctx.createRadialGradient(
        canvas.width * 0.5, -100, 0,
        canvas.width * 0.5, -100, canvas.width * 0.8
      );
      lightGradient.addColorStop(0, 'rgba(0, 212, 255, 0.05)');
      lightGradient.addColorStop(0.5, 'rgba(0, 255, 136, 0.02)');
      lightGradient.addColorStop(1, 'transparent');
      ctx.fillStyle = lightGradient;
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
      style={{ background: 'linear-gradient(180deg, #0a0e1a 0%, #0f1525 50%, #0a0e1a 100%)' }}
    />
  );
}

export default FloatingBackground;
