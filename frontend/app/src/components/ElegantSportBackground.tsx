import { useEffect, useRef } from 'react';

interface Shape {
  id: number;
  x: number;
  y: number;
  size: number;
  opacity: number;
  speedX: number;
  speedY: number;
  rotation: number;
  type: 'line' | 'circle' | 'hex';
}

export function ElegantSportBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shapesRef = useRef<Shape[]>([]);
  const animRef = useRef<number>(0);

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

    const shapes: Shape[] = [];
    for (let i = 0; i < 10; i++) {
      shapes.push({
        id: i,
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: 80 + Math.random() * 150,
        opacity: 0.04 + Math.random() * 0.04,
        speedX: (Math.random() - 0.5) * 0.03,
        speedY: (Math.random() - 0.5) * 0.03,
        rotation: Math.random() * Math.PI * 2,
        type: ['line', 'circle', 'hex'][Math.floor(Math.random() * 3)] as Shape['type'],
      });
    }
    shapesRef.current = shapes;

    const drawLine = (s: Shape) => {
      ctx.save();
      ctx.translate(s.x, s.y);
      ctx.rotate(s.rotation);
      ctx.globalAlpha = s.opacity;
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(-s.size, 0);
      ctx.lineTo(s.size, 0);
      ctx.stroke();
      ctx.restore();
    };

    const drawCircle = (s: Shape) => {
      ctx.save();
      ctx.globalAlpha = s.opacity;
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.size * 0.4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };

    const drawHex = (s: Shape) => {
      ctx.save();
      ctx.translate(s.x, s.y);
      ctx.rotate(s.rotation);
      ctx.globalAlpha = s.opacity;
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = (i * Math.PI) / 3;
        const px = Math.cos(a) * s.size * 0.4;
        const py = Math.sin(a) * s.size * 0.4;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    };

    const animate = () => {
      ctx.fillStyle = '#0a0e1a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      shapesRef.current.forEach(s => {
        s.x += s.speedX;
        s.y += s.speedY;
        if (s.x < -200) s.x = canvas.width + 200;
        if (s.x > canvas.width + 200) s.x = -200;
        if (s.y < -200) s.y = canvas.height + 200;
        if (s.y > canvas.height + 200) s.y = -200;

        if (s.type === 'line') drawLine(s);
        else if (s.type === 'circle') drawCircle(s);
        else drawHex(s);
      });

      animRef.current = requestAnimationFrame(animate);
    };

    animate();
    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" />;
}

export default ElegantSportBackground;
