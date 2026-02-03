/**
 * AtmosphericBackground - Sports-themed animated background
 *
 * Design Philosophy:
 * - Subtle, low opacity sports elements
 * - Static player silhouettes at various positions
 * - Floating sport items (balls, trophies, medals) drifting near players
 * - Abstract field lines for depth
 * - Premium, atmospheric feel without distraction
 * - Maintains readability of all overlaid content
 */

import { useEffect, useRef } from 'react';

// ── Types ──────────────────────────────────────────────────────────────
interface FloatingSportItem {
  x: number;
  y: number;
  baseX: number;
  baseY: number;
  size: number;
  opacity: number;
  speedX: number;
  speedY: number;
  rotation: number;
  rotationSpeed: number;
  orbitRadius: number;
  orbitAngle: number;
  orbitSpeed: number;
  type: 'football' | 'basketball' | 'tennis' | 'trophy' | 'medal' | 'whistle' | 'star' | 'goal';
}

interface PlayerSilhouette {
  x: number;
  y: number;
  scale: number;
  opacity: number;
  pose: 'kicking' | 'shooting' | 'running' | 'dribbling' | 'goalkeeper';
  mirrored: boolean;
}

interface FieldLine {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  opacity: number;
  phase: number;
}

// ── Drawing Functions ──────────────────────────────────────────────────

function drawFootball(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, opacity: number, rotation: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);

  // Ball outline
  ctx.beginPath();
  ctx.arc(0, 0, size, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(0, 212, 255, ${opacity})`;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Pentagon pattern
  for (let i = 0; i < 5; i++) {
    const angle = (i * Math.PI * 2) / 5 - Math.PI / 2;
    const px = Math.cos(angle) * size * 0.45;
    const py = Math.sin(angle) * size * 0.45;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(px, py);
    ctx.strokeStyle = `rgba(0, 212, 255, ${opacity * 0.5})`;
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  ctx.restore();
}

function drawBasketball(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, opacity: number, rotation: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);

  ctx.beginPath();
  ctx.arc(0, 0, size, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(255, 149, 0, ${opacity})`;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Vertical seam
  ctx.beginPath();
  ctx.moveTo(0, -size);
  ctx.lineTo(0, size);
  ctx.strokeStyle = `rgba(255, 149, 0, ${opacity * 0.6})`;
  ctx.lineWidth = 0.5;
  ctx.stroke();

  // Horizontal seam
  ctx.beginPath();
  ctx.moveTo(-size, 0);
  ctx.lineTo(size, 0);
  ctx.stroke();

  // Curved seams
  ctx.beginPath();
  ctx.arc(-size * 0.3, 0, size * 0.85, -Math.PI / 3, Math.PI / 3);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(size * 0.3, 0, size * 0.85, Math.PI * 2 / 3, Math.PI * 4 / 3);
  ctx.stroke();

  ctx.restore();
}

function drawTennisBall(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, opacity: number, rotation: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);

  ctx.beginPath();
  ctx.arc(0, 0, size, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(0, 255, 136, ${opacity})`;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Seam curves
  ctx.beginPath();
  ctx.arc(-size * 0.5, 0, size * 0.7, -Math.PI / 2.5, Math.PI / 2.5);
  ctx.strokeStyle = `rgba(0, 255, 136, ${opacity * 0.5})`;
  ctx.lineWidth = 0.5;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(size * 0.5, 0, size * 0.7, Math.PI - Math.PI / 2.5, Math.PI + Math.PI / 2.5);
  ctx.stroke();

  ctx.restore();
}

function drawTrophy(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, opacity: number) {
  ctx.save();
  ctx.translate(x, y);

  const color = `rgba(255, 215, 0, ${opacity})`;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;

  // Cup body
  ctx.beginPath();
  ctx.moveTo(-size * 0.5, -size * 0.6);
  ctx.quadraticCurveTo(-size * 0.55, size * 0.1, -size * 0.15, size * 0.3);
  ctx.lineTo(size * 0.15, size * 0.3);
  ctx.quadraticCurveTo(size * 0.55, size * 0.1, size * 0.5, -size * 0.6);
  ctx.stroke();

  // Cup rim
  ctx.beginPath();
  ctx.moveTo(-size * 0.55, -size * 0.6);
  ctx.lineTo(size * 0.55, -size * 0.6);
  ctx.stroke();

  // Handles
  ctx.beginPath();
  ctx.arc(-size * 0.55, -size * 0.2, size * 0.15, -Math.PI / 2, Math.PI / 2, true);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(size * 0.55, -size * 0.2, size * 0.15, -Math.PI / 2, Math.PI / 2);
  ctx.stroke();

  // Stem
  ctx.beginPath();
  ctx.moveTo(0, size * 0.3);
  ctx.lineTo(0, size * 0.55);
  ctx.stroke();

  // Base
  ctx.beginPath();
  ctx.moveTo(-size * 0.25, size * 0.55);
  ctx.lineTo(size * 0.25, size * 0.55);
  ctx.lineTo(size * 0.3, size * 0.65);
  ctx.lineTo(-size * 0.3, size * 0.65);
  ctx.closePath();
  ctx.stroke();

  ctx.restore();
}

function drawMedal(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, opacity: number, rotation: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation * 0.3);

  const color = `rgba(0, 212, 255, ${opacity})`;
  ctx.strokeStyle = color;
  ctx.lineWidth = 0.8;

  // Ribbon
  ctx.beginPath();
  ctx.moveTo(-size * 0.2, -size * 0.8);
  ctx.lineTo(-size * 0.1, -size * 0.3);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(size * 0.2, -size * 0.8);
  ctx.lineTo(size * 0.1, -size * 0.3);
  ctx.stroke();

  // Medal circle
  ctx.beginPath();
  ctx.arc(0, size * 0.05, size * 0.4, 0, Math.PI * 2);
  ctx.stroke();

  // Star in medal
  for (let i = 0; i < 5; i++) {
    const angle = (i * Math.PI * 2) / 5 - Math.PI / 2;
    const sx = Math.cos(angle) * size * 0.2;
    const sy = size * 0.05 + Math.sin(angle) * size * 0.2;
    ctx.beginPath();
    ctx.moveTo(0, size * 0.05);
    ctx.lineTo(sx, sy);
    ctx.strokeStyle = `rgba(0, 212, 255, ${opacity * 0.4})`;
    ctx.stroke();
  }

  ctx.restore();
}

function drawWhistle(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, opacity: number, rotation: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation * 0.2 + Math.PI / 6);

  ctx.strokeStyle = `rgba(148, 163, 184, ${opacity})`;
  ctx.lineWidth = 0.8;

  // Body
  ctx.beginPath();
  ctx.ellipse(0, 0, size * 0.5, size * 0.25, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Mouthpiece
  ctx.beginPath();
  ctx.moveTo(size * 0.5, -size * 0.05);
  ctx.lineTo(size * 0.8, -size * 0.05);
  ctx.lineTo(size * 0.8, size * 0.05);
  ctx.lineTo(size * 0.5, size * 0.05);
  ctx.stroke();

  // Ring
  ctx.beginPath();
  ctx.arc(-size * 0.45, -size * 0.15, size * 0.08, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
}

function drawStar(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, opacity: number, rotation: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);

  ctx.beginPath();
  for (let i = 0; i < 5; i++) {
    const outerAngle = (i * Math.PI * 2) / 5 - Math.PI / 2;
    const innerAngle = outerAngle + Math.PI / 5;
    const ox = Math.cos(outerAngle) * size;
    const oy = Math.sin(outerAngle) * size;
    const ix = Math.cos(innerAngle) * size * 0.4;
    const iy = Math.sin(innerAngle) * size * 0.4;
    if (i === 0) ctx.moveTo(ox, oy);
    else ctx.lineTo(ox, oy);
    ctx.lineTo(ix, iy);
  }
  ctx.closePath();
  ctx.strokeStyle = `rgba(255, 215, 0, ${opacity})`;
  ctx.lineWidth = 0.8;
  ctx.stroke();

  ctx.restore();
}

function drawGoalNet(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, opacity: number) {
  ctx.save();
  ctx.translate(x, y);

  ctx.strokeStyle = `rgba(0, 255, 136, ${opacity})`;
  ctx.lineWidth = 0.8;

  // Frame
  ctx.beginPath();
  ctx.moveTo(-size * 0.6, size * 0.4);
  ctx.lineTo(-size * 0.6, -size * 0.4);
  ctx.lineTo(size * 0.6, -size * 0.4);
  ctx.lineTo(size * 0.6, size * 0.4);
  ctx.stroke();

  // Net lines vertical
  const netOpacity = opacity * 0.3;
  ctx.strokeStyle = `rgba(0, 255, 136, ${netOpacity})`;
  ctx.lineWidth = 0.4;
  for (let i = 1; i < 6; i++) {
    const nx = -size * 0.6 + (size * 1.2 * i) / 6;
    ctx.beginPath();
    ctx.moveTo(nx, -size * 0.4);
    ctx.lineTo(nx, size * 0.4);
    ctx.stroke();
  }
  // Net lines horizontal
  for (let i = 1; i < 4; i++) {
    const ny = -size * 0.4 + (size * 0.8 * i) / 4;
    ctx.beginPath();
    ctx.moveTo(-size * 0.6, ny);
    ctx.lineTo(size * 0.6, ny);
    ctx.stroke();
  }

  ctx.restore();
}

// ── Player Silhouette Drawing ──────────────────────────────────────────

function drawPlayerSilhouette(
  ctx: CanvasRenderingContext2D,
  player: PlayerSilhouette
) {
  ctx.save();
  ctx.translate(player.x, player.y);
  if (player.mirrored) ctx.scale(-1, 1);
  ctx.scale(player.scale, player.scale);

  ctx.strokeStyle = `rgba(0, 212, 255, ${player.opacity})`;
  ctx.lineWidth = 1.5 / player.scale;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  switch (player.pose) {
    case 'kicking':
      drawKickingPlayer(ctx, player.opacity);
      break;
    case 'shooting':
      drawShootingPlayer(ctx, player.opacity);
      break;
    case 'running':
      drawRunningPlayer(ctx, player.opacity);
      break;
    case 'dribbling':
      drawDribblingPlayer(ctx, player.opacity);
      break;
    case 'goalkeeper':
      drawGoalkeeperPlayer(ctx, player.opacity);
      break;
  }

  ctx.restore();
}

function drawHead(ctx: CanvasRenderingContext2D, x: number, y: number, radius: number) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.stroke();
}

function drawKickingPlayer(ctx: CanvasRenderingContext2D, opacity: number) {
  // Head
  drawHead(ctx, 0, -65, 8);

  // Body (leaning back slightly)
  ctx.beginPath();
  ctx.moveTo(0, -57);
  ctx.lineTo(-5, -20);
  ctx.stroke();

  // Left arm (back for balance)
  ctx.beginPath();
  ctx.moveTo(-3, -50);
  ctx.lineTo(-25, -40);
  ctx.lineTo(-30, -48);
  ctx.stroke();

  // Right arm (forward)
  ctx.beginPath();
  ctx.moveTo(-2, -48);
  ctx.lineTo(15, -55);
  ctx.stroke();

  // Standing leg
  ctx.beginPath();
  ctx.moveTo(-5, -20);
  ctx.lineTo(-10, 10);
  ctx.lineTo(-8, 20);
  ctx.stroke();

  // Kicking leg (extended)
  ctx.beginPath();
  ctx.moveTo(-5, -20);
  ctx.lineTo(20, -15);
  ctx.lineTo(35, -20);
  ctx.stroke();
}

function drawShootingPlayer(ctx: CanvasRenderingContext2D, opacity: number) {
  // Head
  drawHead(ctx, 0, -70, 8);

  // Body
  ctx.beginPath();
  ctx.moveTo(0, -62);
  ctx.lineTo(0, -25);
  ctx.stroke();

  // Arms raised (basketball shot)
  ctx.beginPath();
  ctx.moveTo(-2, -52);
  ctx.lineTo(12, -65);
  ctx.lineTo(15, -78);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(2, -52);
  ctx.lineTo(-10, -58);
  ctx.stroke();

  // Legs (slight squat)
  ctx.beginPath();
  ctx.moveTo(0, -25);
  ctx.lineTo(-10, -5);
  ctx.lineTo(-8, 15);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(0, -25);
  ctx.lineTo(10, -5);
  ctx.lineTo(12, 15);
  ctx.stroke();
}

function drawRunningPlayer(ctx: CanvasRenderingContext2D, opacity: number) {
  // Head
  drawHead(ctx, 5, -65, 8);

  // Body (leaning forward)
  ctx.beginPath();
  ctx.moveTo(5, -57);
  ctx.lineTo(-2, -20);
  ctx.stroke();

  // Arms pumping
  ctx.beginPath();
  ctx.moveTo(2, -48);
  ctx.lineTo(18, -38);
  ctx.lineTo(20, -30);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(0, -45);
  ctx.lineTo(-18, -50);
  ctx.lineTo(-22, -42);
  ctx.stroke();

  // Front leg (extended)
  ctx.beginPath();
  ctx.moveTo(-2, -20);
  ctx.lineTo(12, 0);
  ctx.lineTo(18, 15);
  ctx.stroke();

  // Back leg (behind)
  ctx.beginPath();
  ctx.moveTo(-2, -20);
  ctx.lineTo(-18, -8);
  ctx.lineTo(-22, 5);
  ctx.stroke();
}

function drawDribblingPlayer(ctx: CanvasRenderingContext2D, opacity: number) {
  // Head (looking down slightly)
  drawHead(ctx, 3, -62, 8);

  // Body (slight lean forward)
  ctx.beginPath();
  ctx.moveTo(3, -54);
  ctx.lineTo(0, -18);
  ctx.stroke();

  // Arms (one extended for balance)
  ctx.beginPath();
  ctx.moveTo(1, -46);
  ctx.lineTo(18, -40);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(1, -44);
  ctx.lineTo(-12, -35);
  ctx.lineTo(-15, -28);
  ctx.stroke();

  // Legs (mid stride)
  ctx.beginPath();
  ctx.moveTo(0, -18);
  ctx.lineTo(8, 2);
  ctx.lineTo(5, 18);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(0, -18);
  ctx.lineTo(-10, 0);
  ctx.lineTo(-6, 15);
  ctx.stroke();
}

function drawGoalkeeperPlayer(ctx: CanvasRenderingContext2D, opacity: number) {
  // Head
  drawHead(ctx, 0, -68, 8);

  // Body (wide stance)
  ctx.beginPath();
  ctx.moveTo(0, -60);
  ctx.lineTo(0, -22);
  ctx.stroke();

  // Arms spread wide
  ctx.beginPath();
  ctx.moveTo(-2, -50);
  ctx.lineTo(-28, -55);
  ctx.lineTo(-32, -60);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(2, -50);
  ctx.lineTo(28, -55);
  ctx.lineTo(32, -60);
  ctx.stroke();

  // Legs (wide squat)
  ctx.beginPath();
  ctx.moveTo(0, -22);
  ctx.lineTo(-15, 0);
  ctx.lineTo(-18, 18);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(0, -22);
  ctx.lineTo(15, 0);
  ctx.lineTo(18, 18);
  ctx.stroke();
}

// ── Main Component ─────────────────────────────────────────────────────

export function AtmosphericBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playersRef = useRef<PlayerSilhouette[]>([]);
  const itemsRef = useRef<FloatingSportItem[]>([]);
  const linesRef = useRef<FieldLine[]>([]);
  const animationRef = useRef<number>(0);
  const timeRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initElements();
    };

    const initElements = () => {
      const w = canvas.width;
      const h = canvas.height;

      // ── Player silhouettes at fixed positions ──
      const playerConfigs: PlayerSilhouette[] = [
        { x: w * 0.08, y: h * 0.7, scale: 1.2, opacity: 0.04, pose: 'kicking', mirrored: false },
        { x: w * 0.92, y: h * 0.35, scale: 1.0, opacity: 0.035, pose: 'shooting', mirrored: true },
        { x: w * 0.5, y: h * 0.85, scale: 0.9, opacity: 0.03, pose: 'running', mirrored: false },
        { x: w * 0.75, y: h * 0.75, scale: 1.1, opacity: 0.035, pose: 'dribbling', mirrored: true },
        { x: w * 0.2, y: h * 0.3, scale: 1.0, opacity: 0.03, pose: 'goalkeeper', mirrored: false },
      ];
      playersRef.current = playerConfigs;

      // ── Floating sport items ──
      const sportItems: FloatingSportItem[] = [];
      const itemTypes: FloatingSportItem['type'][] = [
        'football', 'basketball', 'tennis', 'trophy', 'medal', 'whistle', 'star', 'goal'
      ];

      // Items that float freely across the canvas
      const freeCount = Math.min(12, Math.floor(w / 160));
      for (let i = 0; i < freeCount; i++) {
        sportItems.push({
          x: Math.random() * w,
          y: Math.random() * h,
          baseX: Math.random() * w,
          baseY: Math.random() * h,
          size: 8 + Math.random() * 14,
          opacity: 0.025 + Math.random() * 0.035,
          speedX: (Math.random() - 0.5) * 0.12,
          speedY: (Math.random() - 0.5) * 0.08,
          rotation: Math.random() * Math.PI * 2,
          rotationSpeed: (Math.random() - 0.5) * 0.003,
          orbitRadius: 0,
          orbitAngle: 0,
          orbitSpeed: 0,
          type: itemTypes[i % itemTypes.length],
        });
      }

      // Items that orbit near player silhouettes
      playerConfigs.forEach((player, pi) => {
        const nearCount = 2 + Math.floor(Math.random() * 2);
        for (let j = 0; j < nearCount; j++) {
          const angle = (j * Math.PI * 2) / nearCount + Math.random() * 0.5;
          const radius = 40 + Math.random() * 60;
          sportItems.push({
            x: player.x + Math.cos(angle) * radius,
            y: player.y + Math.sin(angle) * radius,
            baseX: player.x,
            baseY: player.y,
            size: 6 + Math.random() * 10,
            opacity: 0.03 + Math.random() * 0.03,
            speedX: 0,
            speedY: 0,
            rotation: Math.random() * Math.PI * 2,
            rotationSpeed: (Math.random() - 0.5) * 0.004,
            orbitRadius: radius,
            orbitAngle: angle,
            orbitSpeed: 0.0003 + Math.random() * 0.0005,
            type: itemTypes[Math.floor(Math.random() * itemTypes.length)],
          });
        }
      });

      itemsRef.current = sportItems;

      // ── Abstract field lines ──
      const lines: FieldLine[] = [];
      const hLineCount = Math.floor(h / 140);
      for (let i = 0; i < hLineCount; i++) {
        lines.push({
          x1: 0, y1: i * 140 + 70,
          x2: w, y2: i * 140 + 70,
          opacity: 0.02 + Math.random() * 0.02,
          phase: Math.random() * Math.PI * 2,
        });
      }
      // Center circle
      lines.push({
        x1: w * 0.5, y1: h * 0.5,
        x2: 0, y2: 0, // x2 used as radius marker
        opacity: 0.02,
        phase: 0,
      });
      // Vertical lines
      [0.25, 0.5, 0.75].forEach((pos) => {
        lines.push({
          x1: w * pos, y1: 0,
          x2: w * pos, y2: h,
          opacity: 0.015,
          phase: Math.random() * Math.PI * 2,
        });
      });
      linesRef.current = lines;
    };

    resize();
    window.addEventListener('resize', resize);

    // ── Animation Loop ──
    const animate = () => {
      timeRef.current += 0.001;
      const t = timeRef.current;
      const w = canvas.width;
      const h = canvas.height;

      // Clear
      ctx.fillStyle = '#0a0e1a';
      ctx.fillRect(0, 0, w, h);

      // Subtle gradient overlay
      const gradient = ctx.createLinearGradient(0, 0, 0, h);
      gradient.addColorStop(0, 'rgba(10, 14, 26, 0)');
      gradient.addColorStop(0.5, 'rgba(15, 22, 35, 0.3)');
      gradient.addColorStop(1, 'rgba(10, 14, 26, 0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, w, h);

      // ── Draw field lines ──
      linesRef.current.forEach((line, idx) => {
        // Center circle (special case: last 4 lines include circle + verticals)
        if (line.x2 === 0 && line.y2 === 0) {
          const breathe = Math.sin(t * 0.8) * 5;
          ctx.beginPath();
          ctx.arc(line.x1, line.y1, 80 + breathe, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(0, 212, 255, ${line.opacity})`;
          ctx.lineWidth = 1;
          ctx.stroke();
          return;
        }

        const offset = Math.sin(t + line.phase) * 15;
        ctx.beginPath();
        if (line.x1 === 0) {
          ctx.moveTo(line.x1, line.y1 + offset);
          ctx.lineTo(line.x2, line.y2 + offset);
        } else {
          ctx.moveTo(line.x1 + offset * 0.5, line.y1);
          ctx.lineTo(line.x2 + offset * 0.5, line.y2);
        }
        ctx.strokeStyle = `rgba(0, 212, 255, ${line.opacity})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // ── Draw player silhouettes (static) ──
      playersRef.current.forEach((player) => {
        // Subtle glow behind player
        const glowGrad = ctx.createRadialGradient(
          player.x, player.y - 30 * player.scale, 0,
          player.x, player.y - 30 * player.scale, 60 * player.scale
        );
        glowGrad.addColorStop(0, `rgba(0, 212, 255, ${player.opacity * 0.3})`);
        glowGrad.addColorStop(1, 'rgba(0, 212, 255, 0)');
        ctx.fillStyle = glowGrad;
        ctx.fillRect(
          player.x - 70 * player.scale,
          player.y - 90 * player.scale,
          140 * player.scale,
          140 * player.scale
        );

        drawPlayerSilhouette(ctx, player);
      });

      // ── Draw floating sport items ──
      itemsRef.current.forEach((item) => {
        // Update position
        if (item.orbitRadius > 0) {
          // Orbiting near a player
          item.orbitAngle += item.orbitSpeed;
          item.x = item.baseX + Math.cos(item.orbitAngle) * item.orbitRadius;
          item.y = item.baseY + Math.sin(item.orbitAngle) * item.orbitRadius;
        } else {
          // Free floating
          item.x += item.speedX;
          item.y += item.speedY;

          // Wrap around
          if (item.x < -50) item.x = w + 50;
          if (item.x > w + 50) item.x = -50;
          if (item.y < -50) item.y = h + 50;
          if (item.y > h + 50) item.y = -50;
        }

        item.rotation += item.rotationSpeed;

        // Subtle pulsing opacity
        const pulseOpacity = item.opacity * (0.8 + 0.2 * Math.sin(t * 2 + item.rotation));

        switch (item.type) {
          case 'football':
            drawFootball(ctx, item.x, item.y, item.size, pulseOpacity, item.rotation);
            break;
          case 'basketball':
            drawBasketball(ctx, item.x, item.y, item.size, pulseOpacity, item.rotation);
            break;
          case 'tennis':
            drawTennisBall(ctx, item.x, item.y, item.size, pulseOpacity, item.rotation);
            break;
          case 'trophy':
            drawTrophy(ctx, item.x, item.y, item.size, pulseOpacity);
            break;
          case 'medal':
            drawMedal(ctx, item.x, item.y, item.size, pulseOpacity, item.rotation);
            break;
          case 'whistle':
            drawWhistle(ctx, item.x, item.y, item.size, pulseOpacity, item.rotation);
            break;
          case 'star':
            drawStar(ctx, item.x, item.y, item.size, pulseOpacity, item.rotation);
            break;
          case 'goal':
            drawGoalNet(ctx, item.x, item.y, item.size, pulseOpacity);
            break;
        }
      });

      // ── Vignette effect ──
      const vignette = ctx.createRadialGradient(
        w / 2, h / 2, 0,
        w / 2, h / 2, w * 0.8
      );
      vignette.addColorStop(0, 'rgba(10, 14, 26, 0)');
      vignette.addColorStop(1, 'rgba(10, 14, 26, 0.5)');
      ctx.fillStyle = vignette;
      ctx.fillRect(0, 0, w, h);

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

export default AtmosphericBackground;
