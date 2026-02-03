# NEXUS AI Frontend Architecture

## Overview

Professional sports prediction analytics platform with a trading terminal aesthetic. Built with React, TypeScript, Tailwind CSS, and Recharts.

---

## Design Philosophy

### Anti-Carnival Rule
The interface strictly avoids:
- Flashy visuals and loud colors
- Big decorative graphics, emojis, stickers
- Cartoonish icons or playful elements
- Bouncing/fast animations
- Gradients everywhere or visual noise

### Core Principles
- **Professional minimalism > decoration**
- **Subtle > flashy**
- **Data > graphics**
- **Clarity > effects**

The product must feel like:
- Trading terminal
- Analytics dashboard
- BI tool
- Data science product

NOT like:
- Betting advertisement
- Sports news portal
- Gaming app
- Flashy entertainment UI

---

## Color System

### Primary Colors
```
Background:    #0a0e1a (deep navy)
Card BG:       #0f1623 (slightly lighter)
Primary:       #00d4ff (cyan)
Success:       #00ff88 (green)
Warning:       #ff9500 (amber)
Danger:        #ff3860 (red)
Text Primary:  #ffffff
Text Secondary:#94a3b8 (slate-400)
Text Muted:    #64748b (slate-500)
Border:        rgba(255,255,255,0.1)
```

### Usage Guidelines
- Use cyan (#00d4ff) for primary actions, links, highlights
- Use green (#00ff88) for positive metrics, wins, value indicators
- Use amber (#ff9500) for warnings, medium confidence
- Use red (#ff3860) for losses, live indicators, errors
- Keep backgrounds dark with subtle transparency layers

---

## Navigation Structure

```
/app                    Dashboard (overview)
/app/predictions        Predictions (detailed analysis)
/app/matches            Matches (all matches by league)
/app/handicaps          Handicaps (Asian/European lines)
/app/statistics         Statistics (team/player stats)
/app/reports            Reports (analytics & insights)
/app/models             Model Performance (ML metrics)
/app/history            History (prediction results)
/app/settings           Settings (preferences)
/app/docs               Documentation
```

---

## Component Architecture

### Layout Components (`/components/layout`)
```typescript
AppShell        // Main shell with sidebar navigation
PageLayout      // Page wrapper with breadcrumbs, title, actions
```

### UI Components (`/components/ui`)
Standard shadcn/ui components:
- Button, Card, Badge, Input, Select, Tabs, Table
- Dialog, Dropdown, Tooltip, etc.

### Analytics Components (`/components/analytics`)
```typescript
KPICard           // Key performance indicator card
ProbabilityBar    // Visual probability display
ConfidenceGauge   // Circular confidence indicator
EdgeIndicator     // Edge/value indicator
InsightCard       // Analytical insight display
ModelMetrics      // ML model performance display
RiskIndicator     // Risk level indicator
StreakIndicator   // Win/loss streak display
DataTable         // Professional data table
```

### UI Pro Components (`/components/ui-pro`)
```typescript
InsightPanel      // AI-generated insights with explanations
StatComparison    // Side-by-side statistical comparison
TrendIndicator    // Visual trend direction indicator
EmptyState        // Professional empty state
SkeletonCard      // Loading skeletons
```

---

## Page Structure

### DashboardPage
- KPI cards (predictions, live matches, win rate, ROI)
- Live matches with real-time scores
- Upcoming predictions table
- Hot predictions sidebar
- Top leagues performance

### PredictionsPage
- Search and filter bar
- Confidence threshold slider
- Value bet toggle
- Expandable prediction cards with:
  - Key factors breakdown
  - Bookmaker odds comparison
  - Similar historical matches
  - Edge calculation

### HandicapsPage
- Match selector
- Asian Handicap matrix
- European Handicap display
- Over/Under lines
- Line movement chart (24h)
- Historical ROI by handicap type

### ReportsPage
- Performance overview KPIs
- AI-generated insights panel
- Daily performance chart
- Confidence distribution pie chart
- Sport performance breakdown
- Model accuracy radar
- Calibration analysis

### ModelsPage
- Model cards with metrics
- A/B testing results table
- Feature importance chart
- Training history graph
- Calibration scatter plot
- Prediction distribution

### StatisticsPage
- League selector
- Standings table with xG
- Team analysis radar chart
- Attack/Defense/Possession stats
- Head-to-head history
- H2H summary statistics

### HistoryPage
- Summary statistics cards
- Searchable prediction table
- Monthly performance charts
- Cumulative P/L graph
- Model breakdown table
- Pagination

### SettingsPage
- Regional settings (language, timezone, odds format)
- Notification preferences
- Model configuration
- Display options
- Data export/import
- API configuration

---

## Styling Guidelines

### Tailwind Patterns
```typescript
// Card styling
className="bg-[#0f1623]/80 border-white/10"

// Hover states
className="hover:bg-white/5 hover:border-white/20"

// Active/selected states
className="bg-[#00d4ff]/10 border-[#00d4ff]/20 text-[#00d4ff]"

// Typography hierarchy
// H1: text-2xl font-bold text-white
// H2: text-lg font-medium text-white
// Body: text-sm text-gray-400
// Muted: text-xs text-gray-500
```

### Animation Guidelines
- Use `transition-all duration-200` for hover states
- Use `animate-fade-in` for content appearance
- Use `animate-slide-up` for list items
- Keep animations subtle and professional
- Avoid bouncing, rotating, or flashy animations

---

## Handicap Support

### Asian Handicap Lines
- Full line matrix (-1.5, -1, -0.5, 0, +0.5, +1, +1.5)
- Home/Away odds display
- Probability bars
- Edge indicators
- Line movement indicators
- Volume data

### European Handicap
- Line format (0:0, 1:0, 0:1)
- Three-way odds (Home/Draw/Away)
- Probability distribution

### Over/Under
- Multiple line options (2.0, 2.5, 3.0, 3.5)
- Over/Under odds
- Probability comparison

---

## Reports & Analytics Features

### Key Metrics
- Total predictions count
- Win rate percentage
- ROI calculation
- Average confidence
- Streak tracking

### Visualizations
- Area charts for profit/loss
- Bar charts for volume
- Pie charts for distribution
- Radar charts for model metrics
- Scatter plots for calibration
- Line charts for trends

### Insights
- AI-generated text explanations
- Factor impact breakdown
- Similar match comparison
- Risk assessment
- Value identification

---

## Best Practices

### Component Development
1. Use TypeScript for all components
2. Export props interfaces
3. Provide default exports
4. Use composition patterns
5. Keep components focused and single-purpose

### State Management
1. Use React hooks for local state
2. Lift state when needed for sharing
3. Use controlled components
4. Implement proper loading states

### Performance
1. Use memo for expensive calculations
2. Lazy load routes if needed
3. Optimize re-renders
4. Use proper key props in lists

### Accessibility
1. Proper heading hierarchy
2. ARIA labels where needed
3. Keyboard navigation support
4. Sufficient color contrast

---

## File Structure

```
frontend/app/src/
├── components/
│   ├── layout/           # App shell, page layouts
│   ├── ui/              # shadcn/ui components
│   ├── analytics/       # Data visualization components
│   ├── ui-pro/          # Advanced UI components
│   └── ...              # Other components
├── pages/
│   ├── app/             # Authenticated app pages
│   ├── LandingPage.tsx
│   ├── SignInPage.tsx
│   └── SignUpPage.tsx
├── hooks/               # Custom React hooks
├── lib/                 # Utilities, API clients
├── styles/              # Additional CSS files
├── App.tsx
├── main.tsx
└── index.css
```

---

## Adding New Features

1. **New Page**: Create in `/pages/app/`, add to `index.ts`, add route in `App.tsx`
2. **New Component**: Create in appropriate `/components/` subfolder, export from `index.ts`
3. **New Analytics**: Add to `/components/analytics/` with consistent styling
4. **New Chart**: Use Recharts with consistent color scheme and tooltip styling

---

## Dependencies

```json
{
  "react": "^18.x",
  "react-router-dom": "^6.x",
  "tailwindcss": "^3.x",
  "recharts": "^2.x",
  "lucide-react": "latest",
  "@radix-ui/*": "various",
  "class-variance-authority": "latest",
  "clsx": "latest",
  "tailwind-merge": "latest"
}
```
