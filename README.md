# GOLD AI — Live Multi-Agent Gold Market Intelligence & Forecasting Terminal

A professional Python desktop application for real-time XAU/USD analysis using a multi-agent system that collects evidence from multiple independent sources, detects market regimes, calibrates probabilities, and generates transparent final decisions.

**IMPORTANT**: This application is designed for research, education, backtesting, and paper/simulation analysis only. It does NOT implement automatic real-money order execution or broker order placement.

## 🎯 Core Features

### Multi-Agent Architecture
- **35+ Specialized Agents**: Each analyzing different aspects of the gold market
  - Technical: Trend, Momentum, Price Action, Volatility, Support/Resistance
  - Macro: USD/DXY, Treasury Yields, Real Yields, Fed Policy, Economic Calendar
  - News & Sentiment: NLP-powered news intelligence, geopolitical risk
  - Positioning: CFTC COT data, ETF flows, futures structure
  - ML Ensemble: Logistic Regression, Random Forest, XGBoost, LightGBM
  - Meta-Agents: Regime detection, calibration, drift monitoring, conflict detection

### Live Analysis Pipeline
```
Live Market Data → Specialized Agents → Regime Detection → Evidence Validation 
→ Dynamic Weighting → Conflict Detection → Calibrated Probabilities 
→ Scenario Engine → Final Decision → Continuous Live Update
```

### Key Principles
- ✅ **Evidence-Based**: Every decision backed by traceable evidence
- ✅ **Calibrated Probabilities**: Not raw model outputs, but properly calibrated
- ✅ **Dynamic Weighting**: Agent importance changes with market regime
- ✅ **Conflict Detection**: Identifies when agents disagree
- ✅ **Double-Counting Control**: Prevents counting same evidence multiple times
- ✅ **Freshness Tracking**: All data sources show latency and reliability
- ✅ **Abstention System**: WAIT is a valid output when confidence is low
- ✅ **Anti-Leakage**: Strict timestamp controls prevent future data leakage
- ✅ **Full Explainability**: Every decision includes detailed reasoning

## 📊 What The System Answers

1. **What is happening now?** - Live price, regime, session state
2. **What are the strongest forces?** - Ranked agent contributions
3. **Which agents are bullish/bearish/neutral?** - Agent consensus breakdown
4. **Are agents independent?** - Evidence correlation analysis
5. **What market regime?** - Trending, ranging, high-vol, macro-driven, etc.
6. **What event is approaching?** - Countdown to FOMC, CPI, NFP, etc.
7. **How strong is the evidence?** - Confidence scores (0-100)
8. **Probability estimates?** - P(UP), P(DOWN), P(NEUTRAL) per horizon
9. **Is probability reliable?** - Calibration quality metrics
10. **Any conflicts?** - Directional/evidence/source/timeframe conflicts
11. **Should we abstain?** - WAIT/NO VALID SIGNAL when appropriate
12. **Why did decision change?** - Complete decision timeline with triggers

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

#### Option 1: Automated Install (Windows)
```bash
install.bat
```

#### Option 2: Manual Install
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### Development Mode
```bash
python main.py
```

#### With Custom Configuration
```bash
python main.py --config config/settings.yaml
```

#### Backtesting Mode
```bash
python main.py --mode backtest
```

#### Headless Mode (No GUI)
```bash
python main.py --headless
```

## 📦 Building Executable (.exe)

### Using PyInstaller

```bash
# Windows automated build
build_exe.bat

# Or manual build
pyinstaller --onefile --windowed --name "GOLD_AI" main.py
```

The executable will be created at `dist/GOLD_AI.exe`

See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for detailed packaging guide.

## 🏗️ Architecture

```
gold_ai/
├── app/                    # Application orchestration
│   └── orchestrator.py     # Main event loop and coordination
├── core/                   # Core utilities
│   ├── enums.py            # Enumerations (regimes, states, directions)
│   ├── models.py           # Data models (candles, events, predictions)
│   ├── state.py            # Global state management
│   ├── event_bus.py        # Event-driven communication
│   └── timeutils.py        # UTC timestamp handling
├── data/                   # Data layer
│   ├── providers/          # External data source adapters
│   ├── normalization/      # Schema standardization
│   └── cache/              # Local caching
├── agents/                 # 35+ specialized agents
│   ├── data_integrity/     # Feed health monitoring
│   ├── trend/              # EMA/SMA/multi-TF analysis
│   ├── momentum/           # RSI/MACD/ROC/divergence
│   ├── volatility/         # ATR/regime/shock detection
│   ├── usd/                # DXY analysis
│   ├── rates/              # Treasury yields
│   ├── fed/                # Monetary policy NLP
│   ├── news/               # NLP news intelligence
│   ├── regime/             # Market regime classification
│   ├── ml/                 # Machine learning ensemble
│   ├── calibration/        # Probability calibration
│   ├── conflict/           # Agent disagreement detection
│   └── ...                 # Many more specialized agents
├── engine/                 # Decision engines
│   ├── weighting/          # Dynamic agent weighting
│   ├── evidence/           # Dependency graph, double-count control
│   ├── scenario/           # Multi-scenario generation
│   ├── decision/           # Final decision logic
│   └── validation/         # Anti-leakage checks
├── models/                 # ML model registry
├── backtest/               # Historical evaluation
├── database/               # Storage layer
├── ui/                     # PyQt6 terminal interface
├── services/               # Background services
├── tests/                  # Unit and integration tests
├── config/                 # Configuration files
└── logs/                   # Structured logging
```

## ⚙️ Configuration

Edit `config/settings.yaml` to customize:

- **Data Providers**: Enable/disable external APIs
- **Agent Settings**: Tune parameters, enable/disable specific agents
- **Weighting Rules**: Define how regimes affect agent importance
- **Decision Thresholds**: Configure confidence levels and abstention rules
- **UI Preferences**: Theme, layout, update rates
- **Backtest Parameters**: Walk-forward settings, friction model

### Environment Variables

Copy `.env.example` to `.env` and fill in API keys if using external providers:

```bash
NEWSAPI_KEY=your_key_here
FRED_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
```

**Security Note**: Never commit `.env` to version control (already in `.gitignore`).

## 📈 Understanding The Output

### Final Decision Card Example

```
XAU/USD LIVE PRICE: 4325.70

REGIME: HIGH VOLATILITY / MACRO-DRIVEN
DIRECTION: BULLISH
PROBABILITY: UP 78% | DOWN 16% | NEUTRAL 6%
CONFIDENCE: 81/100
SIGNAL QUALITY: A-

AGENT CONSENSUS:
• 14 Bullish
• 5 Bearish  
• 3 Neutral

DATA QUALITY: 96%
EVENT RISK: HIGH
NEWS RISK: MEDIUM
VOLATILITY: HIGH

TOP BULLISH DRIVERS:
1. USD weakness
2. Trend alignment
3. Real-yield pressure
4. Positive cross-asset confirmation
5. Positive momentum

TOP BEARISH DRIVERS:
1. Major resistance nearby
2. Elevated volatility
3. Event uncertainty

CONFLICT: MEDIUM
SCENARIOS: Bullish 62% | Neutral 18% | Bearish 15% | Shock 5%

FINAL STATE: BULLISH — PAPER/SIMULATION ANALYSIS
```

### When To Trust The Signal

**High Confidence (85+)**: Very strong evidence alignment, good data quality, known regime

**Moderate Confidence (65-85)**: Reasonable signal with some conflicting factors

**Low Confidence (<65)**: Weak evidence, consider waiting

**WAIT State**: Insufficient confidence, poor data quality, extreme conflict, or high event risk

## 🔬 Scientific Validation

### Out-of-Sample Performance Metrics

The system tracks:
- Accuracy, Precision, Recall, F1, Balanced Accuracy
- Brier Score (probability calibration quality)
- Log Loss
- ROC-AUC
- Calibration Error
- Performance by Regime
- Performance by Horizon
- Performance by Confidence Bucket

### Walk-Forward Validation

```python
# Train on past, test on future, roll forward
train_window = 365 days
test_window = 30 days
step_size = 7 days
embargo = 5 days  # Prevent overlap leakage
```

### Friction Model

Realistic backtesting includes:
- Spread costs (30 bps default)
- Slippage (10 bps default)
- Latency (1 second default)
- Missing ticks simulation
- Delayed information handling

## 🛡️ Failure Handling

The application remains stable when:
- ❌ One data provider fails → Others continue, confidence reduced
- ❌ News API offline → News agent marked STALE
- ❌ Network disconnects → Uses cached data, shows warning
- ❌ Chart feed stale → Data Integrity agent blocks decision
- ❌ One agent crashes → Remaining agents continue
- ❌ ML model unavailable → Falls back to simpler baseline

All failures are clearly displayed:
- `DATA DEGRADED`
- `NEWS OFFLINE`
- `MACRO STALE`
- `MODEL UNAVAILABLE`

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=gold_ai --cov-report=html

# Specific test module
pytest tests/test_trend_agent.py -v
```

## 📚 Documentation

- **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** - Packaging and deployment guide
- **[config/settings.yaml](config/settings.yaml)** - Full configuration reference
- **Inline Code Docs** - Comprehensive docstrings throughout

## 🎓 Educational Value

This project demonstrates:
- Multi-agent system design
- Event-driven architecture
- Financial data engineering
- Machine learning pipeline best practices
- Probability calibration
- Time-series validation techniques
- Professional UI/UX for financial applications
- Production-quality Python patterns

## ⚠️ Important Disclaimers

1. **NOT Trading Advice**: For research/education only
2. **No Auto-Trading**: Does not place orders with any broker
3. **Past Performance ≠ Future Results**: Even validated models can fail
4. **Market Risk**: Gold prices can move unpredictably
5. **Data Quality Matters**: Garbage in = garbage out
6. **Model Limitations**: All models have blind spots

## 🤝 Contributing

Contributions welcome! Please ensure:
- Tests pass before submitting PR
- Follow existing code style
- Add documentation for new features
- Maintain architectural principles

## 📄 License

[To be determined by project owner]

## 🆘 Support

For issues or questions:
- Check logs directory: `data/logs/`
- Review configuration: `config/settings.yaml`
- Open GitHub issue with error details

---

**Built with scientific rigor. Designed for transparency. Optimized for trust.**
