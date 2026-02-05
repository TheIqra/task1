from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime
import json


app = FastAPI(
    title="Stock Trading Simulator",
    description="Web-based algorithmic trading simulator using Golden Cross strategy",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class SimulationRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    budget: float = 10000


class TradeLog(BaseModel):
    date: str
    action: str
    price: float
    shares: int
    balance: float


class ChartDataPoint(BaseModel):
    date: str
    close: float
    ma50: Optional[float] = None
    ma200: Optional[float] = None


class SimulationResponse(BaseModel):
    success: bool
    symbol: str
    initial_budget: float
    final_balance: float
    total_profit_loss: float
    percent_return: float
    total_trades: int
    trades: List[TradeLog]
    chart_data: List[ChartDataPoint]
    message: str


class AlgorithmicTrader:
    """Trading algorithm using Golden Cross / Death Cross strategy"""
    
    def __init__(self, symbol: str, from_date: str, to_date: str, budget: float = 10000):
        self.symbol = symbol.upper()
        self.from_date = from_date
        self.to_date = to_date
        self.budget = budget
        self.initial_budget = budget
        
        self.shares_owned = 0
        self.in_position = False
        self.buy_price = 0
        self.trades = []
        self.data = None
    
    def download_data(self) -> bool:
        """Download historical stock data"""
        try:
            self.data = yf.download(self.symbol, start=self.from_date, end=self.to_date, progress=False)
            
            if self.data.empty:
                return False
            
            # Fix multi-index columns
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.get_level_values(0)
            
            return True
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
    
    def clean_data(self):
        """Clean the data - remove duplicates and fill missing values"""
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        self.data = self.data.ffill()
    
    def calculate_moving_averages(self):
        """Calculate 50-day and 200-day moving averages"""
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA_200'] = self.data['Close'].rolling(window=200).mean()
        self.data = self.data.dropna()
    
    def identify_signals(self):
        """Identify buy/sell signals based on Golden Cross / Death Cross"""
        self.data['Signal'] = 0
        
        for i in range(1, len(self.data)):
            current_idx = self.data.index[i]
            prev_idx = self.data.index[i-1]
            
            ma_50_current = self.data.loc[current_idx, 'MA_50']
            ma_200_current = self.data.loc[current_idx, 'MA_200']
            ma_50_prev = self.data.loc[prev_idx, 'MA_50']
            ma_200_prev = self.data.loc[prev_idx, 'MA_200']
            
            # Golden Cross - BUY signal
            if ma_50_prev <= ma_200_prev and ma_50_current > ma_200_current:
                self.data.loc[current_idx, 'Signal'] = 1
            
            # Death Cross - SELL signal
            elif ma_50_prev >= ma_200_prev and ma_50_current < ma_200_current:
                self.data.loc[current_idx, 'Signal'] = -1
    
    def execute_trades(self):
        """Execute trades based on signals"""
        for i, row in self.data.iterrows():
            current_price = float(row['Close'])
            signal = row['Signal']
            
            # BUY signal
            if signal == 1 and not self.in_position:
                shares_to_buy = int(self.budget / current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    self.budget -= cost
                    self.shares_owned = shares_to_buy
                    self.buy_price = current_price
                    self.in_position = True
                    
                    self.trades.append({
                        'date': i.strftime('%Y-%m-%d'),
                        'action': 'BUY',
                        'price': round(current_price, 2),
                        'shares': shares_to_buy,
                        'balance': round(self.budget, 2)
                    })
            
            # SELL signal
            elif signal == -1 and self.in_position:
                revenue = self.shares_owned * current_price
                self.budget += revenue
                
                self.trades.append({
                    'date': i.strftime('%Y-%m-%d'),
                    'action': 'SELL',
                    'price': round(current_price, 2),
                    'shares': 0,
                    'balance': round(self.budget, 2)
                })
                
                self.shares_owned = 0
                self.in_position = False
    
    def force_close_position(self):
        """Force close any open position at the end"""
        if self.in_position and self.shares_owned > 0:
            last_date = self.data.index[-1]
            last_price = float(self.data.iloc[-1]['Close'])
            
            revenue = self.shares_owned * last_price
            self.budget += revenue
            
            self.trades.append({
                'date': last_date.strftime('%Y-%m-%d'),
                'action': 'SELL',
                'price': round(last_price, 2),
                'shares': 0,
                'balance': round(self.budget, 2)
            })
            
            self.shares_owned = 0
            self.in_position = False
    
    def get_chart_data(self) -> List[dict]:
        """Get chart data for visualization"""
        chart_data = []
        
        # Sample data to avoid sending too many points
        step = max(1, len(self.data) // 200)
        
        for i, (idx, row) in enumerate(self.data.iterrows()):
            if i % step == 0 or i == len(self.data) - 1:
                chart_data.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'close': round(float(row['Close']), 2),
                    'ma50': round(float(row['MA_50']), 2) if pd.notna(row['MA_50']) else None,
                    'ma200': round(float(row['MA_200']), 2) if pd.notna(row['MA_200']) else None
                })
        
        return chart_data
    
    def run_strategy(self) -> dict:
        """Run the complete trading strategy"""
        if not self.download_data():
            return {
                'success': False,
                'message': f'Could not download data for symbol {self.symbol}. Please check the symbol and try again.'
            }
        
        if len(self.data) < 200:
            return {
                'success': False,
                'message': f'Not enough data points for {self.symbol}. Need at least 200 trading days for moving average calculation.'
            }
        
        self.clean_data()
        self.calculate_moving_averages()
        self.identify_signals()
        self.execute_trades()
        self.force_close_position()
        
        # Calculate performance
        total_profit = self.budget - self.initial_budget
        percent_return = (total_profit / self.initial_budget) * 100
        
        return {
            'success': True,
            'symbol': self.symbol,
            'initial_budget': self.initial_budget,
            'final_balance': round(self.budget, 2),
            'total_profit_loss': round(total_profit, 2),
            'percent_return': round(percent_return, 2),
            'total_trades': len(self.trades),
            'trades': self.trades,
            'chart_data': self.get_chart_data(),
            'message': 'Simulation completed successfully!'
        }


@app.get("/")
def home():
    """Serve the index.html file"""
    return FileResponse("index.html")


@app.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """
    Run trading simulation
    
    - **symbol**: Stock symbol (e.g., AAPL, MSFT, GOOGL)
    - **start_date**: Start date in YYYY-MM-DD format
    - **end_date**: End date in YYYY-MM-DD format
    - **budget**: Initial trading budget (default: $10,000)
    """
    try:
        # Validate dates
        start = datetime.strptime(request.start_date, '%Y-%m-%d')
        end = datetime.strptime(request.end_date, '%Y-%m-%d')
        
        if start >= end:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        if request.budget <= 0:
            raise HTTPException(status_code=400, detail="Budget must be positive")
        
        # Run simulation
        trader = AlgorithmicTrader(
            symbol=request.symbol,
            from_date=request.start_date,
            to_date=request.end_date,
            budget=request.budget
        )
        
        result = trader.run_strategy()
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['message'])
        
        return SimulationResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running simulation: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Stock Trading Simulator"}


if __name__ == "__main__":
    print("📈 Stock Trading Simulator")
    print("\nStarting server on http://localhost:8000")
    print("\nEndpoints:")
    print("  GET  /          - Web interface")
    print("  POST /simulate  - Run trading simulation")
    print("\nAPI Documentation: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
