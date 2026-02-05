import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

class AlgorithmicTrader:
    
    def __init__(self, symbol, from_date, to_date, budget=5000):
        self.symbol = symbol
        self.from_date = from_date
        self.to_date = to_date
        self.budget = budget
        self.initial_budget = budget
        
        self.shares_owned = 0
        self.in_position = False
        self.buy_price = 0
        self.trades = []
        
        print(f"Starting Algorithmic Trading Adventure!")
        print(f"Symbol: {symbol} | Budget: ${budget} | Period: {from_date} to {to_date}\n")
    
    def download_data(self):
        print("Downloading historical data...")
        self.data = yf.download(self.symbol, start=self.from_date, end=self.to_date)
        
        # Data thik kora
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        
        print(f"Downloaded {len(self.data)} days of data\n")
        return self.data
    
    def clean_data(self):
        print("Cleaning data...")
        
        original_length = len(self.data)
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        duplicates_removed = original_length - len(self.data)
        
        self.data = self.data.ffill()
        
        print(f"Removed {duplicates_removed} duplicates")
        print(f"Filled missing values using forward fill\n")
    
    def calculate_moving_averages(self):
        print("Calculating moving averages...")
        
        # 50 & 200 din er average
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA_200'] = self.data['Close'].rolling(window=200).mean()
        
        self.data = self.data.dropna()
        
        print(f"Calculated 50-day and 200-day moving averages")
        print(f"Data points with valid MAs: {len(self.data)}\n")
    
    def identify_signals(self):
        print("Identifying trading signals...")
        
        self.data['Signal'] = 0
        
        for i in range(1, len(self.data)):
            current_idx = self.data.index[i]
            prev_idx = self.data.index[i-1]
            
            ma_50_current = self.data.loc[current_idx, 'MA_50']
            ma_200_current = self.data.loc[current_idx, 'MA_200']
            
            ma_50_prev = self.data.loc[prev_idx, 'MA_50']
            ma_200_prev = self.data.loc[prev_idx, 'MA_200']
            
            # Kena
            if ma_50_prev <= ma_200_prev and ma_50_current > ma_200_current:
                self.data.loc[current_idx, 'Signal'] = 1
            
            # Bikri
            elif ma_50_prev >= ma_200_prev and ma_50_current < ma_200_current:
                self.data.loc[current_idx, 'Signal'] = -1
        
        buy_signals = (self.data['Signal'] == 1).sum()
        sell_signals = (self.data['Signal'] == -1).sum()
        
        print(f"Found {buy_signals} BUY signals (Golden Cross)")
        print(f"Found {sell_signals} SELL signals (Death Cross)\n")
    
    def execute_trades(self):
        print("Executing trades...\n")
        
        for i, row in self.data.iterrows():
            current_price = row['Close']
            signal = row['Signal']
            
            if signal == 1 and not self.in_position:
                shares_to_buy = int(self.budget / current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    self.budget -= cost
                    self.shares_owned = shares_to_buy
                    self.buy_price = current_price
                    self.in_position = True
                    
                    trade_info = {
                        'Date': i,
                        'Type': 'BUY',
                        'Price': current_price,
                        'Shares': shares_to_buy,
                        'Total': cost
                    }
                    self.trades.append(trade_info)
                    
                    print(f"BUY  | Date: {i.date()} | Price: ${current_price:.2f} | "
                          f"Shares: {shares_to_buy} | Total: ${cost:.2f} | "
                          f"Remaining Budget: ${self.budget:.2f}")
            
            elif signal == -1 and self.in_position:
                revenue = self.shares_owned * current_price
                self.budget += revenue
                profit = revenue - (self.shares_owned * self.buy_price)
                
                trade_info = {
                    'Date': i,
                    'Type': 'SELL',
                    'Price': current_price,
                    'Shares': self.shares_owned,
                    'Total': revenue,
                    'Profit': profit
                }
                self.trades.append(trade_info)
                
                print(f"SELL | Date: {i.date()} | Price: ${current_price:.2f} | "
                      f"Shares: {self.shares_owned} | Total: ${revenue:.2f} | "
                      f"Profit: ${profit:.2f}")
                
                self.shares_owned = 0
                self.in_position = False
        
        print()
    
    def force_close_position(self):
        if self.in_position and self.shares_owned > 0:
            print("Forcing position closure on last day...")
            
            last_date = self.data.index[-1]
            last_price = self.data.iloc[-1]['Close']
            
            revenue = self.shares_owned * last_price
            self.budget += revenue
            profit = revenue - (self.shares_owned * self.buy_price)
            
            trade_info = {
                'Date': last_date,
                'Type': 'FORCED SELL',
                'Price': last_price,
                'Shares': self.shares_owned,
                'Total': revenue,
                'Profit': profit
            }
            self.trades.append(trade_info)
            
            print(f"FORCED SELL | Date: {last_date.date()} | Price: ${last_price:.2f} | "
                  f"Shares: {self.shares_owned} | Total: ${revenue:.2f} | "
                  f"Profit: ${profit:.2f}\n")
            
            self.shares_owned = 0
            self.in_position = False
    
    def calculate_performance(self):
        print("Trading performance summary:")
        
        # Labh-khoti hisab
        total_profit = self.budget - self.initial_budget
        percent_return = (total_profit / self.initial_budget) * 100
        
        print(f"\nInitial Budget:  ${self.initial_budget:.2f}")
        print(f"Final Budget:    ${self.budget:.2f}")
        print(f"Total Profit/Loss: ${total_profit:.2f}")
        print(f"Return:          {percent_return:.2f}%")
        print(f"\nTotal Trades Executed: {len(self.trades)}")
        
        if total_profit > 0:
            print(f"\nCongratulations! Your strategy made a profit of ${total_profit:.2f}!")
        elif total_profit < 0:
            print(f"\nYour strategy resulted in a loss of ${abs(total_profit):.2f}.")
        else:
            print("\nYour strategy broke even.")
    
    def visualize(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Chart 1: Price aar moving averages
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', color='black', alpha=0.5)
        ax1.plot(self.data.index, self.data['MA_50'], label='50-Day MA', color='blue', linewidth=1.5)
        ax1.plot(self.data.index, self.data['MA_200'], label='200-Day MA', color='red', linewidth=1.5)
        
        buy_trades = [t for t in self.trades if t['Type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['Type'] in ['SELL', 'FORCED SELL']]
        
        if buy_trades:
            buy_dates = [t['Date'] for t in buy_trades]
            buy_prices = [t['Price'] for t in buy_trades]
            ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        
        if sell_trades:
            sell_dates = [t['Date'] for t in sell_trades]
            sell_prices = [t['Price'] for t in sell_trades]
            ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{self.symbol} - Golden Cross Trading Strategy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Portfolio value
        portfolio_values = []
        dates = []
        current_cash = self.initial_budget
        current_shares = 0
        
        for idx, row in self.data.iterrows():
            price = row['Close']
            signal = row['Signal']
            
            if signal == 1 and current_shares == 0:
                shares = int(current_cash / price)
                if shares > 0:
                    current_cash -= shares * price
                    current_shares = shares
            
            elif signal == -1 and current_shares > 0:
                current_cash += current_shares * price
                current_shares = 0
            
            portfolio_value = current_cash + (current_shares * price)
            portfolio_values.append(portfolio_value)
            dates.append(idx)
        
        ax2.plot(dates, portfolio_values, label='Portfolio Value', color='purple', linewidth=2)
        ax2.axhline(y=self.initial_budget, color='gray', linestyle='--', label='Initial Budget')
        ax2.fill_between(dates, portfolio_values, self.initial_budget, 
                         where=[pv >= self.initial_budget for pv in portfolio_values], 
                         alpha=0.3, color='green', label='Profit')
        ax2.fill_between(dates, portfolio_values, self.initial_budget, 
                         where=[pv < self.initial_budget for pv in portfolio_values], 
                         alpha=0.3, color='red', label='Loss')
        
        ax2.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_strategy(self):
        self.download_data()
        self.clean_data()
        self.calculate_moving_averages()
        self.identify_signals()
        self.execute_trades()
        self.force_close_position()
        self.calculate_performance()
        self.visualize()


if __name__ == "__main__":
    trader = AlgorithmicTrader("AAPL", "2018-01-01", "2023-12-31")
    trader.run_strategy()
