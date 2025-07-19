import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import mplcursors
import threading
import time
import requests
from bs4 import BeautifulSoup
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay
from textblob import TextBlob

# Set all random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

class NSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('Good Friday', month=4, day=18, year=2025),
        Holiday('Republic Day', month=1, day=26),
        Holiday('Mahashivratri', month=3, day=1, year=2025),
    ]

class FinalStockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.initializer = GlorotUniform(seed=42)
        self.news_impact = ""
        self.last_close = 0
        
    def get_data(self, symbol):
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        data = yf.download(symbol, start='2015-01-01', end=end_date, progress=False)
        if len(data) < 100:
            raise ValueError("Insufficient historical data")
        self.last_close = data['Close'].iloc[-1]
        return data

    def get_market_news(self):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            url = "https://www.google.com/search?q=NIFTY+50+news&tbm=nws"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_items = []
            for item in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
                news_items.append(item.get_text())
            
            # Simple sentiment analysis
            sentiment = 0
            for news in news_items[:3]:  # Analyze top 3 news
                analysis = TextBlob(news)
                sentiment += analysis.sentiment.polarity
            
            avg_sentiment = sentiment / 3 if len(news_items) >= 3 else sentiment
            return avg_sentiment, news_items[:3]
            
        except Exception as e:
            print(f"News fetch error: {e}")
            return 0, ["Could not fetch current news"]

    def preprocess(self, data):
        data = data.copy()
        # Original technical indicators
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['SMA_200'] = data['Close'].rolling(200).mean()
        
        # Original RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))
        
        features = data[['Close', 'SMA_50', 'SMA_200', 'RSI']].dropna()
        scaled = self.scaler.fit_transform(features)
        return features, scaled

    def create_sequences(self, data, n_steps=50):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i+n_steps])
            y.append(data[i+n_steps, 0])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape, 
                 kernel_initializer=self.initializer),
            Dropout(0.2, seed=42),
            LSTM(100, return_sequences=True, kernel_initializer=self.initializer),
            Dropout(0.2, seed=42),
            LSTM(50, return_sequences=False, kernel_initializer=self.initializer),
            Dropout(0.2, seed=42),
            Dense(25, kernel_initializer=self.initializer),
            Dense(1, kernel_initializer=self.initializer)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, last_data, days):
        # Get market sentiment
        sentiment, news = self.get_market_news()
        self.news_impact = "LATEST MARKET NEWS & SENTIMENT:\n" + "\n".join([f"- {n}" for n in news])
        
        # Prepare business days (skip holidays)
        bday = CustomBusinessDay(calendar=NSEHolidayCalendar())
        last_date = last_data.index[-1]
        date_range = pd.date_range(start=last_date + bday, periods=days, freq=bday)
        
        # Initialize sequence (using original 50-step window)
        current_seq = self.scaler.transform(last_data)[-50:].reshape(1, 50, 4)
        predictions = []
        
        # Apply small sentiment adjustment (original ±5% range)
        sentiment_factor = 1 + (0.05 * sentiment)
        
        for target_date in date_range[:days]:
            # Predict next value with sentiment adjustment
            pred = self.model.predict(current_seq, verbose=0)[0,0] * sentiment_factor
            
            # Create new row for sequence
            new_row = np.zeros((1, 1, 4))
            new_row[0,0,0] = pred
            new_row[0,0,1:] = current_seq[0,-1,1:]  # Carry forward indicators
            
            # Update sequence
            current_seq = np.append(current_seq[:,1:,:], new_row, axis=1)
            
            # Inverse transform
            dummy = np.zeros((1, self.scaler.n_features_in_))
            dummy[0,0] = max(0, min(1, pred))
            price = self.scaler.inverse_transform(dummy)[0,0]
            
            predictions.append(price)
        
        return date_range[:days], np.array(predictions), sentiment

class FinalPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NIFTY 50 Advanced Predictor")
        self.root.geometry("1200x900")
        
        # Create main container with scrollbar
        self.main_canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(
                scrollregion=self.main_canvas.bbox("all")
            )
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Configure grid weights
        self.scrollable_frame.columnconfigure(0, weight=1)
        
        # 1. Input Parameters Section
        self.create_input_section()
        
        # 2. Training Progress Section
        self.create_training_section()
        
        # 3. Prediction Results Section
        self.create_results_section()
        
        # 4. Market News Section
        self.create_news_section()
        
        # 5. Daily Predictions Section
        self.create_daily_predictions_section()
        
        # Initialize predictor
        self.predictor = FinalStockPredictor()
        
    def create_input_section(self):
        input_frame = ttk.LabelFrame(self.scrollable_frame, text="1. Prediction Parameters", padding=15)
        input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        # Symbol Input
        ttk.Label(input_frame, text="Stock Symbol:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.symbol_entry = ttk.Entry(input_frame, width=15)
        self.symbol_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.symbol_entry.insert(0, "^NSEI")
        
        # Days to Predict
        ttk.Label(input_frame, text="Prediction Days:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.days_entry = ttk.Spinbox(input_frame, from_=1, to=30, width=5)
        self.days_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.days_entry.set(7)
        
        # Model Parameters
        ttk.Label(input_frame, text="Epochs:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.epochs_entry = ttk.Spinbox(input_frame, from_=10, to=200, width=5)
        self.epochs_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.epochs_entry.set(50)
        
        ttk.Label(input_frame, text="Lookback Days:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.lookback_entry = ttk.Spinbox(input_frame, from_=30, to=100, width=5)
        self.lookback_entry.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.lookback_entry.set(50)
        
        # Action Buttons
        self.run_btn = ttk.Button(input_frame, text="Run Prediction", command=self.start_prediction)
        self.run_btn.grid(row=2, column=0, columnspan=4, pady=10)
        
    def create_training_section(self):
        training_frame = ttk.LabelFrame(self.scrollable_frame, text="2. Training Progress", padding=15)
        training_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        # Progress Bar
        self.progress = ttk.Progressbar(training_frame, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(fill="x", padx=5, pady=5)
        
        # Status Label
        self.status_label = ttk.Label(training_frame, text="Ready to start prediction...")
        self.status_label.pack(fill="x", padx=5, pady=5)
        
        # Epoch Details
        self.epoch_details = scrolledtext.ScrolledText(training_frame, height=8, wrap=tk.WORD)
        self.epoch_details.pack(fill="both", expand=True, padx=5, pady=5)
        self.epoch_details.insert("end", "Training details will appear here...\n")
        
    def create_results_section(self):
        results_frame = ttk.LabelFrame(self.scrollable_frame, text="3. Prediction Results", padding=15)
        results_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        
        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_news_section(self):
        news_frame = ttk.LabelFrame(self.scrollable_frame, text="4. Market News Impact Analysis", padding=15)
        news_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        
        self.news_text = scrolledtext.ScrolledText(news_frame, height=10, wrap=tk.WORD)
        self.news_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.news_text.insert("end", "Market news and sentiment analysis will appear here...\n")
        
    def create_daily_predictions_section(self):
        pred_frame = ttk.LabelFrame(self.scrollable_frame, text="5. Daily Price Predictions", padding=15)
        pred_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=5)
        
        # Treeview for tabular data
        self.prediction_tree = ttk.Treeview(pred_frame, columns=("date", "price", "change"), show="headings")
        self.prediction_tree.heading("date", text="Date")
        self.prediction_tree.heading("price", text="Predicted Price")
        self.prediction_tree.heading("change", text="Change %")
        
        self.prediction_tree.column("date", width=150, anchor="center")
        self.prediction_tree.column("price", width=150, anchor="center")
        self.prediction_tree.column("change", width=100, anchor="center")
        
        self.prediction_tree.pack(fill="both", expand=True, padx=5, pady=5)
        
    def start_prediction(self):
        try:
            symbol = self.symbol_entry.get().strip()
            days = int(self.days_entry.get())
            epochs = int(self.epochs_entry.get())
            lookback = int(self.lookback_entry.get())
            
            if not 1 <= days <= 30:
                raise ValueError("Days must be between 1-30")
            if not 10 <= epochs <= 200:
                raise ValueError("Epochs must be between 10-200")
            if not 30 <= lookback <= 100:
                raise ValueError("Lookback days must be between 30-100")
                
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return
            
        self.run_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_prediction, args=(symbol, days, epochs, lookback), daemon=True).start()
    
    def run_prediction(self, symbol, days, epochs, lookback):
        try:
            # Clear previous results
            self.epoch_details.delete(1.0, tk.END)
            self.news_text.delete(1.0, tk.END)
            for item in self.prediction_tree.get_children():
                self.prediction_tree.delete(item)
            
            # STAGE 1: Data Preparation
            self.update_status("Downloading market data...", 10)
            data = self.predictor.get_data(symbol)
            features, scaled = self.predictor.preprocess(data)
            
            # STAGE 2: Model Training
            self.update_status("Preparing training data...", 20)
            X, y = self.predictor.create_sequences(scaled, n_steps=lookback)
            
            self.update_status("Building model...", 30)
            self.predictor.model = self.predictor.build_model((X.shape[1], X.shape[2]))
            
            # Training loop with detailed progress
            for epoch in range(epochs):
                start_time = time.time()
                history = self.predictor.model.fit(
                    X, y, 
                    epochs=1, 
                    batch_size=32,
                    verbose=0
                )
                elapsed = time.time() - start_time
                progress = 30 + (epoch/epochs)*60
                
                epoch_log = f"Epoch {epoch+1}/{epochs} | Loss: {history.history['loss'][0]:.4f} | Time: {elapsed:.1f}s\n"
                self.epoch_details.insert(tk.END, epoch_log)
                self.epoch_details.see(tk.END)
                
                self.update_status(
                    f"Training model: Epoch {epoch+1}/{epochs}",
                    progress
                )
            
            # STAGE 3: Prediction
            self.update_status("Analyzing market news...", 85)
            last_data = features[['Close', 'SMA_50', 'SMA_200', 'RSI']][-lookback:]
            
            self.update_status("Generating predictions...", 90)
            dates, preds, sentiment = self.predictor.predict(last_data, days)
            
            # Show results
            self.update_status("Prediction complete!", 100)
            self.show_results(data, dates, preds, sentiment)
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}", 0)
            messagebox.showerror("Error", str(e))
        finally:
            self.run_btn.config(state=tk.NORMAL)
    
    def update_status(self, text, progress):
        self.root.after(0, lambda: [
            self.status_label.config(text=text),
            self.progress.config(value=progress),
            self.root.update()
        ])
    
    def show_results(self, data, dates, preds, sentiment):
        # Clear previous plot
        self.ax.clear()
        
        # Plot actual data (last 100 days)
        self.ax.plot(
            data.index[-100:], 
            data['Close'].values[-100:], 
            label='Actual Market Trend', 
            color='green',
            linewidth=2
        )
        
        # Mark prediction start
        self.ax.axvline(
            x=data.index[-1], 
            linestyle='--', 
            color='gray', 
            label='Prediction Start'
        )
        
        # Plot predictions
        self.ax.plot(
            dates, 
            preds, 
            marker='o', 
            linestyle='-', 
            color='blue', 
            label='Predicted Market Trend',
            markersize=6
        )
        
        # Formatting
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Stock Price')
        title = f"NIFTY 50 Price Prediction (Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'})"
        self.ax.set_title(title)
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add interactive hover
        mplcursors.cursor(hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(
                f"{sel.artist.get_label()}\n"
                f"Date: {plt.matplotlib.dates.num2date(sel.target[0]).strftime('%d %b %Y')}\n"
                f"Price: {sel.target[1]:.2f}"
            )
        )
        
        self.canvas.draw()
        
        # Display news and sentiment
        self.news_text.delete(1.0, tk.END)
        self.news_text.insert(tk.END, self.predictor.news_impact + "\n\n")
        self.news_text.insert(tk.END, f"Market Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}\n")
        self.news_text.insert(tk.END, f"Sentiment Score: {sentiment:.2f}\n")
        
        # Display daily predictions in table
        last_price = self.predictor.last_close
        for date, price in zip(dates, preds):
            change_pct = ((price - last_price) / last_price) * 100
            change_str = f"{change_pct:+.2f}%"
            self.prediction_tree.insert("", "end", values=(
                date.strftime('%d %b %Y'),
                f"{price:.2f}",
                change_str
            ))
            last_price = price

if __name__ == "__main__":
    root = tk.Tk()
    app = FinalPredictionApp(root)
    root.mainloop()