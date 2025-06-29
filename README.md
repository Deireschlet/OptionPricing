# Options Pricing and Analysis Tool

![Python Version](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)


## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
  - [Core Files](#core-files)
  - [Directories](#directories)
- [Key Components in Detail](#key-components-in-detail)
  - [Option Contract Representation](#option-contract-representation)
  - [Pricing Models](#pricing-models)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
  - [User Interface](#user-interface)
  - [Data Handling](#data-handling)
- [Design Considerations](#design-considerations)
  - [Modular Architecture](#modular-architecture)
  - [Performance Optimization](#performance-optimization)
  - [Extensibility](#extensibility)
- [Getting Started](#getting-started)
- [Future Enhancements](#future-enhancements)
- [Limitations and Future Work](#limitations-and-future-work)
- [Conclusion](#conclusion)


## Project Overview
This project is a comprehensive options pricing and analysis tool built as a final project for CS50x. It provides both 
a web interface and programmatic tools for options pricing, Greek calculation, volatility estimation, and machine l
earning-based options price prediction. The application combines traditional financial models like Black-Scholes 
with advanced techniques including Monte Carlo simulations and machine learning algorithms to provide a full 
suite of options analysis capabilities.

The project is built on my [Python Project Template](https://github.com/Deireschlet/Python-Project-Template.git), 
a custom framework I developed as a go-to starting point for Python projects with pre-configured package management, 
logging, and configuration systems, allowing me to focus on the financial modeling aspects rather than the project setup.


## Features
- **Contract Setup**: Easy configuration of option contracts with customizable parameters
- **Pricing Models**: Multiple pricing methodologies including:
    - Black-Scholes analytical solution for European options
    - Monte Carlo simulation for European options
    - Longstaff-Schwartz Least Square Monte Carlo for American options

- **Greeks Calculation**: Computation of option sensitivity measures (Delta, Gamma, Theta, Vega, Rho)
- **Implied Volatility**: Calculation and visualization of implied volatility
- **Machine Learning Models**: Price prediction using three different ML approaches:
    - Linear Regression
    - Random Forest
    - XGBoost

- **Interactive UI**: Built with Streamlit for intuitive exploration and analysis

## Project Structure
### Core Files
- **0_Home.py**: Main entry point that provides the option contract setup interface 
- **main.py**: Backend entry point for programmatic usage of the library 
- **pyproject.toml**: Project configuration, dependencies, and metadata 
- **requirements.txt**: Project dependencies for installation 
- **config.ini**: Configuration settings for models, paths, and project parameters 

### Directories
- **/src**: Core library code
    - **/src/models**: Machine learning model implementation
    - **/src/ui**: User interface components
    - **/src/utils**: Utility functions

- **/pages**: Streamlit app pages for different features
- **/setup**: Project setup and configuration code
- **/data**: Data storage for market data and saved models 
- **/tests**: Unit and integration tests
- **/notebooks**: Jupyter notebooks for development and demonstration
- **/logs**: Application logs

## Key Components in Detail
### Option Contract Representation
`src/option.py` defines the core class that encapsulates all parameters of an option contract, including: 
- Option type (call/put)
- Spot price
- Strike price
- Maturity
- Risk-free rate
- Volatility
- Underlying asset information

This class serves as the foundation for all pricing and analysis operations throughout the application.
### Pricing Models
The project implements multiple pricing approaches:
1. **Black-Scholes Model**: Analytical solution for European options that provides a closed-form formula for option pricing.
2. **Monte Carlo Simulation**: Stochastic simulation method that generates thousands of potential price paths to estimate option values, providing distribution insights beyond just the expected price. 
3. **Least Square Monte Carlo**: Advanced technique for pricing American options that allows for early exercise valuation, using regression techniques to estimate continuation values.

### Machine Learning Pipeline
The ML component (`src/models`) implements a complete pipeline for training and evaluating option pricing models:
1. **Preprocessing**: Data preparation, feature engineering, and scaling 
2. **Modeling**: Pipeline creation for three model types with hyperparameter configurations 
3. **Evaluation**: Model assessment with metrics like MAE, RMSE, and R² 
4. **Model Storage**: Persistent storage of trained models with data hashing to avoid retraining

The ML pipeline to be extensible, allowing for easy addition of new models and features. 
The system automatically caches trained models based on data hashes to avoid redundant computation.
### User Interface
The Streamlit-based UI is organized into multiple pages:
1. **Home**: Contract setup and configuration 
2. **Pricing**: Option pricing with different methodologies 
3. **Greeks**: Sensitivity analysis 
4. **Implied Volatility**: Volatility calculation and visualization 
5. **Machine Learning**: ML model training, evaluation, and prediction 

Each page provides interactive controls and visualizations for exploring different aspects of options analysis.
### Data Handling
The component manages data acquisition and processing, including:
- Fetching option data for specific tickers and option types
- Retrieving latest prices
- Data transformation and preparation for pricing models and ML

## Design Considerations
### Modular Architecture
I deliberately designed the system with high modularity to separate concerns and enable flexible usage. 
Each component (pricing, Greeks, ML) can function independently, allowing the system to be used either through the UI 
or programmatically.
### Performance Optimization
For computationally intensive operations like Monte Carlo simulations and ML model training, I implemented:
- Caching mechanisms to avoid redundant calculations
- Parallelization where applicable
- Selective computation of expensive operations only when necessary
### Extensibility
The project is structured to allow easy extension:
- New pricing models can be added without changing existing code
- Additional ML algorithms can be incorporated by extending the modeling module
- The UI is decoupled from core functionality, enabling alternative interfaces

## Getting Started
The project uses [`uv`](https://docs.astral.sh/uv/) as the package manager for faster dependency resolution and installation. The file ensures reproducible environments across different setups.

Set up environment and install dependencies using `uv`:
``` 
   uv venv
   uv pip install -r requirements.txt
```
or you can use
``` 
   uv sync
```
This wil make use of the `pyproject.toml` file.

Run the Streamlit application:
``` 
   streamlit run 0_Home.py
```
For programmatic usage, import the relevant modules:
``` python
   from src.option import Option
   from src.computation import black_scholes
   
   # Create an option contract
   option = Option(
       spot_price=100.0,
       option_type="call",
       strike_price=105.0,
       maturity=30,
       risk_free_rate=0.05,
       volatility=0.2,
       underlying_ticker="AAPL"
   )
   
   # Calculate option price
   price = black_scholes(S0=option.spot_price, option=option)
```
## Limitations and Future Enhancements

This project provides a solid foundation for options pricing and analysis but is still a basic toolkit. It is not yet suited for production-grade use or complex financial environments. Some limitations include:

- No integration with real-time market data  
- Simplified pricing assumptions (e.g. no market frictions or transaction costs)  
- No support for exotic or path-dependent options  
- Limited to basic models like Black-Scholes and Monte Carlo  
- No portfolio-level analysis or risk management tools  

To address these and expand functionality, the following enhancements are planned:

- Integration with real-time market data APIs  
- Implementation of additional pricing models (e.g., Heston model)  
- Support for exotic options  
- Portfolio analysis and optimization  
- Sensitivity analysis and stress testing  
- More advanced calibration and data preprocessing techniques  

Users are encouraged to build on this framework to include more realistic features and improve overall model accuracy.

## Conclusion
This project provides a comprehensive toolkit for options analysis, combining traditional financial models with modern machine learning approaches. It demonstrates how computational methods can enhance financial modeling and provide insights beyond conventional techniques.
