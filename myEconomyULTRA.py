import random
import json
import math
import numpy as np
import pandas as pd
import datetime
import os
import sys
from typing import Dict, List, Tuple, Optional, Union, Any
import uuid
from dataclasses import dataclass, field, asdict

# GUI Libraries
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk  # Modern UI toolkit
from PIL import Image, ImageTk

# For potential web export
import flask
from flask import Flask, render_template, request, jsonify, session

# For plotting and data visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===== APPLICATION CONSTANTS =====
APP_NAME = "myEconomy"
APP_VERSION = "1.0.0"
DEFAULT_FONT = ("Helvetica", 12)
HEADER_FONT = ("Helvetica", 18, "bold")
BUTTON_FONT = ("Helvetica", 10)
PRIMARY_COLOR = "#3498db"  # Tesla blue
SECONDARY_COLOR = "#2c3e50"  # Dark blue/gray
BG_COLOR = "#f8f9fa"  # Light gray
TEXT_COLOR = "#333333"  # Dark gray
SUCCESS_COLOR = "#2ecc71"  # Green
WARNING_COLOR = "#f39c12"  # Orange
DANGER_COLOR = "#e74c3c"  # Red
INFO_COLOR = "#3498db"  # Light blue

# ===== DATA STRUCTURES =====
@dataclass
class Tax:
    name: str
    rate: float  # Percentage as decimal (0.05 = 5%)
    bracketed: bool = False
    brackets: List[Tuple[float, float]] = field(default_factory=list)  # [(income_threshold, rate)]

@dataclass
class Department:
    name: str
    budget: float
    efficiency: float = 0.7  # How effectively the budget is used (0-1)
    staff: int = 0
    infrastructure: int = 0  # Buildings, equipment, etc.
    satisfaction: float = 0.5  # Citizen satisfaction with this department (0-1)
    
    def allocate_budget(self, amount: float) -> None:
        self.budget = amount
        
    def hire_staff(self, count: int, cost_per_staff: float) -> bool:
        if cost_per_staff * count > self.budget:
            return False
        self.staff += count
        self.budget -= cost_per_staff * count
        return True
        
    def build_infrastructure(self, count: int, cost_per_unit: float) -> bool:
        if cost_per_unit * count > self.budget:
            return False
        self.infrastructure += count
        self.budget -= cost_per_unit * count
        return True
        
    def calculate_effectiveness(self) -> float:
        # Complex calculation based on budget, staff, infrastructure and efficiency
        base = (self.budget * 0.3 + self.staff * 0.3 + self.infrastructure * 0.4) 
        return base * self.efficiency

@dataclass
class Product:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = ""
    price: float = 0.0
    cost: float = 0.0
    r_and_d_investment: float = 0.0
    quality: float = 0.5  # 0-1 scale
    marketing_budget: float = 0.0
    brand_awareness: float = 0.1  # 0-1 scale
    units_produced: int = 0
    units_sold: int = 0
    sales_history: Dict[str, int] = field(default_factory=dict)  # date: units_sold
    
    @property
    def profit_margin(self) -> float:
        if self.price == 0:
            return 0
        return (self.price - self.cost) / self.price
    
    def invest_in_research(self, amount: float) -> None:
        self.r_and_d_investment += amount
        # Improve quality based on investment
        quality_increase = amount / 1000000  # Arbitrary formula
        self.quality = min(1.0, self.quality + quality_increase)
    
    def invest_in_marketing(self, amount: float) -> None:
        self.marketing_budget += amount
        # Improve brand awareness based on investment
        awareness_increase = amount / 500000  # Arbitrary formula
        self.brand_awareness = min(1.0, self.brand_awareness + awareness_increase)
    
    def set_price(self, price: float) -> None:
        if price < 0:
            raise ValueError("Price cannot be negative")
        self.price = price
    
    def produce(self, units: int) -> None:
        self.units_produced += units
    
    def sell(self, units: int, date: str) -> float:
        if units > self.units_produced:
            units = self.units_produced  # Can't sell more than produced
        
        self.units_sold += units
        self.units_produced -= units
        
        # Record sales history
        if date in self.sales_history:
            self.sales_history[date] += units
        else:
            self.sales_history[date] = units
        
        return units * self.price

@dataclass
class Business:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    founding_date: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    ceo: str = ""
    headquarters_country: str = ""
    headquarters_city: str = ""
    cash: float = 1000000.0  # Starting cash
    debt: float = 0.0
    credit_rating: str = "BB"  # Standard credit ratings: AAA, AA, A, BBB, BB, B, CCC, CC, C, D
    products: List[Product] = field(default_factory=list)
    employees: int = 10
    total_assets: float = 1000000.0
    total_liabilities: float = 0.0
    revenue_history: Dict[str, float] = field(default_factory=dict)  # date: revenue
    profit_history: Dict[str, float] = field(default_factory=dict)  # date: profit
    market_cap: float = 0.0  # For public companies
    share_price: float = 0.0  # For public companies
    shares_outstanding: int = 0  # For public companies
    is_public: bool = False
    ipo_date: Optional[str] = None
    stock_exchange: Optional[str] = None
    sectors: List[str] = field(default_factory=list)  # Technology, Retail, Manufacturing, etc.
    factories: Dict[str, int] = field(default_factory=dict)  # country: number_of_factories
    retail_stores: Dict[str, int] = field(default_factory=dict)  # country: number_of_stores
    brand_value: float = 1.0  # Overall brand strength (1.0+)
    
    @property
    def net_worth(self) -> float:
        return self.total_assets - self.total_liabilities
    
    @property
    def profit_margin(self) -> float:
        latest_date = max(self.revenue_history.keys()) if self.revenue_history else None
        if not latest_date or self.revenue_history[latest_date] == 0:
            return 0
        return self.profit_history[latest_date] / self.revenue_history[latest_date]
    
    def take_loan(self, amount: float, interest_rate: float, term_years: int) -> bool:
        """Take a loan with yearly interest rate and term"""
        self.debt += amount
        self.cash += amount
        self.total_liabilities += amount
        return True
    
    def pay_debt(self, amount: float) -> bool:
        """Pay off some debt"""
        if amount > self.cash:
            return False
        if amount > self.debt:
            amount = self.debt
            
        self.debt -= amount
        self.cash -= amount
        self.total_liabilities -= amount
        return True
    
    def hire_employees(self, count: int, salary_per_employee: float) -> bool:
        # Check if we can afford the new employees
        annual_cost = count * salary_per_employee
        if annual_cost > self.cash / 4:  # Arbitrary check to ensure we have enough runway
            return False
        
        self.employees += count
        return True
    
    def fire_employees(self, count: int, severance_multiplier: float = 3) -> bool:
        """Fire employees with severance pay"""
        if count > self.employees:
            return False
            
        avg_salary = 50000  # Arbitrary average salary
        severance_cost = count * avg_salary * severance_multiplier
        
        if severance_cost > self.cash:
            return False
            
        self.cash -= severance_cost
        self.employees -= count
        return True
    
    def develop_new_product(self, name: str, category: str, initial_investment: float) -> Optional[Product]:
        """Create a new product with initial investment in R&D"""
        if initial_investment > self.cash:
            return None
            
        self.cash -= initial_investment
        
        product = Product(
            name=name,
            category=category,
            r_and_d_investment=initial_investment,
            quality=0.3 + (initial_investment / 1000000),  # Base quality + investment factor
            cost=initial_investment / 100  # Initial production cost
        )
        
        self.products.append(product)
        return product
    
    def expand_to_country(self, country: str, factory_count: int = 0, store_count: int = 0, 
                          factory_cost: float = 1000000, store_cost: float = 500000) -> bool:
        """Expand business operations to a new country"""
        total_cost = (factory_count * factory_cost) + (store_count * store_cost)
        
        if total_cost > self.cash:
            return False
            
        self.cash -= total_cost
        self.total_assets += total_cost * 0.8  # Assets depreciate somewhat
        
        # Add factories if any
        if factory_count > 0:
            if country in self.factories:
                self.factories[country] += factory_count
            else:
                self.factories[country] = factory_count
                
        # Add retail stores if any
        if store_count > 0:
            if country in self.retail_stores:
                self.retail_stores[country] += store_count
            else:
                self.retail_stores[country] = store_count
                
        return True
    
    def go_public(self, stock_exchange: str, initial_share_price: float, 
                  shares_to_issue: int) -> bool:
        """Take the company public with an IPO"""
        if self.is_public:
            return False
            
        # Minimum requirements for IPO
        if self.total_assets < 10000000 or self.employees < 50:
            return False
            
        self.is_public = True
        self.ipo_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.stock_exchange = stock_exchange
        self.share_price = initial_share_price
        self.shares_outstanding = shares_to_issue
        
        # Calculate raised capital and market cap
        raised_capital = shares_to_issue * initial_share_price
        self.cash += raised_capital
        self.total_assets += raised_capital
        self.market_cap = self.shares_outstanding * self.share_price
        
        return True
    
    def simulate_quarter(self, economic_multiplier: float = 1.0, 
                         competition_factor: float = 1.0) -> Tuple[float, float]:
        """Simulate a business quarter and return revenue and profit"""
        print("Simulating quarter...")
        total_revenue = 0
        total_costs = 0
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Fixed costs (salaries, facilities, etc.)
        avg_salary = 50000  # Annual
        quarterly_salary_expense = (avg_salary / 4) * self.employees
        
        # Facilities costs
        factory_count = sum(self.factories.values())
        store_count = sum(self.retail_stores.values())
        quarterly_facilities_cost = (factory_count * 100000 + store_count * 25000) / 4
        
        # Interest on debt (simple calculation)
        avg_interest_rate = 0.05  # 5%
        quarterly_interest = self.debt * (avg_interest_rate / 4)
        
        total_costs += quarterly_salary_expense + quarterly_facilities_cost + quarterly_interest
        
        # Product sales
        for product in self.products:
            # Calculate potential sales based on quality, marketing, price, and market conditions
            market_size = random.randint(1000, 100000)  # Potential customers
            market_share = (product.quality * 0.4 + 
                           product.brand_awareness * 0.3 + 
                           (1.0 - min(1.0, product.price / 100000)) * 0.3)  # Price factor
            
            # Adjust for external factors
            market_share = market_share * economic_multiplier / competition_factor
            
            # Calculate actual sales
            potential_sales = int(market_size * market_share)
            actual_sales = min(potential_sales, product.units_produced)
            
            # Record sales
            revenue = product.sell(actual_sales, current_date)
            production_cost = actual_sales * product.cost
            
            total_revenue += revenue
            total_costs += production_cost
        
        # Calculate profit
        quarterly_profit = total_revenue - total_costs
        
        # Update business financials
        self.cash += quarterly_profit
        if quarterly_profit > 0:
            self.total_assets += quarterly_profit * 0.5  # Some profit goes to asset growth
        
        # Record historical data
        self.revenue_history[current_date] = total_revenue
        self.profit_history[current_date] = quarterly_profit
        
        # Update brand value based on profitability
        if quarterly_profit > 0:
            self.brand_value += 0.01 * (quarterly_profit / 1000000)
        else:
            self.brand_value = max(1.0, self.brand_value - 0.005)
        
        # Update company valuation if public
        if self.is_public:
            # Simple price model based on profit, assets, and market conditions
            pe_ratio = 15 + (5 * random.random())  # Price-to-earnings ratio with some randomness
            
            # Calculate annual earnings from quarterly profit (annualized)
            annual_earnings = quarterly_profit * 4
            
            if annual_earnings > 0:
                # Calculate new share price based on P/E ratio
                new_price = (annual_earnings / self.shares_outstanding) * pe_ratio
                # Limit how much price can change in one quarter
                max_change = self.share_price * 0.25
                price_change = max(min(new_price - self.share_price, max_change), -max_change)
                self.share_price += price_change
            else:
                # Losing money, so share price drops
                self.share_price *= 0.9
                
            # Update market cap
            self.market_cap = self.shares_outstanding * self.share_price
        
        return total_revenue, quarterly_profit

@dataclass
class ResourceMarket:
    name: str
    current_price: float
    volatility: float = 0.05  # Daily price volatility
    price_history: Dict[str, float] = field(default_factory=dict)  # date: price
    
    def update_price(self, date: str, external_factor: float = 1.0) -> None:
        # Random walk with drift and external factors
        change_percent = (random.random() - 0.5) * 2 * self.volatility * external_factor
        self.current_price *= (1 + change_percent)
        self.price_history[date] = self.current_price

@dataclass
class StockExchange:
    name: str
    country: str
    listed_companies: Dict[str, Business] = field(default_factory=dict)  # company_id: Business
    index_value: float = 1000.0  # Starting index value
    index_history: Dict[str, float] = field(default_factory=dict)  # date: index_value
    daily_volume: float = 0.0  # Trading volume in currency
    market_cap: float = 0.0  # Total market cap of listed companies
    
    def add_company(self, business: Business) -> bool:
        if not business.is_public:
            return False
        self.listed_companies[business.id] = business
        self.market_cap += business.market_cap
        return True
    
    def remove_company(self, business_id: str) -> bool:
        if business_id not in self.listed_companies:
            return False
        business = self.listed_companies[business_id]
        self.market_cap -= business.market_cap
        del self.listed_companies[business_id]
        return True
    
    def update_index(self, date: str) -> None:
        if not self.listed_companies:
            return
            
        # Simple market-cap weighted index
        new_market_cap = sum(company.market_cap for company in self.listed_companies.values())
        
        if self.market_cap > 0:  # Avoid division by zero
            index_change = new_market_cap / self.market_cap
            self.index_value *= index_change
        
        self.market_cap = new_market_cap
        self.index_history[date] = self.index_value
        
        # Simulate trading volume
        self.daily_volume = self.market_cap * (0.01 + 0.02 * random.random())  # 1-3% of market cap

@dataclass
class HousingMarket:
    country: str
    avg_price: float
    inventory: int  # Number of houses for sale
    demand: float  # Relative demand (1.0 = balanced)
    price_history: Dict[str, float] = field(default_factory=dict)  # date: avg_price
    
    def update_market(self, date: str, interest_rate: float, population_growth: float, 
                     income_growth: float, new_construction: int) -> None:
        # Multiple factors affect housing prices
        # Interest rates - lower rates increase prices
        rate_factor = 1.0 - (interest_rate * 2)  # Lower rates = higher prices
        
        # Supply and demand dynamics
        self.inventory += new_construction
        self.demand += population_growth * 0.1  # Population growth increases demand
        self.demand += income_growth * 0.2  # Income growth increases demand
        self.demand -= (self.avg_price / 100000) * 0.05  # Higher prices reduce demand
        
        # Ensure demand stays positive
        self.demand = max(0.1, self.demand)
        
        # Calculate supply/demand balance
        supply_demand_ratio = self.inventory / (self.demand * 10000)
        
        # Price adjustment based on supply/demand and interest rates
        if supply_demand_ratio < 0.8:  # Shortage
            price_change = 0.02 * rate_factor * (1 + (0.8 - supply_demand_ratio))
        elif supply_demand_ratio > 1.2:  # Oversupply
            price_change = -0.01 * (supply_demand_ratio - 1.2)
        else:  # Balanced
            price_change = 0.005 * rate_factor
            
        # Apply some randomness
        price_change += (random.random() - 0.5) * 0.01
        
        # Update price
        self.avg_price *= (1 + price_change)
        self.price_history[date] = self.avg_price
        
        # Simulate sales
        sales = min(self.inventory, int(self.demand * 5000))
        self.inventory -= sales

@dataclass
class Country:
    name: str
    gdp: float
    population: int
    land_area: float  # square kilometers
    government_type: str  # Democracy, Monarchy, Dictatorship, etc.
    gdp_growth: float = 0.02  # Annual growth rate
    inflation_rate: float = 0.02  # Annual inflation rate
    unemployment_rate: float = 0.05  # Fraction of workforce unemployed
    birth_rate: float = 0.01  # Annual births per person
    death_rate: float = 0.008  # Annual deaths per person
    immigration_rate: float = 0.001  # Annual net immigration per person
    average_income: float = field(init=False)
    tax_rate: float = 0.3  # Overall effective tax rate
    taxes: Dict[str, Tax] = field(default_factory=dict)  # Tax name: Tax object
    departments: Dict[str, Department] = field(default_factory=dict)  # Department name: Department
    national_debt: float = 0.0
    credit_rating: str = "AA"  # AAA to D
    interest_rate: float = 0.03  # Central bank interest rate
    currency: str = "Dollar"
    currency_strength: float = 1.0  # Relative to global average
    businesses: Dict[str, Business] = field(default_factory=dict)  # Business ID: Business
    stock_exchanges: Dict[str, StockExchange] = field(default_factory=dict)  # Exchange name: Exchange
    housing_market: HousingMarket = field(default=None)
    natural_resources: Dict[str, float] = field(default_factory=dict)  # Resource: amount
    resource_markets: Dict[str, ResourceMarket] = field(default_factory=dict)  # Resource: Market
    political_stability: float = 0.8  # 0-1 scale
    happiness: float = 0.7  # 0-1 scale
    education_level: float = 0.6  # 0-1 scale
    healthcare_quality: float = 0.7  # 0-1 scale
    infrastructure_quality: float = 0.6  # 0-1 scale
    environmental_quality: float = 0.7  # 0-1 scale
    international_relations: Dict[str, float] = field(default_factory=dict)  # Country: relation (-1 to 1)
    trade_balance: Dict[str, float] = field(default_factory=dict)  # Country: net export value
    sectors: Dict[str, float] = field(default_factory=dict)  # Sector name: contribution to GDP
    history: Dict[str, Dict[str, float]] = field(default_factory=dict)  # date: {metric: value}
    controlled_by_player: bool = False
    
    def __post_init__(self):
        # Calculate derived values
        self.average_income = self.gdp / self.population
        
        # Initialize default taxes if empty
        if not self.taxes:
            self.taxes = {
                "income_tax": Tax(
                    name="Income Tax",
                    rate=0.25,
                    bracketed=True,
                    brackets=[
                        (0, 0.10),          # 10% on first bracket
                        (10000, 0.15),      # 15% on income over $10k
                        (40000, 0.25),      # 25% on income over $40k
                        (90000, 0.30),      # 30% on income over $90k
                        (190000, 0.35),     # 35% on income over $190k
                        (415000, 0.396)     # 39.6% on income over $415k
                    ]
                ),
                "corporate_tax": Tax(name="Corporate Tax", rate=0.21),
                "sales_tax": Tax(name="Sales Tax", rate=0.07),
                "property_tax": Tax(name="Property Tax", rate=0.01),
            }
        
        # Initialize default departments if empty
        if not self.departments:
            gdp_fraction = self.gdp * 0.4  # 40% of GDP goes to government spending
            
            self.departments = {
                "healthcare": Department(name="Healthcare", budget=gdp_fraction * 0.2),
                "education": Department(name="Education", budget=gdp_fraction * 0.15),
                "defense": Department(name="Defense", budget=gdp_fraction * 0.15),
                "infrastructure": Department(name="Infrastructure", budget=gdp_fraction * 0.1),
                "welfare": Department(name="Welfare", budget=gdp_fraction * 0.2),
                "administration": Department(name="Administration", budget=gdp_fraction * 0.05),
                "research": Department(name="Research", budget=gdp_fraction * 0.05),
                "environment": Department(name="Environment", budget=gdp_fraction * 0.05),
                "justice": Department(name="Justice", budget=gdp_fraction * 0.05),
            }
        
        # Initialize sectors if empty
        if not self.sectors:
            self.sectors = {
                "agriculture": 0.03,
                "manufacturing": 0.15,
                "services": 0.55,
                "technology": 0.12,
                "construction": 0.06,
                "government": 0.09,
            }
        
        # Initialize housing market if not provided
        if self.housing_market is None:
            avg_house_price = self.average_income * 5  # 5x average income
            
            self.housing_market = HousingMarket(
                country=self.name,
                avg_price=avg_house_price,
                inventory=int(self.population / 100),  # 1% of population's worth of houses for sale
                demand=1.0  # Balanced demand
            )
            
        # Initialize natural resources
        if not self.natural_resources:
            # Random resource endowments based on land area
            self.natural_resources = {
                "oil": random.uniform(0, 100) * self.land_area / 1000000,
                "natural_gas": random.uniform(0, 100) * self.land_area / 1000000,
                "coal": random.uniform(0, 100) * self.land_area / 1000000,
                "iron": random.uniform(0, 100) * self.land_area / 1000000,
                "copper": random.uniform(0, 100) * self.land_area / 1000000,
                "gold": random.uniform(0, 10) * self.land_area / 1000000,
                "uranium": random.uniform(0, 5) * self.land_area / 1000000,
                "arable_land": random.uniform(0, 40) * self.land_area / 100,  # percentage
                "fresh_water": random.uniform(0, 30) * self.land_area / 100,  # percentage
            }
            
        # Initialize resource markets
        if not self.resource_markets:
            base_prices = {
                "oil": 60.0,         # per barrel
                "natural_gas": 3.0,   # per MCF
                "coal": 50.0,         # per ton
                "iron": 90.0,         # per ton
                "copper": 6000.0,     # per ton
                "gold": 1500.0,       # per ounce
                "uranium": 30.0,      # per pound
            }
            
            for resource, base_price in base_prices.items():
                self.resource_markets[resource] = ResourceMarket(
                    name=resource,
                    current_price=base_price * (0.8 + 0.4 * random.random()),  # Randomize around base price
                    volatility=0.03 + 0.07 * random.random(),  # Different volatilities
                )
    
    @property
    def gdp_per_capita(self) -> float:
        return self.gdp / self.population
        
    @property
    def government_budget(self) -> float:
        # Calculate total tax revenue
        return self.gdp * self.tax_rate
        
    @property
    def government_expenses(self) -> float:
        return sum(dept.budget for dept in self.departments.values())
        
    @property
    def budget_balance(self) -> float:
        return self.government_budget - self.government_expenses
    
    def change_tax(self, tax_name: str, new_rate: float) -> bool:
        """Change a tax rate"""
        if tax_name not in self.taxes:
            return False
            
        self.taxes[tax_name].rate = new_rate
        return True
    
    def modify_tax_bracket(self, tax_name: str, bracket_index: int, 
                          new_threshold: float, new_rate: float) -> bool:
        """Modify a specific tax bracket"""
        if (tax_name not in self.taxes or 
            not self.taxes[tax_name].bracketed or 
            bracket_index >= len(self.taxes[tax_name].brackets)):
            return False
            
        self.taxes[tax_name].brackets[bracket_index] = (new_threshold, new_rate)
        return True
    
    def adjust_department_budget(self, dept_name: str, new_budget: float) -> bool:
        """Change a department's budget allocation"""
        if dept_name not in self.departments:
            return False
            
        # Check if total budget still balances
        current_budget = self.departments[dept_name].budget
        budget_change = new_budget - current_budget
        
        if self.government_budget < self.government_expenses + budget_change:
            # Not enough funds available
            return False
            
        self.departments[dept_name].budget = new_budget
        return True
    
    def adjust_interest_rate(self, new_rate: float) -> None:
        """Central bank changes interest rate"""
        self.interest_rate = new_rate
        
        # Effect on economy (simplified)
        if new_rate > self.interest_rate:
            # Raising rates slows growth but reduces inflation
            self.gdp_growth *= 0.95
            self.inflation_rate *= 0.9
        else:
            # Lowering rates increases growth but may increase inflation
            self.gdp_growth *= 1.05
            self.inflation_rate *= 1.1
    
    def issue_debt(self, amount: float) -> bool:
        """Government issues bonds/debt"""
        # Interest rate based on credit rating and existing debt
        debt_to_gdp = self.national_debt / self.gdp
        risk_premium = debt_to_gdp * 0.1
        
        # Credit rating affects interest rate
        rating_to_premium = {
            "AAA": 0.0, "AA": 0.005, "A": 0.01, "BBB": 0.02,
            "BB": 0.04, "B": 0.07, "CCC": 0.12, "CC": 0.18, "C": 0.25, "D": 0.4
        }
        
        credit_premium = rating_to_premium.get(self.credit_rating, 0.05)
        
        # Calculate effective interest rate
        effective_rate = self.interest_rate + risk_premium + credit_premium
        
        # Add to national debt
        self.national_debt += amount
        
        # Adjust credit rating if debt becomes too high
        if debt_to_gdp > 1.0 and random.random() < 0.3:
            ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
            current_index = ratings.index(self.credit_rating)
            if current_index < len(ratings) - 1:
                self.credit_rating = ratings[current_index + 1]
        
        return True
    
    def simulate_day(self) -> Dict[str, float]:
        """Simulates one day in the country's economy"""
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        metrics = {}
        
        # Daily GDP growth (annual growth / 365)
        daily_gdp_growth = self.gdp_growth / 365
        self.gdp *= (1 + daily_gdp_growth)
        metrics["gdp"] = self.gdp
        
        # Daily inflation
        daily_inflation = self.inflation_rate / 365
        metrics["inflation"] = daily_inflation
        
        # Population change
        daily_births = self.population * (self.birth_rate / 365)
        daily_deaths = self.population * (self.death_rate / 365)
        daily_immigration = self.population * (self.immigration_rate / 365)
        population_change = daily_births - daily_deaths + daily_immigration
        self.population += int(population_change)
        metrics["population"] = self.population
        
        # Update housing market
        self.housing_market.update_market(
            date=date,
            interest_rate=self.interest_rate,
            population_growth=population_change / self.population,
            income_growth=daily_gdp_growth,
            new_construction=int(self.population * 0.0001 * random.random())  # Random new construction
        )
        metrics["housing_price"] = self.housing_market.avg_price
        
        # Update resource markets
        for resource, market in self.resource_markets.items():
            market.update_price(date)
            metrics[f"resource_{resource}"] = market.current_price
        
        # Update businesses if any
        for business in self.businesses.values():
            # Only quarterly updates for businesses
            current_day = datetime.datetime.now().day
            if current_day % 90 == 0:  # Rough quarterly check
                revenue, profit = business.simulate_quarter()
                # Record business metrics
                metrics[f"business_{business.id}_revenue"] = revenue
                metrics[f"business_{business.id}_profit"] = profit
        
        # Update stock exchanges
        for exchange in self.stock_exchanges.values():
            exchange.update_index(date)
            metrics[f"exchange_{exchange.name}"] = exchange.index_value
        
        # Record history
        self.history[date] = metrics
        
        return metrics
    
    def get_tax_revenue(self) -> float:
        """Calculate total tax revenue based on current tax rates and economy"""
        total_revenue = 0.0
        
        # Income tax (simplified)
        if "income_tax" in self.taxes:
            income_tax = self.taxes["income_tax"]
            if income_tax.bracketed:
                # Complex calculation using brackets and income distribution
                # This is simplified - real tax calculations would be much more complex
                income_tax_revenue = 0
                
                # Log-normal income distribution approximation
                mean_log_income = math.log(self.average_income) - 0.5
                income_distribution = np.random.lognormal(mean=mean_log_income, sigma=1.0, size=1000)
                
                for income in income_distribution:
                    tax_paid = 0
                    remaining_income = income
                    
                    # Sort brackets by threshold for calculation
                    sorted_brackets = sorted(income_tax.brackets, key=lambda x: x[0])
                    
                    for i, (threshold, rate) in enumerate(sorted_brackets):
                        if i < len(sorted_brackets) - 1:
                            next_threshold = sorted_brackets[i + 1][0]
                            if income > threshold:
                                taxable_amount = min(next_threshold - threshold, income - threshold)
                                tax_paid += taxable_amount * rate
                        else:  # Highest bracket
                            if income > threshold:
                                tax_paid += (income - threshold) * rate
                                
                    income_tax_revenue += tax_paid
                
                # Scale to full population
                total_revenue += income_tax_revenue * (self.population / 1000)
            else:
                # Simplified calculation
                total_revenue += self.gdp * 0.6 * income_tax.rate  # Assume 60% of GDP is income
        
        # Corporate tax
        if "corporate_tax" in self.taxes:
            corporate_profits = self.gdp * 0.2  # Assume 20% of GDP is corporate profits
            total_revenue += corporate_profits * self.taxes["corporate_tax"].rate
        
        # Sales tax
        if "sales_tax" in self.taxes:
            consumer_spending = self.gdp * 0.65  # Assume 65% of GDP is consumer spending
            total_revenue += consumer_spending * self.taxes["sales_tax"].rate
        
        # Property tax
        if "property_tax" in self.taxes:
            property_value = self.gdp * 2.5  # Assume property value is 2.5x GDP
            total_revenue += property_value * self.taxes["property_tax"].rate
        
        return total_revenue
    
    def elect_new_government(self) -> None:
        """Simulate an election with potential changes in policies"""
        # This would be more complex in a real simulation
        # For now, just make random changes to policies
        
        # Small adjustments to tax rates
        for tax in self.taxes.values():
            change = (random.random() - 0.5) * 0.05  # -2.5% to +2.5% change
            tax.rate = max(0, min(1, tax.rate + change))
        
        # Adjust department budget allocations
        total_budget = self.government_budget
        new_allocations = [random.random() for _ in range(len(self.departments))]
        allocation_sum = sum(new_allocations)
        
        for i, dept_name in enumerate(self.departments):
            allocation_ratio = new_allocations[i] / allocation_sum
            self.departments[dept_name].budget = total_budget * allocation_ratio
        
        # Impact on political stability depending on happiness
        if self.happiness < 0.4:
            self.political_stability *= 0.9  # Decline in stability
        elif self.happiness > 0.7:
            self.political_stability = min(1.0, self.political_stability * 1.1)
            
        # Random event: government scandal
        if random.random() < 0.1:  # 10% chance
            self.political_stability *= 0.85
            self.happiness *= 0.9

class GameState:
    def __init__(self):
        self.game_mode = None  # "country" or "business"
        self.current_date = datetime.datetime.now()
        self.countries: Dict[str, Country] = {}
        self.businesses: Dict[str, Business] = {}
        self.player_country_id = None
        self.player_business_id = None
        self.game_speed = 1  # Days per update
        self.paused = True
        
        # Initialize resource and stock markets
        self.global_markets = {
            "oil": ResourceMarket(name="Oil", current_price=60.0),
            "gold": ResourceMarket(name="Gold", current_price=1500.0),
            "wheat": ResourceMarket(name="Wheat", current_price=180.0),
            "stocks": StockExchange(name="Global Stock Index", country="Global")
        }
        
        # List of available presets
        self.country_presets = self._generate_country_presets()
        self.business_presets = self._generate_business_presets()
        
    def _generate_country_presets(self) -> Dict[str, Dict[str, Any]]:
        """Generate country presets based on real-world data"""
        return {
            "usa": {
                "name": "United States",
                "gdp": 22000000000000,  # $22 trillion
                "population": 330000000,
                "land_area": 9833517,
                "government_type": "Democracy",
                "gdp_growth": 0.024,
                "inflation_rate": 0.021,
                "unemployment_rate": 0.038,
                "tax_rate": 0.32,
                "national_debt": 30000000000000,  # $30 trillion
                "credit_rating": "AA+",
                "currency": "US Dollar",
                "currency_strength": 1.3,
                "political_stability": 0.85,
            },
            "china": {
                "name": "China",
                "gdp": 16000000000000,  # $16 trillion
                "population": 1400000000,
                "land_area": 9596960,
                "government_type": "Single-party",
                "gdp_growth": 0.058,
                "inflation_rate": 0.025,
                "unemployment_rate": 0.042,
                "tax_rate": 0.30,
                "national_debt": 8000000000000,  # $8 trillion
                "credit_rating": "A+",
                "currency": "Yuan",
                "currency_strength": 0.9,
                "political_stability": 0.9,
            },
            "germany": {
                "name": "Germany",
                "gdp": 4000000000000,  # $4 trillion
                "population": 83000000,
                "land_area": 357022,
                "government_type": "Democracy",
                "gdp_growth": 0.015,
                "inflation_rate": 0.016,
                "unemployment_rate": 0.032,
                "tax_rate": 0.39,
                "national_debt": 2500000000000,  # $2.5 trillion
                "credit_rating": "AAA",
                "currency": "Euro",
                "currency_strength": 1.2,
                "political_stability": 0.9,
            },
            "japan": {
                "name": "Japan",
                "gdp": 5000000000000,  # $5 trillion
                "population": 126000000,
                "land_area": 377975,
                "government_type": "Democracy",
                "gdp_growth": 0.009,
                "inflation_rate": 0.008,
                "unemployment_rate": 0.026,
                "tax_rate": 0.35,
                "national_debt": 12000000000000,  # $12 trillion
                "credit_rating": "A+",
                "currency": "Yen",
                "currency_strength": 1.0,
                "political_stability": 0.85,
            },
            "india": {
                "name": "India",
                "gdp": 3000000000000,  # $3 trillion
                "population": 1380000000,
                "land_area": 3287263,
                "government_type": "Democracy",
                "gdp_growth": 0.065,
                "inflation_rate": 0.042,
                "unemployment_rate": 0.078,
                "tax_rate": 0.28,
                "national_debt": 2200000000000,  # $2.2 trillion
                "credit_rating": "BBB-",
                "currency": "Rupee",
                "currency_strength": 0.7,
                "political_stability": 0.75,
            },
        }
    
    def _generate_business_presets(self) -> Dict[str, Dict[str, Any]]:
        """Generate business presets based on real-world companies"""
        return {
            "apple": {
                "name": "Apple Inc.",
                "founding_date": "1976-04-01",
                "ceo": "Tim Cook",
                "headquarters_country": "United States",
                "headquarters_city": "Cupertino",
                "cash": 50000000000,  # $50 billion in cash
                "debt": 100000000000,  # $100 billion in debt
                "credit_rating": "AA+",
                "employees": 154000,
                "total_assets": 350000000000,  # $350 billion
                "total_liabilities": 280000000000,  # $280 billion
                "is_public": True,
                "ipo_date": "1980-12-12",
                "stock_exchange": "NASDAQ",
                "sectors": ["Technology", "Consumer Electronics", "Software"],
                "brand_value": 9.5,  # Very high brand value
                "products": [
                    {
                        "name": "iPhone",
                        "category": "Smartphones",
                        "price": 999,
                        "cost": 400,
                        "quality": 0.9,
                        "brand_awareness": 0.95,
                    },
                    {
                        "name": "MacBook",
                        "category": "Computers",
                        "price": 1299,
                        "cost": 700,
                        "quality": 0.88,
                        "brand_awareness": 0.9,
                    },
                ],
            },
            "microsoft": {
                "name": "Microsoft Corporation",
                "founding_date": "1975-04-04",
                "ceo": "Satya Nadella",
                "headquarters_country": "United States",
                "headquarters_city": "Redmond",
                "cash": 136000000000,  # $136 billion in cash
                "debt": 70000000000,  # $70 billion in debt
                "credit_rating": "AAA",
                "employees": 181000,
                "total_assets": 301000000000,  # $301 billion
                "total_liabilities": 183000000000,  # $183 billion
                "is_public": True,
                "ipo_date": "1986-03-13",
                "stock_exchange": "NASDAQ",
                "sectors": ["Technology", "Software", "Cloud Computing"],
                "brand_value": 9.2,
                "products": [
                    {
                        "name": "Windows",
                        "category": "Operating Systems",
                        "price": 199,
                        "cost": 30,
                        "quality": 0.85,
                        "brand_awareness": 0.95,
                    },
                    {
                        "name": "Xbox",
                        "category": "Gaming Consoles",
                        "price": 499,
                        "cost": 350,
                        "quality": 0.87,
                        "brand_awareness": 0.85,
                    },
                ],
            },
            "tesla": {
                "name": "Tesla, Inc.",
                "founding_date": "2003-07-01",
                "ceo": "Elon Musk",
                "headquarters_country": "United States",
                "headquarters_city": "Austin",
                "cash": 17000000000,  # $17 billion
                "debt": 13000000000,  # $13 billion
                "credit_rating": "BB",
                "employees": 99290,
                "total_assets": 62000000000,  # $62 billion
                "total_liabilities": 30000000000,  # $30 billion
                "is_public": True,
                "ipo_date": "2010-06-29",
                "stock_exchange": "NASDAQ",
                "sectors": ["Automotive", "Energy", "Technology"],
                "brand_value": 8.5,
                "products": [
                    {
                        "name": "Model 3",
                        "category": "Electric Vehicles",
                        "price": 42000,
                        "cost": 35000,
                        "quality": 0.87,
                        "brand_awareness": 0.9,
                    },
                    {
                        "name": "Powerwall",
                        "category": "Energy Storage",
                        "price": 8500,
                        "cost": 5000,
                        "quality": 0.85,
                        "brand_awareness": 0.75,
                    },
                ],
            },
            "samsung": {
                "name": "Samsung Electronics",
                "founding_date": "1969-01-13",
                "ceo": "Kim Ki-nam",
                "headquarters_country": "South Korea",
                "headquarters_city": "Suwon",
                "cash": 84000000000,  # $84 billion
                "debt": 21000000000,  # $21 billion
                "credit_rating": "AA-",
                "employees": 267937,
                "total_assets": 304000000000,  # $304 billion
                "total_liabilities": 89000000000,  # $89 billion
                "is_public": True,
                "ipo_date": "1975-06-11",
                "stock_exchange": "KRX",
                "sectors": ["Technology", "Consumer Electronics", "Semiconductors"],
                "brand_value": 8.1,
                "products": [
                    {
                        "name": "Galaxy S23",
                        "category": "Smartphones",
                        "price": 899,
                        "cost": 350,
                        "quality": 0.86,
                        "brand_awareness": 0.88,
                    },
                    {
                        "name": "QLED TV",
                        "category": "Televisions",
                        "price": 1499,
                        "cost": 800,
                        "quality": 0.88,
                        "brand_awareness": 0.85,
                    },
                ],
            },
        }
        
    def start_country_mode(self, country_data: Dict[str, Any]) -> str:
        """Start a new game in country mode"""
        self.game_mode = "country"
        
        # Create the player's country
        new_country = Country(**country_data)
        new_country.controlled_by_player = True
        
        country_id = str(uuid.uuid4())
        self.countries[country_id] = new_country
        self.player_country_id = country_id
        
        return country_id
    
    def start_business_mode(self, business_data: Dict[str, Any]) -> str:
        """Start a new game in business mode"""
        self.game_mode = "business"
        
        # Create products if provided
        products = []
        if "products" in business_data:
            product_dicts = business_data.pop("products")
            for product_dict in product_dicts:
                products.append(Product(**product_dict))
        
        # Create the business
        new_business = Business(**business_data)
        new_business.products = products
        
        business_id = str(uuid.uuid4())
        self.businesses[business_id] = new_business
        self.player_business_id = business_id
        
        return business_id
    
    def add_ai_country(self, country_data: Dict[str, Any]) -> str:
        """Add an AI-controlled country to the simulation"""
        new_country = Country(**country_data)
        country_id = str(uuid.uuid4())
        self.countries[country_id] = new_country
        return country_id
    
    def add_business_to_country(self, business_data: Dict[str, Any], country_id: str) -> str:
        """Add a business to a specific country"""
        if country_id not in self.countries:
            raise ValueError(f"Country with ID {country_id} not found")
            
        # Create the business
        products = []
        if "products" in business_data:
            product_dicts = business_data.pop("products")
            for product_dict in product_dicts:
                products.append(Product(**product_dict))
            
        new_business = Business(**business_data)
        new_business.products = products
        
        business_id = str(uuid.uuid4())
        self.businesses[business_id] = new_business
        
        # Add to country's businesses
        self.countries[country_id].businesses[business_id] = new_business
        
        return business_id
    
    def add_stock_exchange(self, name: str, country_id: str) -> bool:
        """Create a new stock exchange in the specified country"""
        if country_id not in self.countries:
            return False
            
        country = self.countries[country_id]
        exchange = StockExchange(
            name=name,
            country=country.name
        )
        
        country.stock_exchanges[name] = exchange
        return True
    
    def advance_time(self, days: int = 1) -> None:
        """Advance simulation by specified number of days"""
        if self.paused:
            return
            
        for _ in range(days):
            self.current_date += datetime.timedelta(days=1)
            
            # Simulate all countries
            for country in self.countries.values():
                country.simulate_day()
            if self.game_mode == "business":
                # It is business mode
                for biz in self.businesses.values():
                    # For businesses not inside countries
                    # (player business lives here!)
                    sim_days_passed = (self.current_date - self.start_date).days  # store start_date accordingly
                    # or keep an integer counter incremented in advance_time()
                    if sim_days_passed % 90 == 0:   # approx every quarter
                        biz.simulate_quarter()
            
            # Update global markets
            for market in self.global_markets.values():
                if isinstance(market, ResourceMarket):
                    market.update_price(self.current_date.strftime("%Y-%m-%d"))
                elif isinstance(market, StockExchange):
                    market.update_index(self.current_date.strftime("%Y-%m-%d"))
    
    def save_game(self, filepath: str) -> bool:
        """Save the current game state to a file"""
        try:
            # Convert dataclasses to dictionaries
            data = {
                "game_mode": self.game_mode,
                "current_date": self.current_date.isoformat(),
                "game_speed": self.game_speed,
                "paused": self.paused,
                "player_country_id": self.player_country_id,
                "player_business_id": self.player_business_id,
                "countries": {k: asdict(v) for k, v in self.countries.items()},
                "businesses": {k: asdict(v) for k, v in self.businesses.items()},
                "global_markets": {k: asdict(v) for k, v in self.global_markets.items() 
                                  if isinstance(v, (ResourceMarket, StockExchange))}
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving game: {e}")
            return False
    
    def load_game(self, filepath: str) -> bool:
        """Load a game state from a file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.game_mode = data["game_mode"]
            self.current_date = datetime.datetime.fromisoformat(data["current_date"])
            self.game_speed = data["game_speed"]
            self.paused = data["paused"]
            self.player_country_id = data["player_country_id"]
            self.player_business_id = data["player_business_id"]
            
            # Recreate countries (complex due to nested dataclasses)
            self.countries = {}
            for country_id, country_dict in data["countries"].items():
                # Recreate departments
                departments = {}
                for dept_name, dept_dict in country_dict.pop("departments").items():
                    departments[dept_name] = Department(**dept_dict)
                
                # Recreate taxes
                taxes = {}
                for tax_name, tax_dict in country_dict.pop("taxes").items():
                    taxes[tax_name] = Tax(**tax_dict)
                
                # Recreate housing market
                housing_dict = country_dict.pop("housing_market")
                housing_market = HousingMarket(**housing_dict)
                
                # Recreate resource markets
                resource_markets = {}
                for resource, market_dict in country_dict.pop("resource_markets").items():
                    resource_markets[resource] = ResourceMarket(**market_dict)
                
                # Recreate stock exchanges
                stock_exchanges = {}
                for exchange_name, exchange_dict in country_dict.pop("stock_exchanges").items():
                    stock_exchanges[exchange_name] = StockExchange(**exchange_dict)
                
                # Create the country
                country = Country(**country_dict)
                country.departments = departments
                country.taxes = taxes
                country.housing_market = housing_market
                country.resource_markets = resource_markets
                country.stock_exchanges = stock_exchanges
                
                self.countries[country_id] = country
            
            # Recreate businesses
            self.businesses = {}
            for business_id, business_dict in data["businesses"].items():
                # Recreate products
                products = []
                for product_dict in business_dict.pop("products"):
                    products.append(Product(**product_dict))
                
                # Create the business
                business = Business(**business_dict)
                business.products = products
                
                self.businesses[business_id] = business
            
            # Recreate global markets
            self.global_markets = {}
            for market_name, market_dict in data["global_markets"].items():
                if "index_value" in market_dict:
                    self.global_markets[market_name] = StockExchange(**market_dict)
                else:
                    self.global_markets[market_name] = ResourceMarket(**market_dict)
            
            return True
        except Exception as e:
            print(f"Error loading game: {e}")
            return False

    def get_player_entity(self) -> Union[Country, Business, None]:
        """Get the entity (country or business) controlled by the player"""
        if self.game_mode == "country" and self.player_country_id:
            return self.countries.get(self.player_country_id)
        elif self.game_mode == "business" and self.player_business_id:
            return self.businesses.get(self.player_business_id)
        return None

class MyEconomyApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Set customtkinter appearance
        ctk.set_appearance_mode("System")  # "System", "Dark" or "Light"
        ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
        
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1280x720")
        self.minsize(1000, 600)
        
        # Initialize game state
        self.game_state = GameState()
        
        # Store our interface frames
        self.current_frame = None
        self.frames = {}
        
        # Set up the navigation sidebar
        self.setup_navigation()
        
        # Initially show the start screen
        self.show_frame("start")
        
        # Set up a timer for game simulation updates
        self.update_timer_id = None
        self.start_update_timer()
    
    def setup_navigation(self):
        # Create a frame for the sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        
        # App title in sidebar
        ctk.CTkLabel(
            self.sidebar, 
            text=APP_NAME, 
            font=("Helvetica", 20, "bold")
        ).pack(pady=20, padx=10)
        
        # Initialize frames
        self.frames["start"] = StartScreen(self)
        self.frames["country_setup"] = CountrySetupScreen(self)
        self.frames["business_setup"] = BusinessSetupScreen(self)
        self.frames["country_main"] = CountryMainScreen(self)
        self.frames["business_main"] = BusinessMainScreen(self)
        
        # Create navigation buttons - initially hidden until game starts
        self.nav_buttons = {}
        
        # Add buttons that will be shown once game starts
        nav_options = [
            ("Overview", "overview", self.show_overview),
            ("Economy", "economy", self.show_economy),
            ("Government", "government", self.show_government),
            ("International", "international", self.show_international),
            ("Development", "development", self.show_development),
            ("Settings", "settings", self.show_settings),
        ]
        
        for text, name, command in nav_options:
            btn = ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                height=35,
                corner_radius=5,
                anchor="center",
            )
            btn.pack(padx=20, pady=5, fill=tk.X)
            self.nav_buttons[name] = btn
            # Hide buttons initially
            btn.pack_forget()
        
        # Quit button always visible
        self.quit_button = ctk.CTkButton(
            self.sidebar,
            text="Quit Game",
            command=self.quit_game,
            fg_color=DANGER_COLOR,
            hover_color="#c0392b",
            height=35,
        )
        self.quit_button.pack(padx=20, pady=5, fill=tk.X, side=tk.BOTTOM)
        
        # Save button
        self.save_button = ctk.CTkButton(
            self.sidebar,
            text="Save Game",
            command=self.save_game,
            fg_color=PRIMARY_COLOR,
            height=35,
        )
        self.save_button.pack(padx=20, pady=5, fill=tk.X, side=tk.BOTTOM)
        self.save_button.pack_forget()  # Hide initially
        
        # Time controls frame
        self.time_frame = ctk.CTkFrame(self.sidebar)
        self.time_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.time_frame.pack_forget()  # Hide initially
        
        # Date display
        self.date_label = ctk.CTkLabel(
            self.time_frame, 
            text="Date: --/--/----",
            font=("Helvetica", 12)
        )
        self.date_label.pack(pady=5)
        
        # Speed control
        speed_frame = ctk.CTkFrame(self.time_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(speed_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        
        self.speed_var = tk.IntVar(value=1)
        speeds = [("1×", 1), ("2×", 2), ("5×", 5), ("10×", 10)]
        
        for text, value in speeds:
            ctk.CTkRadioButton(
                speed_frame,
                text=text,
                variable=self.speed_var,
                value=value,
                command=self.change_game_speed
            ).pack(side=tk.LEFT, padx=2)
        
        # Pause/Play button
        self.pause_button = ctk.CTkButton(
            self.time_frame,
            text="Pause",
            command=self.toggle_pause,
            width=80,
            height=25
        )
        self.pause_button.pack(pady=5)
    
    def show_frame(self, frame_name):
        # Hide current frame if any
        if self.current_frame:
            self.current_frame.pack_forget()
        
        # Show the requested frame
        frame = self.frames.get(frame_name)
        if frame:
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.current_frame = frame
            
            # Update navigation visibility based on game state
            self.update_navigation()
    
    def update_navigation(self):
        # Show/hide navigation based on game state
        in_game = self.game_state.game_mode is not None
        
        # Show navigation buttons and time controls in-game
        if in_game:
            # Show the navigation buttons relevant to the game mode
            if self.game_state.game_mode == "country":
                for name in ["overview", "economy", "government", "international", "development"]:
                    self.nav_buttons[name].pack(padx=20, pady=5, fill=tk.X)
            elif self.game_state.game_mode == "business":
                for name in ["overview", "economy", "development"]:
                    self.nav_buttons[name].pack(padx=20, pady=5, fill=tk.X)
            
            # Always show settings
            self.nav_buttons["settings"].pack(padx=20, pady=5, fill=tk.X)
            
            # Show time controls and save button
            self.time_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
            self.save_button.pack(padx=20, pady=5, fill=tk.X, side=tk.BOTTOM)
        else:
            # Hide all navigation except quit
            for btn in self.nav_buttons.values():
                btn.pack_forget()
                
            # Hide time controls and save button
            self.time_frame.pack_forget()
            self.save_button.pack_forget()
    
    def start_country_game(self, country_data):
        # Create the player country
        self.game_state.start_country_mode(country_data)
        
        # Add some AI countries for competition/relations
        for preset_name, preset_data in self.game_state.country_presets.items():
            # Don't add the player's country
            if preset_data["name"] != country_data["name"]:
                self.game_state.add_ai_country(preset_data)
        
        # Start simulation
        self.game_state.paused = False
        
        # Show the main country screen
        self.show_frame("country_main")
    
    def start_business_game(self, business_data):
        # Create the player business
        self.game_state.start_business_mode(business_data)
        
        # Add some countries with economies
        for preset_name, preset_data in self.game_state.country_presets.items():
            country_id = self.game_state.add_ai_country(preset_data)
            
            # Add some AI businesses to these countries
            for i in range(3):  # Add 3 random businesses per country
                if self.game_state.business_presets:
                    # Choose random preset and modify slightly
                    preset_key = random.choice(list(self.game_state.business_presets.keys()))
                    preset = self.game_state.business_presets[preset_key].copy()
                    preset["name"] = f"{preset['name']} {preset_data['name']} Division"
                    preset["headquarters_country"] = preset_data["name"]
                    
                    # Add to the country
                    self.game_state.add_business_to_country(preset, country_id)
            
            # Add stock exchange to this country
            self.game_state.add_stock_exchange(f"{preset_data['name']} Stock Exchange", country_id)
        
        # Start simulation
        self.game_state.paused = False
        
        # Show the main business screen
        self.show_frame("business_main")
    
    def update_simulation(self):
        """Regular updates for the game simulation"""
        if self.game_state and not self.game_state.paused:
            # Advance the simulation based on game speed
            self.game_state.advance_time(self.game_state.game_speed)
            
            # Update the UI to reflect changes
            self.update_ui()
        
        # Schedule the next update
        self.update_timer_id = self.after(1000, self.update_simulation)  # Update every second
    
    def update_ui(self):
        """Update UI elements with current game data"""
        # Update date display
        if self.game_state.current_date:
            date_str = self.game_state.current_date.strftime("%d %b %Y")
            self.date_label.configure(text=f"Date: {date_str}")
        
        # If in game, update current frame
        if self.current_frame and hasattr(self.current_frame, "update_display"):
            self.current_frame.update_display()
    
    def start_update_timer(self):
        """Start the update timer for game simulation"""
        if self.update_timer_id:
            self.after_cancel(self.update_timer_id)
        self.update_timer_id = self.after(1000, self.update_simulation)
    
    def change_game_speed(self):
        """Change the simulation speed"""
        new_speed = self.speed_var.get()
        if self.game_state:
            self.game_state.game_speed = new_speed
    
    def toggle_pause(self):
        """Pause or unpause the game"""
        if self.game_state:
            self.game_state.paused = not self.game_state.paused
            
            # Update button text
            if self.game_state.paused:
                self.pause_button.configure(text="Play")
            else:
                self.pause_button.configure(text="Pause")
    
    def save_game(self):
        """Save the current game state"""
        if not self.game_state or self.game_state.game_mode is None:
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("myEconomy Save Files", "*.json"), ("All Files", "*.*")],
            title="Save Game"
        )
        
        if filename:
            success = self.game_state.save_game(filename)
            if success:
                messagebox.showinfo("Save Game", "Game saved successfully")
            else:
                messagebox.showerror("Save Error", "Failed to save the game")
    
    def load_game(self):
        """Load a saved game state"""
        filename = filedialog.askopenfilename(
            filetypes=[("myEconomy Save Files", "*.json"), ("All Files", "*.*")],
            title="Load Game"
        )
        
        if filename:
            success = self.game_state.load_game(filename)
            if success:
                # Determine which screen to show based on game mode
                if self.game_state.game_mode == "country":
                    self.show_frame("country_main")
                elif self.game_state.game_mode == "business":
                    self.show_frame("business_main")
                
                messagebox.showinfo("Load Game", "Game loaded successfully")
            else:
                messagebox.showerror("Load Error", "Failed to load the game")
    
    # Navigation command handlers
    def show_overview(self):
        if self.game_state.game_mode == "country":
            # Switch to country overview tab in country main screen
            if isinstance(self.current_frame, CountryMainScreen):
                self.current_frame.show_tab("overview")
        else:
            # Switch to business overview tab in business main screen
            if isinstance(self.current_frame, BusinessMainScreen):
                self.current_frame.show_tab("overview")
    
    def show_economy(self):
        if self.game_state.game_mode == "country":
            # Switch to economy tab in country main screen
            if isinstance(self.current_frame, CountryMainScreen):
                self.current_frame.show_tab("economy")
        else:
            # Switch to finances tab in business main screen
            if isinstance(self.current_frame, BusinessMainScreen):
                self.current_frame.show_tab("finances")
    
    def show_government(self):
        if self.game_state.game_mode == "country":
            # Switch to government tab in country main screen
            if isinstance(self.current_frame, CountryMainScreen):
                self.current_frame.show_tab("government")
    
    def show_international(self):
        if self.game_state.game_mode == "country":
            # Switch to international tab in country main screen
            if isinstance(self.current_frame, CountryMainScreen):
                self.current_frame.show_tab("international")
    
    def show_development(self):
        if self.game_state.game_mode == "country":
            # Switch to development tab in country main screen
            if isinstance(self.current_frame, CountryMainScreen):
                self.current_frame.show_tab("development")
        else:
            # Switch to development tab in business main screen
            if isinstance(self.current_frame, BusinessMainScreen):
                self.current_frame.show_tab("research")
    
    def show_settings(self):
        # Show the settings dialog
        SettingsDialog(self)
    
    def quit_game(self):
        """Prompt to save and exit the game"""
        if self.game_state and self.game_state.game_mode is not None:
            result = messagebox.askyesnocancel("Quit Game", "Save game before quitting?")
            if result is None:  # Cancel
                return
            elif result:  # Yes, save
                self.save_game()
        
        self.destroy()

class StartScreen(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.app = master
        
        # Center content
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        content_frame = ctk.CTkFrame(self)
        content_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Title and logo
        ctk.CTkLabel(
            content_frame, 
            text=APP_NAME, 
            font=("Helvetica", 36, "bold")
        ).pack(pady=20)
        
        ctk.CTkLabel(
            content_frame,
            text="A Geopolitical and Business Simulator",
            font=("Helvetica", 16)
        ).pack(pady=10)
        
        # Game mode selection
        mode_frame = ctk.CTkFrame(content_frame)
        mode_frame.pack(pady=30, padx=10, fill=tk.X)
        
        # Country Mode button
        ctk.CTkButton(
            mode_frame,
            text="Country Mode",
            font=("Helvetica", 16),
            height=60,
            command=self.select_country_mode,
            fg_color=PRIMARY_COLOR
        ).pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        # Business Mode button
        ctk.CTkButton(
            mode_frame,
            text="Business Mode",
            font=("Helvetica", 16),
            height=60,
            command=self.select_business_mode,
            fg_color=SECONDARY_COLOR
        ).pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        # Load game button
        ctk.CTkButton(
            content_frame,
            text="Load Saved Game",
            font=("Helvetica", 16),
            height=40,
            command=self.app.load_game
        ).pack(pady=20, padx=40, fill=tk.X)
        
        # Version info
        ctk.CTkLabel(
            content_frame,
            text=f"Version {APP_VERSION}",
            font=("Helvetica", 10)
        ).pack(pady=10)
    
    def select_country_mode(self):
        self.app.show_frame("country_setup")
    
    def select_business_mode(self):
        self.app.show_frame("business_setup")

class CountrySetupScreen(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.app = master
        
        # Country data
        self.country_data = {}
        self.selected_preset = tk.StringVar(value="custom")
        
        # Layout
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        ctk.CTkLabel(
            self, 
            text="Set Up Your Country", 
            font=HEADER_FONT
        ).pack(pady=(20, 10))
        
        # Preset selection
        ctk.CTkLabel(
            self,
            text="Select a preset or create a custom country:"
        ).pack(pady=(10, 0), padx=20, anchor="w")
        
        presets_frame = ctk.CTkFrame(self)
        presets_frame.pack(padx=20, pady=10, fill=tk.X)
        
        # Custom option
        ctk.CTkRadioButton(
            presets_frame,
            text="Custom Country",
            variable=self.selected_preset,
            value="custom",
            command=self.preset_selected
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Add preset options
        for preset_key, preset_data in self.app.game_state.country_presets.items():
            ctk.CTkRadioButton(
                presets_frame,
                text=preset_data["name"],
                variable=self.selected_preset,
                value=preset_key,
                command=self.preset_selected
            ).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Scrollable area for country settings
        self.settings_canvas = ctk.CTkScrollableFrame(self)
        self.settings_canvas.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # Basic country form
        self.create_country_form()
        
        # Buttons at bottom
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(padx=20, pady=15, fill=tk.X)
        
        ctk.CTkButton(
            button_frame,
            text="Back",
            command=lambda: self.app.show_frame("start"),
            fg_color=SECONDARY_COLOR,
            width=100
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Start Game",
            command=self.start_game,
            fg_color=SUCCESS_COLOR,
            width=150
        ).pack(side=tk.RIGHT, padx=10)
    
    def create_country_form(self):
        # Clear existing widgets
        for widget in self.settings_canvas.winfo_children():
            widget.destroy()
        
        self.country_fields = {}
        
        # Basic country info frame
        basics_frame = ctk.CTkFrame(self.settings_canvas)
        basics_frame.pack(padx=10, pady=10, fill=tk.X)
        
        ctk.CTkLabel(
            basics_frame,
            text="Country Basics",
            font=("Helvetica", 14, "bold")
        ).pack(pady=10)
        
        # Country name
        field_frame = ctk.CTkFrame(basics_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Country Name:").pack(side=tk.LEFT, padx=10)
        name_entry = ctk.CTkEntry(field_frame, width=250)
        name_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.country_fields["name"] = name_entry
        
        # Government type
        field_frame = ctk.CTkFrame(basics_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Government Type:").pack(side=tk.LEFT, padx=10)
        gov_types = ["Democracy", "Republic", "Monarchy", "Dictatorship", "Oligarchy", "Theocracy"]
        gov_type = ctk.CTkOptionMenu(field_frame, values=gov_types)
        gov_type.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.country_fields["government_type"] = gov_type
        
        # Economic frame
        economy_frame = ctk.CTkFrame(self.settings_canvas)
        economy_frame.pack(padx=10, pady=10, fill=tk.X)
        
        ctk.CTkLabel(
            economy_frame,
            text="Economy",
            font=("Helvetica", 14, "bold")
        ).pack(pady=10)
        
        # GDP
        field_frame = ctk.CTkFrame(economy_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="GDP (billions $):").pack(side=tk.LEFT, padx=10)
        gdp_entry = ctk.CTkEntry(field_frame, width=150)
        gdp_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.country_fields["gdp"] = gdp_entry
        
        # Population
        field_frame = ctk.CTkFrame(economy_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Population (millions):").pack(side=tk.LEFT, padx=10)
        pop_entry = ctk.CTkEntry(field_frame, width=150)
        pop_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.country_fields["population"] = pop_entry
        
        # GDP Growth
        field_frame = ctk.CTkFrame(economy_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="GDP Growth Rate (%):").pack(side=tk.LEFT, padx=10)
        growth_entry = ctk.CTkEntry(field_frame, width=150)
        growth_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.country_fields["gdp_growth"] = growth_entry
        
        # Inflation
        field_frame = ctk.CTkFrame(economy_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Inflation Rate (%):").pack(side=tk.LEFT, padx=10)
        inflation_entry = ctk.CTkEntry(field_frame, width=150)
        inflation_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.country_fields["inflation_rate"] = inflation_entry
        
        # Unemployment
        field_frame = ctk.CTkFrame(economy_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Unemployment Rate (%):").pack(side=tk.LEFT, padx=10)
        unemployment_entry = ctk.CTkEntry(field_frame, width=150)
        unemployment_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.country_fields["unemployment_rate"] = unemployment_entry
        
        # Geography frame
        geo_frame = ctk.CTkFrame(self.settings_canvas)
        geo_frame.pack(padx=10, pady=10, fill=tk.X)
        
        ctk.CTkLabel(
            geo_frame,
            text="Geography",
            font=("Helvetica", 14, "bold")
        ).pack(pady=10)
        
        # Land area
        field_frame = ctk.CTkFrame(geo_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Land Area (sq km):").pack(side=tk.LEFT, padx=10)
        area_entry = ctk.CTkEntry(field_frame, width=150)
        area_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.country_fields["land_area"] = area_entry
        
        # Financial frame
        finance_frame = ctk.CTkFrame(self.settings_canvas)
        finance_frame.pack(padx=10, pady=10, fill=tk.X)
        
        ctk.CTkLabel(
            finance_frame,
            text="Government Finances",
            font=("Helvetica", 14, "bold")
        ).pack(pady=10)
        
        # Tax rate
        field_frame = ctk.CTkFrame(finance_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Overall Tax Rate (%):").pack(side=tk.LEFT, padx=10)
        tax_entry = ctk.CTkEntry(field_frame, width=150)
        tax_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.country_fields["tax_rate"] = tax_entry
        
        # National debt
        field_frame = ctk.CTkFrame(finance_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="National Debt (billions $):").pack(side=tk.LEFT, padx=10)
        debt_entry = ctk.CTkEntry(field_frame, width=150)
        debt_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.country_fields["national_debt"] = debt_entry
    
    def preset_selected(self):
        preset_key = self.selected_preset.get()
        
        if preset_key == "custom":
            # Clear fields
            for field in self.country_fields.values():
                if isinstance(field, ctk.CTkEntry):
                    field.delete(0, tk.END)
                elif isinstance(field, ctk.CTkOptionMenu):
                    field.set(field._values[0])
        else:
            # Load preset data
            preset_data = self.app.game_state.country_presets[preset_key]
            
            # Format and set values
            self.country_fields["name"].delete(0, tk.END)
            self.country_fields["name"].insert(0, preset_data["name"])
            
            self.country_fields["government_type"].set(preset_data["government_type"])
            
            self.country_fields["gdp"].delete(0, tk.END)
            self.country_fields["gdp"].insert(0, str(preset_data["gdp"] / 1000000000))  # Convert to billions
            
            self.country_fields["population"].delete(0, tk.END)
            self.country_fields["population"].insert(0, str(preset_data["population"] / 1000000))  # Convert to millions
            
            self.country_fields["gdp_growth"].delete(0, tk.END)
            self.country_fields["gdp_growth"].insert(0, str(preset_data["gdp_growth"] * 100))  # Convert to percentage
            
            self.country_fields["inflation_rate"].delete(0, tk.END)
            self.country_fields["inflation_rate"].insert(0, str(preset_data["inflation_rate"] * 100))
            
            self.country_fields["unemployment_rate"].delete(0, tk.END)
            self.country_fields["unemployment_rate"].insert(0, str(preset_data["unemployment_rate"] * 100))
            
            self.country_fields["land_area"].delete(0, tk.END)
            self.country_fields["land_area"].insert(0, str(preset_data["land_area"]))
            
            self.country_fields["tax_rate"].delete(0, tk.END)
            self.country_fields["tax_rate"].insert(0, str(preset_data["tax_rate"] * 100))
            
            self.country_fields["national_debt"].delete(0, tk.END)
            self.country_fields["national_debt"].insert(0, str(preset_data["national_debt"] / 1000000000))
    
    def validate_and_gather_data(self) -> Dict[str, Any]:
        """Validate form fields and gather into a dictionary"""
        data = {}
        errors = []
        
        try:
            # Basic text fields
            data["name"] = self.country_fields["name"].get().strip()
            if not data["name"]:
                errors.append("Country name is required")
            
            # Government type
            data["government_type"] = self.country_fields["government_type"].get()
            
            # Numerical fields
            gdp_billions = float(self.country_fields["gdp"].get())
            data["gdp"] = gdp_billions * 1000000000  # Convert to actual value
            
            pop_millions = float(self.country_fields["population"].get())
            data["population"] = int(pop_millions * 1000000)  # Convert to actual value
            
            growth_percent = float(self.country_fields["gdp_growth"].get())
            data["gdp_growth"] = growth_percent / 100  # Convert to decimal
            
            inflation_percent = float(self.country_fields["inflation_rate"].get())
            data["inflation_rate"] = inflation_percent / 100
            
            unemployment_percent = float(self.country_fields["unemployment_rate"].get())
            data["unemployment_rate"] = unemployment_percent / 100
            
            data["land_area"] = float(self.country_fields["land_area"].get())
            
            tax_percent = float(self.country_fields["tax_rate"].get())
            data["tax_rate"] = tax_percent / 100
            
            debt_billions = float(self.country_fields["national_debt"].get())
            data["national_debt"] = debt_billions * 1000000000
        
        except ValueError as e:
            errors.append(f"Invalid number format: {str(e)}")
        
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return None
            
        return data
    
    def start_game(self):
        """Validate and start the game with the configured country"""
        country_data = self.validate_and_gather_data()
        if country_data:
            self.app.start_country_game(country_data)

class BusinessSetupScreen(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.app = master
        
        # Business data
        self.business_data = {}
        self.selected_preset = tk.StringVar(value="custom")
        self.product_entries = []
        
        # Layout
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        ctk.CTkLabel(
            self, 
            text="Set Up Your Business", 
            font=HEADER_FONT
        ).pack(pady=(20, 10))
        
        # Preset selection
        ctk.CTkLabel(
            self,
            text="Select a preset or create a custom business:"
        ).pack(pady=(10, 0), padx=20, anchor="w")
        
        presets_frame = ctk.CTkFrame(self)
        presets_frame.pack(padx=20, pady=10, fill=tk.X)
        
        # Custom option
        ctk.CTkRadioButton(
            presets_frame,
            text="Custom Business",
            variable=self.selected_preset,
            value="custom",
            command=self.preset_selected
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Add preset options
        for preset_key, preset_data in self.app.game_state.business_presets.items():
            ctk.CTkRadioButton(
                presets_frame,
                text=preset_data["name"],
                variable=self.selected_preset,
                value=preset_key,
                command=self.preset_selected
            ).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Scrollable area for business settings
        self.settings_canvas = ctk.CTkScrollableFrame(self)
        self.settings_canvas.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # Business form
        self.create_business_form()
        
        # Buttons at bottom
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(padx=20, pady=15, fill=tk.X)
        
        ctk.CTkButton(
            button_frame,
            text="Back",
            command=lambda: self.app.show_frame("start"),
            fg_color=SECONDARY_COLOR,
            width=100
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Start Game",
            command=self.start_game,
            fg_color=SUCCESS_COLOR,
            width=150
        ).pack(side=tk.RIGHT, padx=10)
    
    def create_business_form(self):
        # Clear existing widgets
        for widget in self.settings_canvas.winfo_children():
            widget.destroy()
        
        self.business_fields = {}
        
        # Basic business info frame
        basics_frame = ctk.CTkFrame(self.settings_canvas)
        basics_frame.pack(padx=10, pady=10, fill=tk.X)
        
        ctk.CTkLabel(
            basics_frame,
            text="Business Basics",
            font=("Helvetica", 14, "bold")
        ).pack(pady=10)
        
        # Business name
        field_frame = ctk.CTkFrame(basics_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Business Name:").pack(side=tk.LEFT, padx=10)
        name_entry = ctk.CTkEntry(field_frame, width=250)
        name_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.business_fields["name"] = name_entry
        
        # CEO
        field_frame = ctk.CTkFrame(basics_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="CEO Name:").pack(side=tk.LEFT, padx=10)
        ceo_entry = ctk.CTkEntry(field_frame, width=250)
        ceo_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.business_fields["ceo"] = ceo_entry
        
        # Headquarters country
        field_frame = ctk.CTkFrame(basics_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="HQ Country:").pack(side=tk.LEFT, padx=10)
        hq_country_entry = ctk.CTkEntry(field_frame, width=150)
        hq_country_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.business_fields["headquarters_country"] = hq_country_entry
        
        # Headquarters city
        field_frame = ctk.CTkFrame(basics_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="HQ City:").pack(side=tk.LEFT, padx=10)
        hq_city_entry = ctk.CTkEntry(field_frame, width=150)
        hq_city_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.business_fields["headquarters_city"] = hq_city_entry
        
        # Financial frame
        finance_frame = ctk.CTkFrame(self.settings_canvas)
        finance_frame.pack(padx=10, pady=10, fill=tk.X)
        
        ctk.CTkLabel(
            finance_frame,
            text="Finances",
            font=("Helvetica", 14, "bold")
        ).pack(pady=10)
        
        # Cash
        field_frame = ctk.CTkFrame(finance_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Starting Cash (millions $):").pack(side=tk.LEFT, padx=10)
        cash_entry = ctk.CTkEntry(field_frame, width=150)
        cash_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.business_fields["cash"] = cash_entry
        
        # Employees
        field_frame = ctk.CTkFrame(finance_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Number of Employees:").pack(side=tk.LEFT, padx=10)
        employees_entry = ctk.CTkEntry(field_frame, width=150)
        employees_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.business_fields["employees"] = employees_entry
        
        # Total assets
        field_frame = ctk.CTkFrame(finance_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Total Assets (millions $):").pack(side=tk.LEFT, padx=10)
        assets_entry = ctk.CTkEntry(field_frame, width=150)
        assets_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.business_fields["total_assets"] = assets_entry
        
        # Sectors
        field_frame = ctk.CTkFrame(basics_frame)
        field_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ctk.CTkLabel(field_frame, text="Business Sectors:").pack(side=tk.LEFT, padx=10)
        sectors_entry = ctk.CTkEntry(field_frame, width=250)
        sectors_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        ctk.CTkLabel(field_frame, text="(comma separated)").pack(side=tk.LEFT, padx=10)
        self.business_fields["sectors"] = sectors_entry
        
        # Products Frame
        products_frame = ctk.CTkFrame(self.settings_canvas)
        products_frame.pack(padx=10, pady=10, fill=tk.X)
        
        ctk.CTkLabel(
            products_frame,
            text="Initial Products",
            font=("Helvetica", 14, "bold")
        ).pack(pady=10)
        
        # Container for product entries
        self.products_container = ctk.CTkFrame(products_frame)
        self.products_container.pack(padx=10, pady=5, fill=tk.X)
        
        # Initial product
        self.add_product_entry()
        
        # Add product button
        ctk.CTkButton(
            products_frame,
            text="+ Add Another Product",
            command=self.add_product_entry,
            fg_color=PRIMARY_COLOR,
            height=30
        ).pack(pady=10, padx=10)
    
    def add_product_entry(self):
        """Add fields for a new product"""
        product_frame = ctk.CTkFrame(self.products_container)
        product_frame.pack(padx=5, pady=5, fill=tk.X)
        
        # Product number label
        product_num = len(self.product_entries) + 1
        ctk.CTkLabel(
            product_frame,
            text=f"Product #{product_num}",
            font=("Helvetica", 12, "bold")
        ).pack(pady=5)
        
        # Product fields
        fields = {}
        
        # Name
        field_row = ctk.CTkFrame(product_frame)
        field_row.pack(padx=5, pady=2, fill=tk.X)
        
        ctk.CTkLabel(field_row, text="Name:").pack(side=tk.LEFT, padx=5)
        name_entry = ctk.CTkEntry(field_row, width=200)
        name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        fields["name"] = name_entry
        
        # Category
        field_row = ctk.CTkFrame(product_frame)
        field_row.pack(padx=5, pady=2, fill=tk.X)
        
        ctk.CTkLabel(field_row, text="Category:").pack(side=tk.LEFT, padx=5)
        category_entry = ctk.CTkEntry(field_row, width=150)
        category_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        fields["category"] = category_entry
        
        # Price and Cost row
        field_row = ctk.CTkFrame(product_frame)
        field_row.pack(padx=5, pady=2, fill=tk.X)
        
        # Price
        ctk.CTkLabel(field_row, text="Price ($):").pack(side=tk.LEFT, padx=5)
        price_entry = ctk.CTkEntry(field_row, width=100)
        price_entry.pack(side=tk.LEFT, padx=5)
        fields["price"] = price_entry
        
        # Cost
        ctk.CTkLabel(field_row, text="Cost ($):").pack(side=tk.LEFT, padx=5)
        cost_entry = ctk.CTkEntry(field_row, width=100)
        cost_entry.pack(side=tk.LEFT, padx=5)
        fields["cost"] = cost_entry
        
        # Quality and Brand Awareness
        field_row = ctk.CTkFrame(product_frame)
        field_row.pack(padx=5, pady=2, fill=tk.X)
        
        # Quality
        ctk.CTkLabel(field_row, text="Quality (0-1):").pack(side=tk.LEFT, padx=5)
        quality_entry = ctk.CTkEntry(field_row, width=100)
        quality_entry.pack(side=tk.LEFT, padx=5)
        fields["quality"] = quality_entry
        
        # Brand awareness
        ctk.CTkLabel(field_row, text="Brand Awareness (0-1):").pack(side=tk.LEFT, padx=5)
        awareness_entry = ctk.CTkEntry(field_row, width=100)
        awareness_entry.pack(side=tk.LEFT, padx=5)
        fields["brand_awareness"] = awareness_entry
        
        # Remove button
        if len(self.product_entries) > 0:  # Only add remove button if more than one product
            ctk.CTkButton(
                product_frame,
                text="Remove",
                command=lambda f=product_frame: self.remove_product_entry(f),
                fg_color=DANGER_COLOR,
                width=80,
                height=25
            ).pack(pady=5)
        
        self.product_entries.append((product_frame, fields))
    
    def remove_product_entry(self, frame):
        """Remove a product entry frame and update product numbers"""
        # Find the index of the frame
        for i, (prod_frame, _) in enumerate(self.product_entries):
            if prod_frame == frame:
                self.product_entries.pop(i)
                frame.destroy()
                break
        
        # Update product numbers
        for i, (prod_frame, _) in enumerate(self.product_entries):
            # Update the label of the first child (the product number label)
            for child in prod_frame.winfo_children():
                if isinstance(child, ctk.CTkLabel) and "Product #" in child.cget("text"):
                    child.configure(text=f"Product #{i+1}")
                    break
    
    def preset_selected(self):
        preset_key = self.selected_preset.get()
        
        # Clear current product entries except the first one
        while len(self.product_entries) > 1:
            frame, _ = self.product_entries[-1]
            self.remove_product_entry(frame)
            
        # Clear the first product entry fields
        if self.product_entries:
            _, fields = self.product_entries[0]
            for entry in fields.values():
                entry.delete(0, tk.END)
        
        if preset_key == "custom":
            # Clear fields
            for field in self.business_fields.values():
                field.delete(0, tk.END)
                
            # Default values
            self.business_fields["cash"].insert(0, "10")  # 10 million
            self.business_fields["employees"].insert(0, "50")
            self.business_fields["total_assets"].insert(0, "15")  # 15 million
        else:
            # Load preset data
            preset_data = self.app.game_state.business_presets[preset_key]
            
            # Format and set values
            self.business_fields["name"].delete(0, tk.END)
            self.business_fields["name"].insert(0, preset_data["name"])
            
            self.business_fields["ceo"].delete(0, tk.END)
            self.business_fields["ceo"].insert(0, preset_data["ceo"])
            
            self.business_fields["headquarters_country"].delete(0, tk.END)
            self.business_fields["headquarters_country"].insert(0, preset_data["headquarters_country"])
            
            self.business_fields["headquarters_city"].delete(0, tk.END)
            self.business_fields["headquarters_city"].insert(0, preset_data["headquarters_city"])
            
            self.business_fields["cash"].delete(0, tk.END)
            self.business_fields["cash"].insert(0, str(preset_data["cash"] / 1000000))  # Convert to millions
            
            self.business_fields["employees"].delete(0, tk.END)
            self.business_fields["employees"].insert(0, str(preset_data["employees"]))
            
            self.business_fields["total_assets"].delete(0, tk.END)
            self.business_fields["total_assets"].insert(0, str(preset_data["total_assets"] / 1000000))  # Convert to millions
            
            self.business_fields["sectors"].delete(0, tk.END)
            self.business_fields["sectors"].insert(0, ", ".join(preset_data["sectors"]))
            
            # Remove all existing product entries
            while self.product_entries:
                frame, _ = self.product_entries[-1]
                self.remove_product_entry(frame)
            
            # Add product entries for each preset product
            for product in preset_data.get("products", []):
                self.add_product_entry()
                _, fields = self.product_entries[-1]
                
                fields["name"].insert(0, product["name"])
                fields["category"].insert(0, product["category"])
                fields["price"].insert(0, str(product["price"]))
                fields["cost"].insert(0, str(product["cost"]))
                fields["quality"].insert(0, str(product["quality"]))
                fields["brand_awareness"].insert(0, str(product["brand_awareness"]))
    
    def validate_and_gather_data(self) -> Dict[str, Any]:
        """Validate form fields and gather into a dictionary"""
        data = {}
        errors = []
        
        try:
            # Basic text fields
            data["name"] = self.business_fields["name"].get().strip()
            if not data["name"]:
                errors.append("Business name is required")
            
            data["ceo"] = self.business_fields["ceo"].get().strip()
            if not data["ceo"]:
                data["ceo"] = "You"  # Default CEO name
            
            data["headquarters_country"] = self.business_fields["headquarters_country"].get().strip()
            data["headquarters_city"] = self.business_fields["headquarters_city"].get().strip()
            
            # Numerical fields
            cash_millions = float(self.business_fields["cash"].get())
            data["cash"] = cash_millions * 1000000  # Convert to actual value
            
            data["employees"] = int(self.business_fields["employees"].get())
            
            assets_millions = float(self.business_fields["total_assets"].get())
            data["total_assets"] = assets_millions * 1000000
            
            # Sectors
            sectors_text = self.business_fields["sectors"].get().strip()
            if sectors_text:
                data["sectors"] = [s.strip() for s in sectors_text.split(",") if s.strip()]
            else:
                data["sectors"] = []
                
            # Process products
            products = []
            for _, fields in self.product_entries:
                product = {}
                product_name = fields["name"].get().strip()
                
                if not product_name:  # Skip empty product entries
                    continue
                    
                product["name"] = product_name
                product["category"] = fields["category"].get().strip()
                product["price"] = float(fields["price"].get())
                product["cost"] = float(fields["cost"].get())
                product["quality"] = float(fields["quality"].get())
                product["brand_awareness"] = float(fields["brand_awareness"].get())
                
                if not 0 <= product["quality"] <= 1:
                    errors.append(f"Quality for {product_name} must be between 0 and 1")
                
                if not 0 <= product["brand_awareness"] <= 1:
                    errors.append(f"Brand awareness for {product_name} must be between 0 and 1")
                
                products.append(product)
            
            data["products"] = products
            
            if not products:
                errors.append("You need at least one product to start a business")
        
        except ValueError as e:
            errors.append(f"Invalid number format: {str(e)}")
        
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return None
            
        return data
    
    def start_game(self):
        """Validate and start the game with the configured business"""
        business_data = self.validate_and_gather_data()
        if business_data:
            self.app.start_business_game(business_data)

class CountryMainScreen(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.app = master
        
        # Tabs/sub-screens
        self.tabs = {}
        self.current_tab = None
        
        # Create layout
        self.create_widgets()
    
    def create_widgets(self):
        # Top info bar
        self.info_bar = ctk.CTkFrame(self, height=60)
        self.info_bar.pack(fill=tk.X, padx=10, pady=10)
        
        # Country name and flag
        self.country_label = ctk.CTkLabel(
            self.info_bar, 
            text="", 
            font=("Helvetica", 18, "bold")
        )
        self.country_label.pack(side=tk.LEFT, padx=15)
        
        # Key stats in info bar
        self.gdp_label = self.create_stat_label(self.info_bar, "GDP:", "$0T")
        self.population_label = self.create_stat_label(self.info_bar, "Population:", "0M")
        self.growth_label = self.create_stat_label(self.info_bar, "Growth:", "0%")
        self.approval_label = self.create_stat_label(self.info_bar, "Approval:", "0%")
        
        # Tabs for different sections
        self.tab_bar = ctk.CTkFrame(self)
        self.tab_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Create tab content frames
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tab buttons
        self.tabs["overview"] = self.create_tab("Overview", self.create_overview_tab)
        self.tabs["economy"] = self.create_tab("Economy", self.create_economy_tab)
        self.tabs["government"] = self.create_tab("Government", self.create_government_tab)
        self.tabs["international"] = self.create_tab("International", self.create_international_tab)
        self.tabs["development"] = self.create_tab("Development", self.create_development_tab)
        
        for tab_id, (button, content_creator) in self.tabs.items():
            if callable(content_creator):
                content_frame = content_creator()
                content_frame.pack_forget()  # Hide initially
                self.tabs[tab_id] = (button, content_frame)
            else:
                # Create an empty frame if no content creator
                empty_frame = ctk.CTkFrame(self.content_frame)
                self.tabs[tab_id] = (button, empty_frame)
        
        # Show the overview tab initially
        self.show_tab("overview")
    
    def create_stat_label(self, parent, label_text, value_text):
        """Helper to create a statistic label with value"""
        frame = ctk.CTkFrame(parent)
        frame.pack(side=tk.LEFT, padx=15)
        
        ctk.CTkLabel(
            frame,
            text=label_text,
            font=("Helvetica", 12)
        ).pack(side=tk.LEFT)
        
        value_label = ctk.CTkLabel(
            frame,
            text=value_text,
            font=("Helvetica", 12, "bold")
        )
        value_label.pack(side=tk.LEFT, padx=5)
        
        return value_label
    
    def create_tab(self, text, content_creator):
        """Create a tab button and its associated content"""
        button = ctk.CTkButton(
            self.tab_bar,
            text=text,
            height=30,
            border_width=1,
            fg_color="transparent",
            text_color=TEXT_COLOR,
            command=lambda t=text.lower(): self.show_tab(t)
        )
        button.pack(side=tk.LEFT, padx=5)
        
        return (button, content_creator)  # Content will be created when needed
    
    def show_tab(self, tab_id):
        """Switch to the specified tab"""
        # Update button styles
        for tid, (button, _) in self.tabs.items():
            if tid == tab_id:
                button.configure(fg_color=PRIMARY_COLOR, text_color="white", border_width=0)
            else:
                button.configure(fg_color="transparent", text_color=TEXT_COLOR, border_width=1)
        
        # Hide current content frame
        if self.current_tab and self.current_tab in self.tabs:
            _, content = self.tabs[self.current_tab]
            if content:
                content.pack_forget()
        
        # Show selected content frame
        _, content = self.tabs[tab_id]
        if content:
            content.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        self.current_tab = tab_id
    
    def create_overview_tab(self):
        """Create content for overview tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Split into columns
        left_col = ctk.CTkFrame(frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_col = ctk.CTkFrame(frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Key statistics in left column
        stats_frame = ctk.CTkFrame(left_col)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(
            stats_frame,
            text="Key Statistics",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Create statistics table
        self.stat_labels = {}
        self.create_stat_row(stats_frame, "GDP", "$0B")
        self.create_stat_row(stats_frame, "GDP per capita", "$0")
        self.create_stat_row(stats_frame, "GDP Growth", "0%")
        self.create_stat_row(stats_frame, "Inflation", "0%")
        self.create_stat_row(stats_frame, "Unemployment", "0%")
        self.create_stat_row(stats_frame, "Population", "0")
        self.create_stat_row(stats_frame, "Tax Revenue", "$0B")
        self.create_stat_row(stats_frame, "Budget Balance", "$0B")
        self.create_stat_row(stats_frame, "National Debt", "$0B")
        self.create_stat_row(stats_frame, "Credit Rating", "AA")
        
        # GDP chart in right column
        chart_frame = ctk.CTkFrame(right_col)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            chart_frame,
            text="GDP Growth Trend",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Create a figure for the chart
        self.gdp_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.gdp_chart = self.gdp_figure.add_subplot(111)
        
        # Add the plot to the tkinter window
        self.gdp_canvas = FigureCanvasTkAgg(self.gdp_figure, master=chart_frame)
        self.gdp_canvas.draw()
        self.gdp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Event notifications
        events_frame = ctk.CTkFrame(left_col)
        events_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(10, 5))
        
        ctk.CTkLabel(
            events_frame,
            text="Recent Events",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        self.events_text = ctk.CTkTextbox(events_frame, height=200)
        self.events_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.events_text.configure(state=tk.DISABLED)  # Read-only
        
        return frame
    
    def create_stat_row(self, parent, label, value):
        """Create a row in the statistics table"""
        row = ctk.CTkFrame(parent)
        row.pack(fill=tk.X, padx=10, pady=2)
        
        ctk.CTkLabel(row, text=label, width=120, anchor="w").pack(side=tk.LEFT)
        value_label = ctk.CTkLabel(row, text=value, anchor="e")
        value_label.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        self.stat_labels[label] = value_label
        return value_label
    
    def create_economy_tab(self):
        """Create content for economy tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Create notebook/tabs for economy sections
        tabview = ctk.CTkTabview(frame)
        tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs
        tabview.add("Overview")
        tabview.add("Taxes")
        tabview.add("Markets")
        tabview.add("Banking")
        tabview.add("Housing")
        
        # Economy overview tab content
        overview_frame = ctk.CTkFrame(tabview.tab("Overview"))
        overview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # GDP Breakdown chart
        gdp_chart_frame = ctk.CTkFrame(overview_frame)
        gdp_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            gdp_chart_frame,
            text="GDP by Sector",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Create figure for GDP chart
        self.sectors_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.sectors_chart = self.sectors_figure.add_subplot(111)
        
        # Add chart to frame
        self.sectors_canvas = FigureCanvasTkAgg(self.sectors_figure, master=gdp_chart_frame)
        self.sectors_canvas.draw()
        self.sectors_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Economic indicators
        indicators_frame = ctk.CTkFrame(overview_frame)
        indicators_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            indicators_frame,
            text="Economic Indicators",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Create economic indicators table
        self.economy_indicators = {}
        self.create_indicator_row(indicators_frame, "Interest Rate", "0%")
        self.create_indicator_row(indicators_frame, "Inflation", "0%")
        self.create_indicator_row(indicators_frame, "Unemployment", "0%")
        self.create_indicator_row(indicators_frame, "Consumer Confidence", "0")
        self.create_indicator_row(indicators_frame, "Business Confidence", "0")
        self.create_indicator_row(indicators_frame, "Stock Market Index", "0")
        
        # Interest rate slider
        rate_frame = ctk.CTkFrame(indicators_frame)
        rate_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            rate_frame,
            text="Adjust Central Bank Interest Rate:",
            font=("Helvetica", 12)
        ).pack(pady=(10, 0))
        
        self.interest_slider = ctk.CTkSlider(
            rate_frame,
            from_=0,
            to=10,
            number_of_steps=100
        )
        self.interest_slider.pack(fill=tk.X, padx=20, pady=5)
        
        slider_labels = ctk.CTkFrame(rate_frame)
        slider_labels.pack(fill=tk.X, padx=20)
        
        ctk.CTkLabel(slider_labels, text="0%").pack(side=tk.LEFT)
        self.interest_value_label = ctk.CTkLabel(slider_labels, text="3.0%")
        self.interest_value_label.pack(side=tk.TOP)
        ctk.CTkLabel(slider_labels, text="10%").pack(side=tk.RIGHT)
        
        self.interest_slider.bind("<Motion>", self.update_interest_label)
        
        ctk.CTkButton(
            rate_frame,
            text="Set Interest Rate",
            command=self.set_interest_rate
        ).pack(pady=10)
        
        # Taxes tab content
        taxes_frame = ctk.CTkFrame(tabview.tab("Taxes"))
        taxes_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tax overview
        tax_overview = ctk.CTkFrame(taxes_frame)
        tax_overview.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            tax_overview,
            text="Tax System Overview",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Tax revenue chart
        self.tax_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.tax_chart = self.tax_figure.add_subplot(111)
        
        # Add chart to frame
        self.tax_canvas = FigureCanvasTkAgg(self.tax_figure, master=tax_overview)
        self.tax_canvas.draw()
        self.tax_canvas.get_tk_widget().pack(fill=tk.X, padx=10, pady=10)
        self.tax_canvas.get_tk_widget().config(height=200)
        
        # Tax adjustment section
        tax_adjustment = ctk.CTkFrame(taxes_frame)
        tax_adjustment.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            tax_adjustment,
            text="Adjust Tax Rates",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Create scrollable tax adjustment area
        tax_scroll = ctk.CTkScrollableFrame(tax_adjustment)
        tax_scroll.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tax sliders - will be populated dynamically
        self.tax_sliders = {}
        
        # Markets tab content
        markets_frame = ctk.CTkFrame(tabview.tab("Markets"))
        markets_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Stock market section
        stock_frame = ctk.CTkFrame(markets_frame)
        stock_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            stock_frame,
            text="Stock Markets",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Stock market chart
        self.stock_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.stock_chart = self.stock_figure.add_subplot(111)
        
        # Add chart to frame
        self.stock_canvas = FigureCanvasTkAgg(self.stock_figure, master=stock_frame)
        self.stock_canvas.draw()
        self.stock_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Resource markets section
        resource_frame = ctk.CTkFrame(markets_frame)
        resource_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            resource_frame,
            text="Resource Markets",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Resource prices table
        self.resource_prices = {}
        resource_table = ctk.CTkFrame(resource_frame)
        resource_table.pack(fill=tk.X, padx=10, pady=10)
        
        # Add headers
        headers = ["Resource", "Price", "Change", "Reserves"]
        header_row = ctk.CTkFrame(resource_table)
        header_row.pack(fill=tk.X, pady=5)
        
        for i, header in enumerate(headers):
            ctk.CTkLabel(
                header_row, 
                text=header,
                font=("Helvetica", 12, "bold"),
                width=100
            ).pack(side=tk.LEFT, padx=5, expand=(i == 0))
        
        # Banking tab content
        banking_frame = ctk.CTkFrame(tabview.tab("Banking"))
        banking_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # National debt section
        debt_frame = ctk.CTkFrame(banking_frame)
        debt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            debt_frame,
            text="National Debt Management",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Debt info
        debt_info = ctk.CTkFrame(debt_frame)
        debt_info.pack(fill=tk.X, padx=10, pady=10)
        
        # Debt stats
        self.debt_stats = {}
        self.create_stat_row(debt_info, "Total Debt", "$0B")
        self.create_stat_row(debt_info, "Debt to GDP", "0%")
        self.create_stat_row(debt_info, "Interest Payments", "$0B/year")
        self.create_stat_row(debt_info, "Average Interest Rate", "0%")
        self.create_stat_row(debt_info, "Credit Rating", "AAA")
        
        # Debt issuance controls
        debt_controls = ctk.CTkFrame(debt_frame)
        debt_controls.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            debt_controls,
            text="Issue Government Bonds",
            font=("Helvetica", 12, "bold")
        ).pack(pady=5)
        
        # Amount to borrow
        amount_frame = ctk.CTkFrame(debt_controls)
        amount_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(amount_frame, text="Amount (billions):").pack(side=tk.LEFT, padx=5)
        self.debt_amount = ctk.CTkEntry(amount_frame, width=150)
        self.debt_amount.pack(side=tk.LEFT, padx=5)
        self.debt_amount.insert(0, "10")
        
        # Term length
        term_frame = ctk.CTkFrame(debt_controls)
        term_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(term_frame, text="Term (years):").pack(side=tk.LEFT, padx=5)
        self.debt_term = ctk.CTkOptionMenu(term_frame, values=["1", "2", "5", "10", "30"])
        self.debt_term.pack(side=tk.LEFT, padx=5)
        
        # Issue button
        ctk.CTkButton(
            debt_controls,
            text="Issue Bonds",
            command=self.issue_government_debt
        ).pack(pady=10)
        
        # Current bonds table
        bonds_frame = ctk.CTkFrame(debt_frame)
        bonds_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            bonds_frame,
            text="Outstanding Bonds",
            font=("Helvetica", 12, "bold")
        ).pack(pady=5)
        
        # Bonds table (placeholder)
        self.bonds_table = ctk.CTkTextbox(bonds_frame, height=150)
        self.bonds_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.bonds_table.insert(tk.END, "No bonds currently issued.")
        self.bonds_table.configure(state=tk.DISABLED)
        
        # Housing tab content
        housing_frame = ctk.CTkFrame(tabview.tab("Housing"))
        housing_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Housing market overview
        housing_overview = ctk.CTkFrame(housing_frame)
        housing_overview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            housing_overview,
            text="Housing Market",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Housing market stats
        self.housing_stats = {}
        self.create_stat_row(housing_overview, "Average House Price", "$0")
        self.create_stat_row(housing_overview, "Price to Income Ratio", "0")
        self.create_stat_row(housing_overview, "Annual Price Change", "0%")
        self.create_stat_row(housing_overview, "Houses for Sale", "0")
        self.create_stat_row(housing_overview, "Market Demand", "0")
        
        # Housing price chart
        housing_chart_frame = ctk.CTkFrame(housing_overview)
        housing_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure for housing chart
        self.housing_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.housing_chart = self.housing_figure.add_subplot(111)
        
        # Add chart to frame
        self.housing_canvas = FigureCanvasTkAgg(self.housing_figure, master=housing_chart_frame)
        self.housing_canvas.draw()
        self.housing_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Housing policy controls
        policy_frame = ctk.CTkFrame(housing_frame)
        policy_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            policy_frame,
            text="Housing Policies",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        policies = [
            "Subsidize New Construction",
            "Implement Rent Control",
            "Offer First-time Buyer Grants",
            "Restrict Foreign Buyers",
            "Increase Property Taxes"
        ]
        
        for policy in policies:
            policy_row = ctk.CTkFrame(policy_frame)
            policy_row.pack(fill=tk.X, padx=10, pady=2)
            
            ctk.CTkLabel(policy_row, text=policy).pack(side=tk.LEFT, padx=5)
            ctk.CTkButton(
                policy_row,
                text="Implement",
                width=100,
                command=lambda p=policy: self.implement_housing_policy(p)
            ).pack(side=tk.RIGHT, padx=5)
        
        return frame
    
    def create_government_tab(self):
        """Create content for government tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Create notebook/tabs for government sections
        tabview = ctk.CTkTabview(frame)
        tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs
        tabview.add("Cabinet")
        tabview.add("Budget")
        tabview.add("Departments")
        tabview.add("Policies")
        tabview.add("Elections")
        
        # Cabinet tab content
        cabinet_frame = ctk.CTkFrame(tabview.tab("Cabinet"))
        cabinet_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            cabinet_frame,
            text="Government Cabinet",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Cabinet members list
        cabinet_list = ctk.CTkScrollableFrame(cabinet_frame)
        cabinet_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        cabinet_positions = [
            "President/Prime Minister",
            "Foreign Minister",
            "Finance Minister",
            "Defense Minister",
            "Justice Minister",
            "Health Minister",
            "Education Minister",
            "Energy Minister",
            "Agriculture Minister",
            "Labor Minister"
        ]
        
        for position in cabinet_positions:
            pos_frame = ctk.CTkFrame(cabinet_list)
            pos_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkLabel(
                pos_frame,
                text=position,
                font=("Helvetica", 12, "bold")
            ).pack(side=tk.LEFT, padx=10)
            
            if position == "President/Prime Minister":
                name = "You"
            else:
                # Random names for cabinet members
                first_names = ["John", "Sarah", "Michael", "Emma", "David", "Michelle"]
                last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller"]
                name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            ctk.CTkLabel(
                pos_frame,
                text=name
            ).pack(side=tk.LEFT, padx=10)
            
            if position != "President/Prime Minister":  # Can't replace yourself
                ctk.CTkButton(
                    pos_frame,
                    text="Replace",
                    width=80,
                    command=lambda p=position: self.replace_cabinet_member(p)
                ).pack(side=tk.RIGHT, padx=10)
        
        # Budget tab content
        budget_frame = ctk.CTkFrame(tabview.tab("Budget"))
        budget_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            budget_frame,
            text="Government Budget",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Split into columns
        left_budget = ctk.CTkFrame(budget_frame)
        left_budget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_budget = ctk.CTkFrame(budget_frame)
        right_budget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Revenue pie chart
        revenue_frame = ctk.CTkFrame(left_budget)
        revenue_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            revenue_frame,
            text="Revenue Sources",
            font=("Helvetica", 12, "bold")
        ).pack(pady=5)
        
        # Create figure for revenue chart
        self.revenue_figure = plt.Figure(figsize=(4, 3), dpi=100)
        self.revenue_chart = self.revenue_figure.add_subplot(111)
        
        # Add chart to frame
        self.revenue_canvas = FigureCanvasTkAgg(self.revenue_figure, master=revenue_frame)
        self.revenue_canvas.draw()
        self.revenue_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Expenditure pie chart
        expenditure_frame = ctk.CTkFrame(right_budget)
        expenditure_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            expenditure_frame,
            text="Expenditure Allocation",
            font=("Helvetica", 12, "bold")
        ).pack(pady=5)
        
        # Create figure for expenditure chart
        self.expenditure_figure = plt.Figure(figsize=(4, 3), dpi=100)
        self.expenditure_chart = self.expenditure_figure.add_subplot(111)
        
        # Add chart to frame
        self.expenditure_canvas = FigureCanvasTkAgg(self.expenditure_figure, master=expenditure_frame)
        self.expenditure_canvas.draw()
        self.expenditure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Budget summary
        summary_frame = ctk.CTkFrame(budget_frame)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            summary_frame,
            text="Budget Summary",
            font=("Helvetica", 12, "bold")
        ).pack(pady=5)
        
        # Budget stats
        self.budget_stats = {}
        self.create_stat_row(summary_frame, "Total Revenue", "$0B")
        self.create_stat_row(summary_frame, "Total Expenditure", "$0B")
        self.create_stat_row(summary_frame, "Budget Balance", "$0B")
        self.create_stat_row(summary_frame, "Budget as % of GDP", "0%")
        
        # Departments tab content
        departments_frame = ctk.CTkFrame(tabview.tab("Departments"))
        departments_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            departments_frame,
            text="Government Departments",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Create scrollable departments list
        departments_list = ctk.CTkScrollableFrame(departments_frame)
        departments_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Will be populated dynamically
        self.department_frames = {}
        
        # Policies tab content
        policies_frame = ctk.CTkFrame(tabview.tab("Policies"))
        policies_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            policies_frame,
            text="Government Policies",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Policy categories
        policies_tabview = ctk.CTkTabview(policies_frame)
        policies_tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add policy category tabs
        policies_tabview.add("Economic")
        policies_tabview.add("Social")
        policies_tabview.add("Foreign")
        policies_tabview.add("Environmental")
        
        # Economic policies
        econ_policies = ctk.CTkScrollableFrame(policies_tabview.tab("Economic"))
        econ_policies.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        economic_policies = [
            "Minimum Wage Increase",
            "Corporate Tax Reform",
            "Infrastructure Investment",
            "Research & Development Grants",
            "Small Business Loans",
            "Trade Tariffs",
            "Universal Basic Income",
            "Banking Regulations"
        ]
        
        for policy in economic_policies:
            policy_frame = ctk.CTkFrame(econ_policies)
            policy_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkLabel(
                policy_frame,
                text=policy,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            # Status indicator - not implemented
            status = ctk.CTkLabel(
                policy_frame,
                text="Not Implemented",
                text_color="#777777"
            )
            status.pack(side=tk.LEFT, padx=10)
            
            # Implement button
            ctk.CTkButton(
                policy_frame,
                text="Implement",
                width=100,
                command=lambda p=policy: self.implement_policy(p, "Economic")
            ).pack(side=tk.RIGHT, padx=10)
        
        # Social policies (similar structure to economic)
        social_policies = ctk.CTkScrollableFrame(policies_tabview.tab("Social"))
        social_policies.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        social_policy_list = [
            "Universal Healthcare",
            "Free College Education",
            "Affordable Housing Initiative",
            "Police Reform",
            "Immigration Reform",
            "Drug Decriminalization",
            "Gun Control Measures",
            "Parental Leave Policy"
        ]
        
        for policy in social_policy_list:
            policy_frame = ctk.CTkFrame(social_policies)
            policy_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkLabel(
                policy_frame,
                text=policy,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            # Status indicator - not implemented
            status = ctk.CTkLabel(
                policy_frame,
                text="Not Implemented",
                text_color="#777777"
            )
            status.pack(side=tk.LEFT, padx=10)
            
            # Implement button
            ctk.CTkButton(
                policy_frame,
                text="Implement",
                width=100,
                command=lambda p=policy: self.implement_policy(p, "Social")
            ).pack(side=tk.RIGHT, padx=10)
        
        # Elections tab content
        elections_frame = ctk.CTkFrame(tabview.tab("Elections"))
        elections_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Election overview section
        election_overview = ctk.CTkFrame(elections_frame)
        election_overview.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            election_overview,
            text="Election Timeline",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Next election info
        next_election = ctk.CTkFrame(election_overview)
        next_election.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            next_election,
            text="Next Election:",
            font=("Helvetica", 12, "bold")
        ).pack(side=tk.LEFT, padx=10)
        
        self.next_election_label = ctk.CTkLabel(
            next_election,
            text="January 1, 2025 (365 days remaining)",
            font=("Helvetica", 12)
        )
        self.next_election_label.pack(side=tk.LEFT, padx=10)
        
        # Approval ratings
        approval_frame = ctk.CTkFrame(elections_frame)
        approval_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            approval_frame,
            text="Approval Ratings",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Create figure for approval chart
        self.approval_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.approval_chart = self.approval_figure.add_subplot(111)
        
        # Add chart to frame
        self.approval_canvas = FigureCanvasTkAgg(self.approval_figure, master=approval_frame)
        self.approval_canvas.draw()
        self.approval_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Polling data
        polling_frame = ctk.CTkFrame(elections_frame)
        polling_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            polling_frame,
            text="Current Polling",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Polling stats
        self.polling_stats = {}
        self.create_stat_row(polling_frame, "Your Party", "39%")
        self.create_stat_row(polling_frame, "Opposition Party 1", "35%")
        self.create_stat_row(polling_frame, "Opposition Party 2", "15%")
        self.create_stat_row(polling_frame, "Others", "11%")
        
        # Call election button
        ctk.CTkButton(
            elections_frame,
            text="Call Early Election",
            command=self.call_early_election,
            fg_color=WARNING_COLOR,
            hover_color="#e67e22"
        ).pack(pady=20)
        
        return frame
    
    def create_international_tab(self):
        """Create content for international tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Create notebook/tabs for international sections
        tabview = ctk.CTkTabview(frame)
        tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs
        tabview.add("Relations")
        tabview.add("Trade")
        tabview.add("Alliances")
        tabview.add("Diplomacy")
        
        # Relations tab content
        relations_frame = ctk.CTkFrame(tabview.tab("Relations"))
        relations_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            relations_frame,
            text="International Relations",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # World map visualization
        map_frame = ctk.CTkFrame(relations_frame)
        map_frame.pack(fill=tk.X, padx=10, pady=10)
        map_frame.configure(height=300)
        
        ctk.CTkLabel(
            map_frame,
            text="World Relations Map",
            font=("Helvetica", 12)
        ).pack(pady=10)
        
        # Country relations list
        relations_list_frame = ctk.CTkFrame(relations_frame)
        relations_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            relations_list_frame,
            text="Diplomatic Relations",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Scrollable list of countries and relations
        self.relations_list = ctk.CTkScrollableFrame(relations_list_frame)
        self.relations_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Will be populated dynamically with country relations
        
        # Trade tab content
        trade_frame = ctk.CTkFrame(tabview.tab("Trade"))
        trade_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            trade_frame,
            text="International Trade",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Trade overview
        trade_overview = ctk.CTkFrame(trade_frame)
        trade_overview.pack(fill=tk.X, padx=10, pady=10)
        
        # Trade balance stats
        self.trade_stats = {}
        self.create_stat_row(trade_overview, "Total Exports", "$0B")
        self.create_stat_row(trade_overview, "Total Imports", "$0B")
        self.create_stat_row(trade_overview, "Trade Balance", "$0B")
        self.create_stat_row(trade_overview, "Largest Export Partner", "N/A")
        self.create_stat_row(trade_overview, "Largest Import Partner", "N/A")
        
        # Trade chart
        trade_chart_frame = ctk.CTkFrame(trade_frame)
        trade_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure for trade chart
        self.trade_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.trade_chart = self.trade_figure.add_subplot(111)
        
        # Add chart to frame
        self.trade_canvas = FigureCanvasTkAgg(self.trade_figure, master=trade_chart_frame)
        self.trade_canvas.draw()
        self.trade_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Trade agreements
        agreements_frame = ctk.CTkFrame(trade_frame)
        agreements_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            agreements_frame,
            text="Trade Agreements",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Scrollable list of trade agreements
        self.agreements_list = ctk.CTkScrollableFrame(agreements_frame, height=150)
        self.agreements_list.pack(fill=tk.X, padx=5, pady=5)        
        # Will be populated with trade agreements
        
        # Alliances tab content
        alliances_frame = ctk.CTkFrame(tabview.tab("Alliances"))
        alliances_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            alliances_frame,
            text="Military Alliances",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Current alliances
        current_alliances = ctk.CTkFrame(alliances_frame)
        current_alliances.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            current_alliances,
            text="Current Alliances",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Alliance list
        self.alliances_list = ctk.CTkScrollableFrame(current_alliances)
        self.alliances_list.pack(fill=tk.X, padx=5, pady=5)
        self.alliances_list.configure(height=150)

        
        # Military comparison
        military_compare = ctk.CTkFrame(alliances_frame)
        military_compare.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            military_compare,
            text="Military Strength Comparison",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Create figure for military comparison
        self.military_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.military_chart = self.military_figure.add_subplot(111)
        
        # Add chart to frame
        self.military_canvas = FigureCanvasTkAgg(self.military_figure, master=military_compare)
        self.military_canvas.draw()
        self.military_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Diplomacy tab content
        diplomacy_frame = ctk.CTkFrame(tabview.tab("Diplomacy"))
        diplomacy_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            diplomacy_frame,
            text="Diplomatic Actions",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Country selector
        selection_frame = ctk.CTkFrame(diplomacy_frame)
        selection_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            selection_frame,
            text="Select Country:",
            font=("Helvetica", 12)
        ).pack(side=tk.LEFT, padx=10)
        
        self.diplomacy_country = ctk.CTkOptionMenu(
            selection_frame,
            values=["United States", "China", "Russia", "Germany", "United Kingdom"]
        )
        self.diplomacy_country.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Action selector
        action_frame = ctk.CTkFrame(diplomacy_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            action_frame,
            text="Diplomatic Action:",
            font=("Helvetica", 12)
        ).pack(side=tk.LEFT, padx=10)
        
        self.diplomacy_action = ctk.CTkOptionMenu(
            action_frame,
            values=[
                "Improve Relations", 
                "Establish Trade Agreement", 
                "Form Alliance",
                "Impose Sanctions",
                "Declare War",
                "Send Aid",
                "Cultural Exchange"
            ]
        )
        self.diplomacy_action.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Execute action button
        ctk.CTkButton(
            diplomacy_frame,
            text="Execute Diplomatic Action",
            command=self.execute_diplomatic_action
        ).pack(pady=20)
        
        # Recent actions log
        log_frame = ctk.CTkFrame(diplomacy_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            log_frame,
            text="Diplomatic Actions Log",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        self.diplomacy_log = ctk.CTkTextbox(log_frame)
        self.diplomacy_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.diplomacy_log.insert(tk.END, "No diplomatic actions recorded yet.")
        self.diplomacy_log.configure(state=tk.DISABLED)
        
        return frame
    
    def create_development_tab(self):
        """Create content for development tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Create notebook/tabs for development sections
        tabview = ctk.CTkTabview(frame)
        tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs
        tabview.add("Infrastructure")
        tabview.add("Technology")
        tabview.add("Projects")
        tabview.add("Resources")
        
        # Infrastructure tab content
        infra_frame = ctk.CTkFrame(tabview.tab("Infrastructure"))
        infra_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            infra_frame,
            text="Infrastructure Development",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Infrastructure overview
        infra_overview = ctk.CTkFrame(infra_frame)
        infra_overview.pack(fill=tk.X, padx=10, pady=10)
        
        # Infrastructure stats
        self.infra_stats = {}
        self.create_stat_row(infra_overview, "Overall Infrastructure", "0/10")
        self.create_stat_row(infra_overview, "Transportation", "0/10")
        self.create_stat_row(infra_overview, "Energy", "0/10")
        self.create_stat_row(infra_overview, "Communications", "0/10")
        self.create_stat_row(infra_overview, "Water & Sanitation", "0/10")
        
        # Infrastructure projects
        projects_frame = ctk.CTkFrame(infra_frame)
        projects_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            projects_frame,
            text="Available Projects",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Scrollable list of infrastructure projects
        self.infra_projects = ctk.CTkScrollableFrame(projects_frame)
        self.infra_projects.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sample infrastructure projects
        infra_project_types = [
            "Highway Network Expansion",
            "High-Speed Rail",
            "Airport Modernization",
            "Renewable Energy Plants",
            "Nuclear Power Plant",
            "5G Network Deployment",
            "Water Treatment Facilities",
            "Smart Grid Implementation"
        ]
        
        for project in infra_project_types:
            project_frame = ctk.CTkFrame(self.infra_projects)
            project_frame.pack(fill=tk.X, padx=5, pady=5)
            
            cost = random.randint(5, 50)  # Random cost in billions
            time = random.randint(1, 5)  # Random time in years
            
            ctk.CTkLabel(
                project_frame,
                text=project,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                project_frame,
                text=f"Cost: ${cost}B | Time: {time} years"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                project_frame,
                text="Initiate",
                width=80,
                command=lambda p=project, c=cost, t=time: self.start_infra_project(p, c, t)
            ).pack(side=tk.RIGHT, padx=10)
        
        # Technology tab content
        tech_frame = ctk.CTkFrame(tabview.tab("Technology"))
        tech_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            tech_frame,
            text="Technology Research",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Research statistics
        research_stats = ctk.CTkFrame(tech_frame)
        research_stats.pack(fill=tk.X, padx=10, pady=10)
        
        # Research stats
        self.research_stats = {}
        self.create_stat_row(research_stats, "R&D Spending", "$0B (0% of GDP)")
        self.create_stat_row(research_stats, "Technology Level", "0/10")
        self.create_stat_row(research_stats, "Research Institutions", "0")
        self.create_stat_row(research_stats, "Patents Filed", "0")
        
        # Technology tree visualization (simplified)
        tech_tree_frame = ctk.CTkFrame(tech_frame)
        tech_tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            tech_tree_frame,
            text="Technology Research Areas",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Research areas
        research_areas = [
            "Artificial Intelligence",
            "Biotechnology",
            "Renewable Energy",
            "Quantum Computing",
            "Nanotechnology",
            "Space Technology",
            "Advanced Materials",
            "Robotics"
        ]
        
        for area in research_areas:
            area_frame = ctk.CTkFrame(tech_tree_frame)
            area_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Random progress
            progress = random.randint(0, 100)
            
            ctk.CTkLabel(
                area_frame,
                text=area,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkProgressBar(
                area_frame,
                width=200
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                area_frame,
                text="Fund Research",
                width=100,
                command=lambda a=area: self.fund_research(a)
            ).pack(side=tk.RIGHT, padx=10)
        
        # Projects tab content
        projects_frame = ctk.CTkFrame(tabview.tab("Projects"))
        projects_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            projects_frame,
            text="National Projects",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Active projects
        active_projects = ctk.CTkFrame(projects_frame)
        active_projects.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            active_projects,
            text="Active Projects",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Scrollable list of active projects (empty for now)
        self.active_projects_list = ctk.CTkScrollableFrame(active_projects)
        self.active_projects_list.configure(height=150)
        self.active_projects_list.pack(fill=tk.BOTH, padx=5, pady=5)
        
        ctk.CTkLabel(
            self.active_projects_list,
            text="No active projects"
        ).pack(pady=10)
        
        # Available national projects
        available_projects = ctk.CTkFrame(projects_frame)
        available_projects.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            available_projects,
            text="Available National Projects",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Scrollable list of available projects
        self.available_projects_list = ctk.CTkScrollableFrame(available_projects)
        self.available_projects_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sample national projects
        national_projects = [
            "International Space Station Participation",
            "National Genomics Database",
            "Universal Internet Access",
            "Autonomous Vehicle Infrastructure",
            "Green Energy Transition",
            "National AI Research Institute",
            "Urban Renewal Program",
            "Carbon Capture Initiative"
        ]
        
        for project in national_projects:
            project_frame = ctk.CTkFrame(self.available_projects_list)
            project_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Random cost and benefits
            cost = random.randint(10, 100)  # Cost in billions
            years = random.randint(3, 10)  # Years to complete
            
            ctk.CTkLabel(
                project_frame,
                text=project,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                project_frame,
                text=f"Cost: ${cost}B | Time: {years} years"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                project_frame,
                text="Start Project",
                width=100,
                command=lambda p=project, c=cost, y=years: self.start_national_project(p, c, y)
            ).pack(side=tk.RIGHT, padx=10)
        
        # Resources tab content
        resources_frame = ctk.CTkFrame(tabview.tab("Resources"))
        resources_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            resources_frame,
            text="Natural Resources",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Resources overview
        resources_overview = ctk.CTkFrame(resources_frame)
        resources_overview.pack(fill=tk.X, padx=10, pady=10)
        
        # Create figure for resources chart
        self.resources_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.resources_chart = self.resources_figure.add_subplot(111)
        
        # Add chart to frame
        self.resources_canvas = FigureCanvasTkAgg(self.resources_figure, master=resources_overview)
        self.resources_canvas.draw()
        self.resources_canvas.get_tk_widget().pack(fill=tk.BOTH, padx=10, pady=10)
        self.resources_canvas.get_tk_widget().configure(height=250)        
 
        # Resources management
        resources_management = ctk.CTkFrame(resources_frame)
        resources_management.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            resources_management,
            text="Resource Management",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Resource list
        resources_list = ctk.CTkScrollableFrame(resources_management)
        resources_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Resource types
        resource_types = [
            "Oil", "Natural Gas", "Coal", "Iron", "Copper",
            "Gold", "Uranium", "Farmland", "Forests", "Water"
        ]
        
        self.resource_labels = {}
        
        for resource in resource_types:
            resource_frame = ctk.CTkFrame(resources_list)
            resource_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkLabel(
                resource_frame,
                text=resource,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            # Resource amount (will be updated dynamically)
            amount_label = ctk.CTkLabel(
                resource_frame,
                text="0 units"
            )
            amount_label.pack(side=tk.LEFT, padx=10)
            self.resource_labels[resource] = amount_label
            
            # Resource actions
            actions_frame = ctk.CTkFrame(resource_frame)
            actions_frame.pack(side=tk.RIGHT)
            
            ctk.CTkButton(
                actions_frame,
                text="Extract",
                width=80,
                command=lambda r=resource: self.extract_resource(r)
            ).pack(side=tk.LEFT, padx=5)
            
            ctk.CTkButton(
                actions_frame,
                text="Trade",
                width=80,
                command=lambda r=resource: self.trade_resource(r)
            ).pack(side=tk.LEFT, padx=5)
        
        return frame
    
    def create_indicator_row(self, parent, label, value):
        """Create a row in the economic indicators table"""
        row = ctk.CTkFrame(parent)
        row.pack(fill=tk.X, padx=10, pady=2)
        
        ctk.CTkLabel(row, text=label, width=150, anchor="w").pack(side=tk.LEFT)
        value_label = ctk.CTkLabel(row, text=value, anchor="e")
        value_label.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        self.economy_indicators[label] = value_label
        return value_label
    
    def update_display(self):
        """Update the UI with current data"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
        
        # Update country name and info bar stats
        self.country_label.configure(text=country.name)
        
        # Format GDP (trillions, billions, or millions depending on size)
        if country.gdp >= 1e12:
            gdp_display = f"${country.gdp / 1e12:.2f}T"
        elif country.gdp >= 1e9:
            gdp_display = f"${country.gdp / 1e9:.2f}B"
        else:
            gdp_display = f"${country.gdp / 1e6:.2f}M"
        
        self.gdp_label.configure(text=gdp_display)
        
        # Format population (billions, millions, or thousands)
        if country.population >= 1e9:
            pop_display = f"{country.population / 1e9:.2f}B"
        elif country.population >= 1e6:
            pop_display = f"{country.population / 1e6:.2f}M"
        else:
            pop_display = f"{country.population / 1e3:.2f}K"
        
        self.population_label.configure(text=pop_display)
        
        # GDP growth rate
        growth_display = f"{country.gdp_growth * 100:.1f}%"
        self.growth_label.configure(text=growth_display)
        
        # Approval rating (simplified)
        approval = country.happiness * 100
        approval_display = f"{approval:.1f}%"
        self.approval_label.configure(text=approval_display)
        
        # If overview tab is active, update detailed stats
        if self.current_tab == "overview" and hasattr(self, "stat_labels"):
            self.stat_labels["GDP"].configure(text=gdp_display)
            
            # GDP per capita
            gdp_per_capita = country.gdp / country.population
            gdp_per_capita_display = f"${gdp_per_capita:.2f}"
            self.stat_labels["GDP per capita"].configure(text=gdp_per_capita_display)
            
            self.stat_labels["GDP Growth"].configure(text=growth_display)
            self.stat_labels["Inflation"].configure(text=f"{country.inflation_rate * 100:.1f}%")
            self.stat_labels["Unemployment"].configure(text=f"{country.unemployment_rate * 100:.1f}%")
            self.stat_labels["Population"].configure(text=f"{country.population:,}")
            
            # Calculate tax revenue
            tax_revenue = country.get_tax_revenue()
            if tax_revenue >= 1e12:
                tax_display = f"${tax_revenue / 1e12:.2f}T/yr"
            elif tax_revenue >= 1e9:
                tax_display = f"${tax_revenue / 1e9:.2f}B/yr"
            else:
                tax_display = f"${tax_revenue / 1e6:.2f}M/yr"
            
            self.stat_labels["Tax Revenue"].configure(text=tax_display)
            
            # Budget balance
            if country.budget_balance >= 1e12:
                budget_display = f"${country.budget_balance / 1e12:.2f}T/yr"
            elif country.budget_balance >= 1e9:
                budget_display = f"${country.budget_balance / 1e9:.2f}B/yr"
            elif country.budget_balance >= 0:
                budget_display = f"${country.budget_balance / 1e6:.2f}M/yr"
            elif country.budget_balance > -1e9:
                budget_display = f"-${abs(country.budget_balance) / 1e6:.2f}M/yr"
            elif country.budget_balance > -1e12:
                budget_display = f"-${abs(country.budget_balance) / 1e9:.2f}B/yr"
            else:
                budget_display = f"-${abs(country.budget_balance) / 1e12:.2f}T/yr"
            
            self.stat_labels["Budget Balance"].configure(text=budget_display)
            
            # National debt
            if country.national_debt >= 1e12:
                debt_display = f"${country.national_debt / 1e12:.2f}T"
            elif country.national_debt >= 1e9:
                debt_display = f"${country.national_debt / 1e9:.2f}B"
            else:
                debt_display = f"${country.national_debt / 1e6:.2f}M"
            
            self.stat_labels["National Debt"].configure(text=debt_display)
            
            # Credit rating
            self.stat_labels["Credit Rating"].configure(text=country.credit_rating)
            
            # Update GDP chart
            self.update_gdp_chart(country)
            
        # Update economy indicators if on that tab
        if self.current_tab == "economy" and hasattr(self, "economy_indicators"):
            if "Interest Rate" in self.economy_indicators:
                self.economy_indicators["Interest Rate"].configure(text=f"{country.interest_rate * 100:.1f}%")
            
            if "Inflation" in self.economy_indicators:
                self.economy_indicators["Inflation"].configure(text=f"{country.inflation_rate * 100:.1f}%")
            
            if "Unemployment" in self.economy_indicators:
                self.economy_indicators["Unemployment"].configure(text=f"{country.unemployment_rate * 100:.1f}%")
            
            # Update other indicators with simulated values
            consumer_confidence = 50 + (country.gdp_growth * 100) - (country.inflation_rate * 100)
            business_confidence = 50 + (country.gdp_growth * 200) - (country.unemployment_rate * 100)
            
            if "Consumer Confidence" in self.economy_indicators:
                self.economy_indicators["Consumer Confidence"].configure(text=f"{consumer_confidence:.1f}")
            
            if "Business Confidence" in self.economy_indicators:
                self.economy_indicators["Business Confidence"].configure(text=f"{business_confidence:.1f}")
            
            # Update sectors chart
            self.update_sectors_chart(country)
            
            # Update interest rate slider value
            if hasattr(self, "interest_slider"):
                self.interest_slider.set(country.interest_rate * 100)
                self.interest_value_label.configure(text=f"{country.interest_rate * 100:.1f}%")
            
            # Update tax sliders
            self.update_tax_sliders(country)
            
            # Update housing market stats
            if hasattr(self, "housing_stats"):
                housing = country.housing_market
                
                if "Average House Price" in self.housing_stats:
                    self.housing_stats["Average House Price"].configure(text=f"${housing.avg_price:,.0f}")
                
                if "Price to Income Ratio" in self.housing_stats:
                    ratio = housing.avg_price / country.average_income
                    self.housing_stats["Price to Income Ratio"].configure(text=f"{ratio:.1f}x")
                
                # Calculate annual price change from history
                if len(housing.price_history) >= 2:
                    dates = sorted(housing.price_history.keys())
                    if len(dates) >= 2:
                        newest_price = housing.price_history[dates[-1]]
                        year_ago_price = housing.price_history[dates[0]]  # Not exactly a year ago, but earliest record
                        annual_change = ((newest_price / year_ago_price) - 1) * 100
                        
                        if "Annual Price Change" in self.housing_stats:
                            self.housing_stats["Annual Price Change"].configure(text=f"{annual_change:.1f}%")
                
                if "Houses for Sale" in self.housing_stats:
                    self.housing_stats["Houses for Sale"].configure(text=f"{housing.inventory:,}")
                
                if "Market Demand" in self.housing_stats:
                    self.housing_stats["Market Demand"].configure(text=f"{housing.demand:.1f}")
                
                # Update housing price chart
                self.update_housing_chart(housing)
                
        # Update government tab data if selected
        if self.current_tab == "government" and hasattr(self, "department_frames"):
            # Update department budgets
            self.update_department_frames(country)
            
            # Update budget charts
            self.update_budget_charts(country)
            
            # Update election timeline
            self.update_election_timeline()
        
        # Update international relations if that tab is selected
        if self.current_tab == "international" and hasattr(self, "relations_list"):
            self.update_relations_list(country)
        
        # Update development tab
        if self.current_tab == "development" and hasattr(self, "resource_labels"):
            self.update_resource_info(country)
    
    def update_gdp_chart(self, country):
        """Update the GDP growth chart with historical data"""
        if not hasattr(self, "gdp_chart"):
            return
            
        # Clear previous plot
        self.gdp_chart.clear()
        
        # Create some simulated GDP history if country history is empty
        if not country.history:
            # Generate some historical data
            base_gdp = country.gdp * 0.9  # Start at 90% of current GDP
            dates = []
            gdp_values = []
            
            # Generate data for past 12 months
            current_date = self.app.game_state.current_date
            for i in range(12, 0, -1):
                past_date = current_date - datetime.timedelta(days=i*30)
                date_str = past_date.strftime("%Y-%m-%d")
                dates.append(date_str)
                
                # Add some randomization to historical growth
                growth_factor = 1 + (country.gdp_growth / 12) + (random.random() - 0.5) * 0.01
                base_gdp *= growth_factor
                gdp_values.append(base_gdp)
            
            # Add current GDP
            dates.append(current_date.strftime("%Y-%m-%d"))
            gdp_values.append(country.gdp)
        else:
            # Use actual historical data
            dates = sorted(country.history.keys())
            gdp_values = [country.history[date].get("gdp", 0) for date in dates]
            
            # If too many data points, just use the last 30
            if len(dates) > 30:
                dates = dates[-30:]
                gdp_values = gdp_values[-30:]
        
        # Convert dates to datetime objects for plotting
        date_objects = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
        
        # Plot the data
        self.gdp_chart.plot(date_objects, gdp_values, marker='o', linestyle='-', color=PRIMARY_COLOR)
        
        # Format the plot
        self.gdp_chart.set_title("GDP Growth")
        self.gdp_chart.set_xlabel("Date")
        self.gdp_chart.set_ylabel("GDP")
        
        # Format y-axis with trillion/billion labels
        max_gdp = max(gdp_values)
        if max_gdp >= 1e12:
            self.gdp_chart.set_ylabel("GDP (Trillions $)")
            self.gdp_chart.yaxis.set_major_formatter(lambda x, pos: f"${x/1e12:.1f}T")
        elif max_gdp >= 1e9:
            self.gdp_chart.set_ylabel("GDP (Billions $)")
            self.gdp_chart.yaxis.set_major_formatter(lambda x, pos: f"${x/1e9:.1f}B")
        else:
            self.gdp_chart.set_ylabel("GDP (Millions $)")
            self.gdp_chart.yaxis.set_major_formatter(lambda x, pos: f"${x/1e6:.1f}M")
        
        # Format x-axis dates
        self.gdp_chart.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        self.gdp_chart.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(self.gdp_chart.xaxis.get_majorticklabels(), rotation=45)
        
        # Add grid
        self.gdp_chart.grid(True, linestyle='--', alpha=0.7)
        
        # Update the canvas
        self.gdp_figure.tight_layout()
        self.gdp_canvas.draw()
    
    def update_sectors_chart(self, country):
        """Update the GDP by sector pie chart"""
        if not hasattr(self, "sectors_chart"):
            return
            
        # Clear previous plot
        self.sectors_chart.clear()
        
        # Get sector data
        sectors = country.sectors
        
        # Format data for pie chart
        labels = list(sectors.keys())
        sizes = [sectors[sector] * country.gdp for sector in labels]
        
        # Create pie chart
        self.sectors_chart.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%', 
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        self.sectors_chart.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        self.sectors_chart.set_title("GDP by Sector")
        
        # Update the canvas
        self.sectors_figure.tight_layout()
        self.sectors_canvas.draw()
    
    def update_interest_label(self, event=None):
        """Update the interest rate label when slider moves"""
        if hasattr(self, "interest_slider") and hasattr(self, "interest_value_label"):
            value = self.interest_slider.get()
            self.interest_value_label.configure(text=f"{value:.1f}%")
    
    def set_interest_rate(self):
        """Set the central bank interest rate"""
        if not hasattr(self, "interest_slider"):
            return
            
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        # Get value from slider
        value = self.interest_slider.get()
        
        # Convert to decimal (0.03 = 3%)
        rate = value / 100
        
        # Update country's interest rate
        country.adjust_interest_rate(rate)
        
        # Show feedback message
        messagebox.showinfo("Interest Rate", f"Central bank interest rate set to {value:.1f}%")
    
    def update_tax_sliders(self, country):
        """Update the tax sliders with current tax rates"""
        if not hasattr(self, "tax_sliders"):
            self.tax_sliders = {}
            
        # Check if we need to create the sliders
        if not self.tax_sliders and hasattr(self, "taxes_frame"):
            # Find the tax adjustment scroll frame
            for child in self.winfo_children():
                if isinstance(child, ctk.CTkFrame) and child.winfo_name() == "!ctktabview":
                    for tab_child in child.winfo_children():
                        if isinstance(tab_child, ctk.CTkFrame) and "frame" in tab_child.winfo_name():
                            for frame in tab_child.winfo_children():
                                if isinstance(frame, ctk.CTkScrollableFrame):
                                    tax_scroll = frame
                                    break
            
            # Create sliders for each tax
            for tax_name, tax in country.taxes.items():
                tax_frame = ctk.CTkFrame(tax_scroll)
                tax_frame.pack(fill=tk.X, padx=5, pady=5)
                
                # Tax name and current rate
                name_frame = ctk.CTkFrame(tax_frame)
                name_frame.pack(fill=tk.X, padx=5, pady=2)
                
                ctk.CTkLabel(
                    name_frame,
                    text=tax.name,
                    font=("Helvetica", 12, "bold")
                ).pack(side=tk.LEFT, padx=5)
                
                rate_label = ctk.CTkLabel(
                    name_frame,
                    text=f"Current rate: {tax.rate * 100:.1f}%"
                )
                rate_label.pack(side=tk.RIGHT, padx=5)
                
                # Slider
                slider_frame = ctk.CTkFrame(tax_frame)
                slider_frame.pack(fill=tk.X, padx=5, pady=2)
                
                slider = ctk.CTkSlider(
                    slider_frame,
                    from_=0,
                    to=50,  # Max 50% tax
                    number_of_steps=100,
                )
                slider.pack(fill=tk.X, padx=10, pady=5)
                slider.set(tax.rate * 100)  # Set to current percentage
                
                # Value label
                value_frame = ctk.CTkFrame(slider_frame)
                value_frame.pack(fill=tk.X)
                
                ctk.CTkLabel(value_frame, text="0%").pack(side=tk.LEFT, padx=10)
                value_label = ctk.CTkLabel(value_frame, text=f"{tax.rate * 100:.1f}%")
                value_label.pack(side=tk.TOP)
                ctk.CTkLabel(value_frame, text="50%").pack(side=tk.RIGHT, padx=10)
                
                # Apply button
                apply_btn = ctk.CTkButton(
                    tax_frame,
                    text="Apply New Rate",
                    command=lambda tn=tax_name, s=slider, rl=rate_label, vl=value_label: 
                        self.apply_tax_rate(tn, s, rl, vl)
                )
                apply_btn.pack(pady=5)
                
                # Update label on slider change
                slider.configure(command=lambda value, vl=value_label: vl.configure(text=f"{value:.1f}%"))
                
                # Store references
                self.tax_sliders[tax_name] = {
                    "slider": slider,
                    "rate_label": rate_label,
                    "value_label": value_label
                }
                
                # If tax has brackets, add a button to edit them
                if tax.bracketed and tax.brackets:
                    ctk.CTkButton(
                        tax_frame,
                        text="Edit Tax Brackets",
                        command=lambda tn=tax_name: self.edit_tax_brackets(tn)
                    ).pack(pady=5)
        else:
            # Update existing sliders
            for tax_name, tax in country.taxes.items():
                if tax_name in self.tax_sliders:
                    controls = self.tax_sliders[tax_name]
                    controls["rate_label"].configure(text=f"Current rate: {tax.rate * 100:.1f}%")
                    controls["slider"].set(tax.rate * 100)
                    controls["value_label"].configure(text=f"{tax.rate * 100:.1f}%")
    
    def apply_tax_rate(self, tax_name, slider, rate_label, value_label):
        """Apply the new tax rate from the slider"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        # Get new rate from slider (convert from percentage to decimal)
        new_rate = slider.get() / 100
        
        # Update the tax rate
        if country.change_tax(tax_name, new_rate):
            # Update labels
            rate_label.configure(text=f"Current rate: {new_rate * 100:.1f}%")
            value_label.configure(text=f"{new_rate * 100:.1f}%")
            
            messagebox.showinfo("Tax Rate", f"{tax_name} rate updated to {new_rate * 100:.1f}%")
        else:
            messagebox.showerror("Error", f"Failed to update {tax_name} rate")
    
    def edit_tax_brackets(self, tax_name):
        """Open a dialog to edit tax brackets"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country) or tax_name not in country.taxes:
            return
            
        tax = country.taxes[tax_name]
        if not tax.bracketed or not tax.brackets:
            return
            
        # Create a dialog window
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Edit {tax_name} Brackets")
        dialog.geometry("500x400")
        dialog.transient(self)  # Make it a modal dialog
        dialog.grab_set()
        
        # Create scrollable frame for brackets
        scroll_frame = ctk.CTkScrollableFrame(dialog)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ctk.CTkFrame(scroll_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(header_frame, text="Income Threshold ($)", width=150).pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(header_frame, text="Tax Rate (%)", width=100).pack(side=tk.LEFT, padx=5)
        
        # Create entries for each bracket
        bracket_entries = []
        for i, (threshold, rate) in enumerate(sorted(tax.brackets, key=lambda x: x[0])):
            bracket_frame = ctk.CTkFrame(scroll_frame)
            bracket_frame.pack(fill=tk.X, padx=5, pady=2)
            
            threshold_entry = ctk.CTkEntry(bracket_frame, width=150)
            threshold_entry.pack(side=tk.LEFT, padx=5)
            threshold_entry.insert(0, str(int(threshold)))
            
            rate_entry = ctk.CTkEntry(bracket_frame, width=100)
            rate_entry.pack(side=tk.LEFT, padx=5)
            rate_entry.insert(0, str(rate * 100))
            
            # Delete button (except for first bracket)
            if i > 0:
                ctk.CTkButton(
                    bracket_frame,
                    text="Delete",
                    width=70,
                    fg_color=DANGER_COLOR,
                    command=lambda frame=bracket_frame, entries=bracket_entries:
                        self.delete_bracket_entry(frame, entries)
                ).pack(side=tk.RIGHT, padx=5)
            
            bracket_entries.append((threshold_entry, rate_entry))
        
        # Add bracket button
        ctk.CTkButton(
            scroll_frame,
            text="+ Add Bracket",
            command=lambda: self.add_bracket_entry(scroll_frame, bracket_entries)
        ).pack(pady=10)
        
        # Save button
        ctk.CTkButton(
            dialog,
            text="Save Changes",
            command=lambda: self.save_tax_brackets(dialog, tax_name, bracket_entries)
        ).pack(pady=10)
    
    def add_bracket_entry(self, parent, entries_list):
        """Add a new bracket entry row"""
        # Add before the Add button
        bracket_frame = ctk.CTkFrame(parent)
        bracket_frame.pack(fill=tk.X, padx=5, pady=2, before=parent.winfo_children()[-1])
        
        threshold_entry = ctk.CTkEntry(bracket_frame, width=150)
        threshold_entry.pack(side=tk.LEFT, padx=5)
        threshold_entry.insert(0, "0")
        
        rate_entry = ctk.CTkEntry(bracket_frame, width=100)
        rate_entry.pack(side=tk.LEFT, padx=5)
        rate_entry.insert(0, "0")
        
        # Delete button
        ctk.CTkButton(
            bracket_frame,
            text="Delete",
            width=70,
            fg_color=DANGER_COLOR,
            command=lambda frame=bracket_frame, entries=entries_list:
                self.delete_bracket_entry(frame, entries)
        ).pack(side=tk.RIGHT, padx=5)
        
        entries_list.append((threshold_entry, rate_entry))
    
    def delete_bracket_entry(self, frame, entries_list):
        """Delete a bracket entry row"""
        # Find the entry in the list
        for i, (threshold_entry, rate_entry) in enumerate(entries_list):
            if threshold_entry.winfo_parent() == frame.winfo_pathname(frame.winfo_id()):
                entries_list.pop(i)
                frame.destroy()
                return
    
    def save_tax_brackets(self, dialog, tax_name, bracket_entries):
        """Save the edited tax brackets"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country) or tax_name not in country.taxes:
            dialog.destroy()
            return
            
        # Extract values from entries
        try:
            new_brackets = []
            for threshold_entry, rate_entry in bracket_entries:
                threshold = float(threshold_entry.get())
                rate = float(rate_entry.get()) / 100  # Convert percentage to decimal
                
                if rate < 0 or rate > 1:
                    raise ValueError("Tax rate must be between 0% and 100%")
                    
                new_brackets.append((threshold, rate))
                
            # Sort by threshold
            new_brackets.sort(key=lambda x: x[0])
            
            # Update the tax brackets
            country.taxes[tax_name].brackets = new_brackets
            
            messagebox.showinfo("Tax Brackets", f"{tax_name} brackets updated successfully")
            dialog.destroy()
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def update_housing_chart(self, housing_market):
        """Update the housing price history chart"""
        if not hasattr(self, "housing_chart"):
            return
            
        # Clear previous plot
        self.housing_chart.clear()
        
        # Check if we have price history data
        if housing_market and housing_market.price_history:
            # Get data
            dates = sorted(housing_market.price_history.keys())
            prices = [housing_market.price_history[date] for date in dates]
            
            # If too many data points, just use the last 30
            if len(dates) > 30:
                dates = dates[-30:]
                prices = prices[-30:]
            
            # Convert dates to datetime objects for plotting
            try:
                date_objects = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
                
                # Plot the data
                self.housing_chart.plot(date_objects, prices, marker='o', linestyle='-', color=PRIMARY_COLOR)
                
                # Format the plot
                self.housing_chart.set_title("Housing Price Trend")
                self.housing_chart.set_xlabel("Date")
                self.housing_chart.set_ylabel("Average Price ($)")
                
                # Format y-axis with dollar amounts
                self.housing_chart.yaxis.set_major_formatter(lambda x, pos: f"${x:,.0f}")
                
                # Format x-axis dates
                self.housing_chart.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                self.housing_chart.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                plt.setp(self.housing_chart.xaxis.get_majorticklabels(), rotation=45)
                
                # Add grid
                self.housing_chart.grid(True, linestyle='--', alpha=0.7)
                
                # Update the canvas
                self.housing_figure.tight_layout()
                self.housing_canvas.draw()
            except Exception as e:
                print(f"Error plotting housing chart: {e}")
        else:
            # No data yet, show a message
            self.housing_chart.text(0.5, 0.5, "No housing price data available yet", 
                                    horizontalalignment='center', verticalalignment='center')
            self.housing_canvas.draw()
    
    def implement_housing_policy(self, policy):
        """Implement a housing policy"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        # Effects of different policies
        effects = {
            "Subsidize New Construction": {
                "inventory": lambda h: h.inventory * 1.2,  # Increase supply
                "message": "Government subsidies will increase housing construction by 20%"
            },
            "Implement Rent Control": {
                "demand": lambda h: h.demand * 0.9,  # Decrease price pressure
                "message": "Rent controls will stabilize housing costs but may reduce construction"
            },
            "Offer First-time Buyer Grants": {
                "demand": lambda h: h.demand * 1.2,  # Increase demand
                "message": "First-time buyer grants will help citizens but may increase prices due to demand"
            },
            "Restrict Foreign Buyers": {
                "demand": lambda h: h.demand * 0.85,  # Reduce demand
                "message": "Foreign buyer restrictions will reduce pressure on the housing market"
            },
            "Increase Property Taxes": {
                "demand": lambda h: h.demand * 0.8,  # Reduce demand
                "message": "Higher property taxes will cool the market but may be unpopular"
            }
        }
        
        # Apply policy effect
        if policy in effects:
            effect = effects[policy]
            housing = country.housing_market
            
            # Apply each effect
            for key, func in effect.items():
                if key != "message" and hasattr(housing, key):
                    setattr(housing, key, func(housing))
            
            # Show message
            messagebox.showinfo("Housing Policy", effect["message"])
            
            # Update display
            self.update_display()
        else:
            messagebox.showerror("Error", "Unknown policy")
    
    def issue_government_debt(self):
        """Issue new government bonds/debt"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        try:
            # Get amount and term from inputs
            amount_billions = float(self.debt_amount.get())
            term_years = int(self.debt_term.get())
            
            # Convert to actual value
            amount = amount_billions * 1000000000  # Convert to actual value
            
            # Issue debt
            success = country.issue_debt(amount)
            
            if success:
                # Add a bond to the list
                if hasattr(self, "bonds_table"):
                    # Clear existing content if it's the placeholder text
                    current_text = self.bonds_table.get("1.0", tk.END).strip()
                    if current_text == "No bonds currently issued.":
                        self.bonds_table.configure(state=tk.NORMAL)
                        self.bonds_table.delete("1.0", tk.END)
                        self.bonds_table.configure(state=tk.DISABLED)
                    
                    # Calculate interest rate based on country's current rate and credit rating
                    debt_to_gdp = country.national_debt / country.gdp
                    risk_premium = debt_to_gdp * 0.1
                    
                    # Credit rating affects interest rate
                    rating_to_premium = {
                        "AAA": 0.0, "AA": 0.005, "A": 0.01, "BBB": 0.02,
                        "BB": 0.04, "B": 0.07, "CCC": 0.12, "CC": 0.18, "C": 0.25, "D": 0.4
                    }
                    
                    credit_premium = rating_to_premium.get(country.credit_rating, 0.05)
                    
                    # Calculate effective interest rate
                    effective_rate = country.interest_rate + risk_premium + credit_premium
                    
                    # Format the note for the bonds table
                    issue_date = self.app.game_state.current_date.strftime("%Y-%m-%d")
                    maturity_date = (self.app.game_state.current_date + 
                                    datetime.timedelta(days=term_years * 365)).strftime("%Y-%m-%d")
                    annual_interest = amount * effective_rate
                    bond_note = (f"${amount_billions}B bond issued on {issue_date}\n"
                                f"Term: {term_years} years, Matures: {maturity_date}\n"
                                f"Interest Rate: {effective_rate*100:.2f}%, "
                                f"Annual Interest: ${annual_interest/1e9:.2f}B\n"
                                f"----------------------------------------------------\n")
                    
                    self.bonds_table.configure(state=tk.NORMAL)
                    self.bonds_table.insert(tk.END, bond_note)
                    self.bonds_table.configure(state=tk.DISABLED)
                
                messagebox.showinfo("Debt Issuance", 
                                   f"Successfully issued ${amount_billions}B in government bonds "
                                   f"with {term_years}-year term")
                
                # Update display
                self.update_display()
            else:
                messagebox.showerror("Error", "Failed to issue debt")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for debt amount and term")
    
    def update_department_frames(self, country):
        """Update the department budget frames"""
        if not hasattr(self, "department_frames"):
            self.department_frames = {}
            
        # Find the departments list
        departments_list = None
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkTabview):
                for tab in widget.winfo_children():
                    if "departments" in tab.winfo_name().lower():
                        for frame in tab.winfo_children():
                            if isinstance(frame, ctk.CTkScrollableFrame):
                                departments_list = frame
                                break
        
        if departments_list is None:
            return
            
        # Clear existing department frames if needed
        if not self.department_frames and country.departments:
            # Create frames for each department
            for dept_name, dept in country.departments.items():
                dept_frame = ctk.CTkFrame(departments_list)
                dept_frame.pack(fill=tk.X, padx=10, pady=5)
                
                # Department header
                header_frame = ctk.CTkFrame(dept_frame)
                header_frame.pack(fill=tk.X, padx=5, pady=5)
                
                ctk.CTkLabel(
                    header_frame,
                    text=dept.name,
                    font=("Helvetica", 14, "bold")
                ).pack(side=tk.LEFT, padx=10)
                
                # Budget display
                budget_display = ctk.CTkLabel(
                    header_frame,
                    text=self.format_currency(dept.budget)
                )
                budget_display.pack(side=tk.RIGHT, padx=10)
                
                # Stats
                stats_frame = ctk.CTkFrame(dept_frame)
                stats_frame.pack(fill=tk.X, padx=5, pady=2)
                
                # Staff count
                staff_label = ctk.CTkLabel(
                    stats_frame,
                    text=f"Staff: {dept.staff:,}"
                )
                staff_label.pack(side=tk.LEFT, padx=10)
                
                # Infrastructure
                infra_label = ctk.CTkLabel(
                    stats_frame,
                    text=f"Infrastructure: {dept.infrastructure}"
                )
                infra_label.pack(side=tk.LEFT, padx=10)
                
                # Efficiency
                efficiency_label = ctk.CTkLabel(
                    stats_frame,
                    text=f"Efficiency: {dept.efficiency * 100:.1f}%"
                )
                efficiency_label.pack(side=tk.LEFT, padx=10)
                
                # Citizen satisfaction
                satis_label = ctk.CTkLabel(
                    stats_frame,
                    text=f"Satisfaction: {dept.satisfaction * 100:.1f}%"
                )
                satis_label.pack(side=tk.LEFT, padx=10)
                
                # Budget controls
                controls_frame = ctk.CTkFrame(dept_frame)
                controls_frame.pack(fill=tk.X, padx=5, pady=5)
                
                ctk.CTkLabel(controls_frame, text="Adjust Budget:").pack(side=tk.LEFT, padx=5)
                
                # Budget slider
                slider = ctk.CTkSlider(
                    controls_frame,
                    from_=0,
                    to=dept.budget * 2,  # Allow doubling the budget
                    number_of_steps=100
                )
                slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                slider.set(dept.budget)  # Set to current budget
                
                # Budget entry for precise input
                budget_entry = ctk.CTkEntry(controls_frame, width=100)
                budget_entry.pack(side=tk.LEFT, padx=5)
                budget_entry.insert(0, str(int(dept.budget / 1e6)))  # Show in millions
                ctk.CTkLabel(controls_frame, text="million").pack(side=tk.LEFT)
                
                # Apply button
                apply_btn = ctk.CTkButton(
                    controls_frame,
                    text="Apply",
                    width=80,
                    command=lambda dn=dept_name, be=budget_entry: self.update_dept_budget(dn, be)
                )
                apply_btn.pack(side=tk.RIGHT, padx=5)
                
                # Actions frame for department-specific actions
                actions_frame = ctk.CTkFrame(dept_frame)
                actions_frame.pack(fill=tk.X, padx=5, pady=5)
                
                # Different actions for different departments
                if dept_name == "Healthcare":
                    ctk.CTkButton(
                        actions_frame,
                        text="Build Hospital",
                        command=lambda dn=dept_name: self.dept_action(dn, "build_hospital")
                    ).pack(side=tk.LEFT, padx=5)
                    
                    ctk.CTkButton(
                        actions_frame,
                        text="Hire Doctors",
                        command=lambda dn=dept_name: self.dept_action(dn, "hire_medical_staff")
                    ).pack(side=tk.LEFT, padx=5)
                elif dept_name == "Education":
                    ctk.CTkButton(
                        actions_frame,
                        text="Build School",
                        command=lambda dn=dept_name: self.dept_action(dn, "build_school")
                    ).pack(side=tk.LEFT, padx=5)
                    
                    ctk.CTkButton(
                        actions_frame,
                        text="Hire Teachers",
                        command=lambda dn=dept_name: self.dept_action(dn, "hire_teachers")
                    ).pack(side=tk.LEFT, padx=5)
                elif dept_name == "Defense":
                    ctk.CTkButton(
                        actions_frame,
                        text="Purchase Equipment",
                        command=lambda dn=dept_name: self.dept_action(dn, "buy_equipment")
                    ).pack(side=tk.LEFT, padx=5)
                    
                    ctk.CTkButton(
                        actions_frame,
                        text="Recruit Soldiers",
                        command=lambda dn=dept_name: self.dept_action(dn, "recruit_soldiers")
                    ).pack(side=tk.LEFT, padx=5)
                else:
                    ctk.CTkButton(
                        actions_frame,
                        text="Improve Efficiency",
                        command=lambda dn=dept_name: self.dept_action(dn, "improve_efficiency")
                    ).pack(side=tk.LEFT, padx=5)
                    
                    ctk.CTkButton(
                        actions_frame,
                        text="Hire Staff",
                        command=lambda dn=dept_name: self.dept_action(dn, "hire_staff")
                    ).pack(side=tk.LEFT, padx=5)
                
                # Store references to update later
                self.department_frames[dept_name] = {
                    "budget_display": budget_display,
                    "staff_label": staff_label,
                    "infra_label": infra_label,
                    "efficiency_label": efficiency_label,
                    "satisfaction_label": satis_label,
                    "slider": slider,
                    "budget_entry": budget_entry
                }
        else:
            # Update existing department frames
            for dept_name, dept in country.departments.items():
                if dept_name in self.department_frames:
                    controls = self.department_frames[dept_name]
                    
                    controls["budget_display"].configure(text=self.format_currency(dept.budget))
                    controls["staff_label"].configure(text=f"Staff: {dept.staff:,}")
                    controls["infra_label"].configure(text=f"Infrastructure: {dept.infrastructure}")
                    controls["efficiency_label"].configure(text=f"Efficiency: {dept.efficiency * 100:.1f}%")
                    controls["satisfaction_label"].configure(text=f"Satisfaction: {dept.satisfaction * 100:.1f}%")
                    
                    controls["slider"].configure(to=dept.budget * 2)
                    controls["slider"].set(dept.budget)
                    
                    controls["budget_entry"].delete(0, tk.END)
                    controls["budget_entry"].insert(0, str(int(dept.budget / 1e6)))
    
    def update_dept_budget(self, dept_name, budget_entry):
        """Update a department's budget"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        try:
            # Get budget in millions
            budget_millions = float(budget_entry.get())
            
            # Convert to actual value
            budget = budget_millions * 1000000
            
            # Update the department budget
            if country.adjust_department_budget(dept_name, budget):
                messagebox.showinfo("Budget Updated", f"{dept_name} budget updated to {self.format_currency(budget)}")
            else:
                messagebox.showerror("Error", "Insufficient funds to allocate this budget")
                
            # Update display
            self.update_display()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid budget amount")
    
    def dept_action(self, dept_name, action):
        """Perform a department-specific action"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country) or dept_name not in country.departments:
            return
            
        dept = country.departments[dept_name]
        
        # Define actions and their effects
        actions = {
            "build_hospital": {
                "cost": 500000000,  # $500 million per hospital
                "effect": lambda d: d.build_infrastructure(1, 500000000),
                "message": "New hospital built! Healthcare capacity increased."
            },
            "hire_medical_staff": {
                "cost": 100000000,  # $100 million for staff
                "effect": lambda d: d.hire_staff(1000, 100000),
                "message": "1,000 new healthcare workers hired!"
            },
            "build_school": {
                "cost": 200000000,  # $200 million per school
                "effect": lambda d: d.build_infrastructure(1, 200000000),
                "message": "New school built! Education capacity increased."
            },
            "hire_teachers": {
                "cost": 80000000,  # $80 million for teachers
                "effect": lambda d: d.hire_staff(1000, 80000),
                "message": "1,000 new teachers hired!"
            },
            "buy_equipment": {
                "cost": 1000000000,  # $1 billion for military equipment
                "effect": lambda d: d.build_infrastructure(5, 200000000),
                "message": "New military equipment purchased! Defense capability improved."
            },
            "recruit_soldiers": {
                "cost": 200000000,  # $200 million for soldiers
                "effect": lambda d: d.hire_staff(10000, 20000),
                "message": "10,000 new soldiers recruited!"
            },
            "improve_efficiency": {
                "cost": 50000000,  # $50 million
                "effect": lambda d: setattr(d, "efficiency", min(1.0, d.efficiency + 0.05)) or True,
                "message": "Department efficiency improved through modernization and training."
            },
            "hire_staff": {
                "cost": 30000000,  # $30 million
                "effect": lambda d: d.hire_staff(500, 60000),
                "message": "500 new staff members hired!"
            }
        }
        
        if action not in actions:
            messagebox.showerror("Error", "Unknown action")
            return
            
        action_info = actions[action]
        
        # Check if department has enough budget
        if dept.budget < action_info["cost"]:
            messagebox.showerror("Insufficient Budget", 
                               f"This action requires {self.format_currency(action_info['cost'])}, "
                               f"but {dept_name} only has {self.format_currency(dept.budget)}")
            return
            
        # Perform the action
        success = action_info["effect"](dept)
        
        if success:
            # Deduct cost
            dept.budget -= action_info["cost"]
            
            # Show success message
            messagebox.showinfo("Action Successful", action_info["message"])
            
            # Update display
            self.update_display()
        else:
            messagebox.showerror("Action Failed", "The action could not be completed")
    
    def update_budget_charts(self, country):
        """Update the budget pie charts"""
        if not hasattr(self, "revenue_chart") or not hasattr(self, "expenditure_chart"):
            return
            
        # Clear previous plots
        self.revenue_chart.clear()
        self.expenditure_chart.clear()
        
        # Revenue chart data
        revenue_sources = {
            "Income Tax": country.gdp * 0.15,
            "Corporate Tax": country.gdp * 0.05,
            "Sales Tax": country.gdp * 0.07,
            "Property Tax": country.gdp * 0.02,
            "Other Taxes": country.gdp * 0.03
        }
        
        revenue_labels = list(revenue_sources.keys())
        revenue_values = list(revenue_sources.values())
        
        # Create revenue pie chart
        self.revenue_chart.pie(
            revenue_values, 
            labels=revenue_labels, 
            autopct='%1.1f%%', 
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        self.revenue_chart.axis('equal')
        self.revenue_chart.set_title("Revenue Sources")
        
        # Expenditure chart data
        expenditure = {}
        for dept_name, dept in country.departments.items():
            expenditure[dept_name] = dept.budget
        
        expenditure_labels = list(expenditure.keys())
        expenditure_values = list(expenditure.values())
        
        # Create expenditure pie chart
        self.expenditure_chart.pie(
            expenditure_values, 
            labels=expenditure_labels, 
            autopct='%1.1f%%', 
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        self.expenditure_chart.axis('equal')
        self.expenditure_chart.set_title("Expenditure Allocation")
        
        # Update the canvases
        self.revenue_figure.tight_layout()
        self.revenue_canvas.draw()
        
        self.expenditure_figure.tight_layout()
        self.expenditure_canvas.draw()
        
        # Update budget summary if present
        if hasattr(self, "budget_stats"):
            # Calculate totals
            total_revenue = sum(revenue_values)
            total_expenditure = sum(expenditure_values)
            balance = total_revenue - total_expenditure
            
            if "Total Revenue" in self.budget_stats:
                self.budget_stats["Total Revenue"].configure(text=self.format_currency(total_revenue))
            
            if "Total Expenditure" in self.budget_stats:
                self.budget_stats["Total Expenditure"].configure(text=self.format_currency(total_expenditure))
            
            if "Budget Balance" in self.budget_stats:
                self.budget_stats["Budget Balance"].configure(text=self.format_currency(balance))
            
            if "Budget as % of GDP" in self.budget_stats:
                budget_percent = (total_expenditure / country.gdp) * 100
                self.budget_stats["Budget as % of GDP"].configure(text=f"{budget_percent:.1f}%")
    
    def update_election_timeline(self):
        """Update the election timeline display"""
        if not hasattr(self, "next_election_label"):
            return
            
        # Simulate a future election date
        current_date = self.app.game_state.current_date
        
        # Example: elections every 4 years
        next_election_year = current_date.year + 4
        next_election_date = datetime.datetime(next_election_year, 11, 4)  # November 4
        
        # If the election has passed, set the next one
        if next_election_date < current_date:
            next_election_year += 4
            next_election_date = datetime.datetime(next_election_year, 11, 4)
        
        # Calculate days remaining
        days_remaining = (next_election_date - current_date).days
        
        # Update the label
        election_str = next_election_date.strftime("%B %d, %Y")
        self.next_election_label.configure(
            text=f"{election_str} ({days_remaining} days remaining)"
        )
    
    def call_early_election(self):
        """Call an early election"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        result = messagebox.askyesno(
            "Call Early Election",
            "Are you sure you want to call an early election? "
            "This will test your current approval rating and could lead to losing power."
        )
        
        if result:
            # Simulate election based on approval/happiness
            approval = country.happiness * 100
            
            # 40% chance of winning if approval is 50%
            # Each point of approval above or below 50% adds or subtracts 2% chance
            win_chance = 40 + (approval - 50) * 2
            
            # Random result
            win = random.random() * 100 < win_chance
            
            if win:
                messagebox.showinfo(
                    "Election Victory!",
                    f"Congratulations! With an approval rating of {approval:.1f}%, "
                    f"you have won the election and will continue to lead the country."
                )
                
                # Update next election date
                current_date = self.app.game_state.current_date
                next_election_year = current_date.year + 4
                next_election_date = datetime.datetime(next_election_year, 11, 4)
                days_remaining = (next_election_date - current_date).days
                
                election_str = next_election_date.strftime("%B %d, %Y")
                self.next_election_label.configure(
                    text=f"{election_str} ({days_remaining} days remaining)"
                )
            else:
                messagebox.showinfo(
                    "Election Defeat",
                    f"With an approval rating of {approval:.1f}%, "
                    f"you have lost the election. The opposition will now lead the country."
                )
                
                # End the game or handle defeat
                defeat_dialog = ctk.CTkToplevel(self)
                defeat_dialog.title("Game Over")
                defeat_dialog.geometry("400x300")
                defeat_dialog.transient(self)  # Make it a modal dialog
                defeat_dialog.grab_set()
                
                ctk.CTkLabel(
                    defeat_dialog,
                    text="Election Lost",
                    font=("Helvetica", 20, "bold")
                ).pack(pady=(20, 10))
                
                ctk.CTkLabel(
                    defeat_dialog,
                    text="You have lost the election and your time in office has ended.",
                    wraplength=350
                ).pack(pady=10)
                
                ctk.CTkLabel(
                    defeat_dialog,
                    text=f"Final approval rating: {approval:.1f}%",
                    font=("Helvetica", 14)
                ).pack(pady=10)
                
                ctk.CTkButton(
                    defeat_dialog,
                    text="Return to Main Menu",
                    command=lambda: self.handle_game_over(defeat_dialog)
                ).pack(pady=20)
    
    def handle_game_over(self, dialog):
        """Handle the end of the game"""
        dialog.destroy()
        self.app.show_frame("start")
    
    def update_relations_list(self, country):
        """Update the list of country relations"""
        if not hasattr(self, "relations_list"):
            return
            
        # Clear existing items
        for widget in self.relations_list.winfo_children():
            widget.destroy()
        
        # Add relations with other countries
        for other_id, other_country in self.app.game_state.countries.items():
            if other_country.name == country.name:
                continue
                
            # Get relation value (or default to neutral)
            relation = country.international_relations.get(other_country.name, 0)
            
            # Create a frame for this country
            relation_frame = ctk.CTkFrame(self.relations_list)
            relation_frame.pack(fill=tk.X, padx=5, pady=3)
            
            # Country name
            ctk.CTkLabel(
                relation_frame,
                text=other_country.name,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            # Relation status
            status_text = self.get_relation_status(relation)
            text_color = self.get_relation_color(relation)
            
            ctk.CTkLabel(
                relation_frame,
                text=status_text,
                text_color=text_color
            ).pack(side=tk.LEFT, padx=10)
            
            # Relation bar (visual representation of relation)
            bar_frame = ctk.CTkFrame(relation_frame, height=20, width=150)
            bar_frame.pack(side=tk.LEFT, padx=10)
            
            # Normalize relation from -1:1 to 0:1 for the progress bar
            normalized_relation = (relation + 1) / 2
            
            relation_bar = ctk.CTkProgressBar(bar_frame, width=150)
            relation_bar.pack(side=tk.LEFT, fill=tk.Y)
            relation_bar.set(normalized_relation)
            
            # Set the color based on the relation value
            if relation < -0.5:
                relation_bar.configure(progress_color=DANGER_COLOR)
            elif relation < 0:
                relation_bar.configure(progress_color=WARNING_COLOR)
            elif relation < 0.5:
                relation_bar.configure(progress_color=PRIMARY_COLOR)
            else:
                relation_bar.configure(progress_color=SUCCESS_COLOR)
            
            # Actions button
            ctk.CTkButton(
                relation_frame,
                text="Diplomatic Actions",
                width=120,
                command=lambda c=other_country.name: self.open_diplomacy_for_country(c)
            ).pack(side=tk.RIGHT, padx=10)
    
    def get_relation_status(self, relation):
        """Convert relation value to text status"""
        if relation < -0.75:
            return "Hostile"
        elif relation < -0.5:
            return "Poor"
        elif relation < -0.25:
            return "Tense"
        elif relation < 0.25:
            return "Neutral"
        elif relation < 0.5:
            return "Cordial"
        elif relation < 0.75:
            return "Friendly"
        else:
            return "Ally"
    
    def get_relation_color(self, relation):
        """Get color based on relation value"""
        if relation < -0.5:
            return DANGER_COLOR
        elif relation < 0:
            return WARNING_COLOR
        elif relation < 0.5:
            return PRIMARY_COLOR
        else:
            return SUCCESS_COLOR
    
    def open_diplomacy_for_country(self, country_name):
        """Open the diplomacy tab and select the given country"""
        # Switch to the international tab
        self.app.show_international()
        
        # Find and select the diplomacy tab within international
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkTabview):
                widget.set("Diplomacy")
                break
        
        # Set the country selector
        if hasattr(self, "diplomacy_country"):
            # Add the country to the values if not present
            values = self.diplomacy_country._values
            if country_name not in values:
                values.append(country_name)
                self.diplomacy_country.configure(values=values)
            
            self.diplomacy_country.set(country_name)
    
    def execute_diplomatic_action(self):
        """Execute the selected diplomatic action on the selected country"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        # Get selected country and action
        target_country_name = self.diplomacy_country.get()
        action = self.diplomacy_action.get()
        
        # Find the target country
        target_country = None
        for c in self.app.game_state.countries.values():
            if c.name == target_country_name:
                target_country = c
                break
        
        if not target_country:
            messagebox.showerror("Error", f"Country '{target_country_name}' not found")
            return
        
        # Initialize relations if not present
        if target_country_name not in country.international_relations:
            country.international_relations[target_country_name] = 0
        
        # Initialize trade balance if not present
        if target_country_name not in country.trade_balance:
            country.trade_balance[target_country_name] = 0
        
        # Initialize target country's relations with player country
        if country.name not in target_country.international_relations:
            target_country.international_relations[country.name] = 0
        
        # Current relation
        current_relation = country.international_relations[target_country_name]
        
        # Process different actions
        result = None
        log_message = None
        
        if action == "Improve Relations":
            # Higher chance of success with more friendly relations
            success_chance = 0.7 + (0.3 * current_relation)
            success = random.random() < success_chance
            
            if success:
                # Improve relations by 0.1 to 0.2
                improvement = 0.1 + (random.random() * 0.1)
                new_relation = min(1.0, current_relation + improvement)
                country.international_relations[target_country_name] = new_relation
                target_country.international_relations[country.name] = new_relation
                
                result = "Success"
                log_message = (f"Diplomatic initiative to improve relations with {target_country_name} "
                              f"was successful. Relations improved from "
                              f"{self.get_relation_status(current_relation)} to "
                              f"{self.get_relation_status(new_relation)}.")
            else:
                result = "Failed"
                log_message = (f"Diplomatic initiative to improve relations with {target_country_name} "
                              f"failed. Relations remain {self.get_relation_status(current_relation)}.")
        
        elif action == "Establish Trade Agreement":
            # Trade agreements require somewhat positive relations
            if current_relation < -0.3:
                result = "Failed"
                log_message = (f"Cannot establish trade agreement with {target_country_name} "
                              f"due to poor relations ({self.get_relation_status(current_relation)}).")
            else:
                # Calculate potential trade volume based on both countries' GDP
                trade_volume = (country.gdp * 0.01) + (target_country.gdp * 0.01)
                
                # Randomize which country benefits more (slightly favor player)
                player_benefit = trade_volume * (0.4 + (random.random() * 0.3))
                target_benefit = trade_volume - player_benefit
                
                # Update trade balance (positive means net exports, negative means net imports)
                country.trade_balance[target_country_name] = player_benefit - target_benefit
                
                # Improve relations
                improvement = 0.15
                new_relation = min(1.0, current_relation + improvement)
                country.international_relations[target_country_name] = new_relation
                target_country.international_relations[country.name] = new_relation
                
                result = "Success"
                log_message = (f"Trade agreement established with {target_country_name}. "
                              f"Expected trade volume: {self.format_currency(trade_volume)}. "
                              f"Relations improved to {self.get_relation_status(new_relation)}.")
        
        # Add more diplomatic actions here as needed
        elif action == "Form Alliance":
            # Alliances require good relations
            if current_relation < 0.5:
                result = "Failed"
                log_message = (f"Cannot form alliance with {target_country_name} "
                              f"due to insufficient relations ({self.get_relation_status(current_relation)}). "
                              f"Relations need to be at least Friendly.")
            else:
                # Form alliance
                new_relation = 0.9  # Set to almost maximum
                country.international_relations[target_country_name] = new_relation
                target_country.international_relations[country.name] = new_relation
                
                result = "Success"
                log_message = (f"Alliance formed with {target_country_name}. "
                              f"Relations are now {self.get_relation_status(new_relation)}.")
        
        elif action == "Impose Sanctions":
            # Sanctions worsen relations but can be used regardless of current relations
            # Calculate economic impact
            target_economy_impact = target_country.gdp * 0.02  # 2% GDP loss
            own_economy_impact = country.gdp * 0.01  # 1% GDP loss (sanctions also hurt the imposer)
            
            # Apply effects
            target_country.gdp -= target_economy_impact
            country.gdp -= own_economy_impact
            
            # Worsen relations
            worsen = 0.3
            new_relation = max(-1.0, current_relation - worsen)
            country.international_relations[target_country_name] = new_relation
            target_country.international_relations[country.name] = new_relation
            
            result = "Imposed"
            log_message = (f"Sanctions imposed on {target_country_name}, causing an estimated "
                          f"{self.format_currency(target_economy_impact)} damage to their economy and "
                          f"{self.format_currency(own_economy_impact)} damage to ours. "
                          f"Relations worsened to {self.get_relation_status(new_relation)}.")
        
        elif action == "Declare War":
            # War is a serious action with major consequences
            confirm = messagebox.askyesno(
                "Confirm War Declaration",
                f"Are you absolutely sure you want to declare war on {target_country_name}? "
                f"This will have severe diplomatic, economic and potentially military consequences."
            )
            
            if confirm:
                # Set relations to minimum
                new_relation = -1.0
                country.international_relations[target_country_name] = new_relation
                target_country.international_relations[country.name] = new_relation
                
                # Economic impacts
                war_cost = country.gdp * 0.05  # 5% GDP loss from mobilization
                country.gdp -= war_cost
                
                # Affect relations with other countries
                for other_country_name in country.international_relations:
                    if other_country_name != target_country_name:
                        # Other countries reacct negatively to war
                        country.international_relations[other_country_name] -= 0.2
                
                result = "War Declared"
                log_message = (f"War declared on {target_country_name}! Initial mobilization cost: "
                              f"{self.format_currency(war_cost)}. Relations with other countries have "
                              f"deteriorated due to this aggressive action.")
            else:
                result = "Cancelled"
                log_message = f"War declaration on {target_country_name} cancelled."
        
        elif action == "Send Aid":
            # Aid improves relations
            aid_amount = country.gdp * 0.005  # 0.5% of GDP as aid
            
            # Check if can afford it
            if aid_amount > country.government_budget * 0.1:
                result = "Failed"
                log_message = (f"Cannot send aid to {target_country_name} due to budget constraints. "
                              f"Required: {self.format_currency(aid_amount)}.")
            else:
                # Send aid
                country.gdp -= aid_amount
                target_country.gdp += aid_amount
                
                # Improve relations
                improvement = 0.2
                new_relation = min(1.0, current_relation + improvement)
                country.international_relations[target_country_name] = new_relation
                target_country.international_relations[country.name] = new_relation
                
                result = "Success"
                log_message = (f"Aid package of {self.format_currency(aid_amount)} sent to "
                              f"{target_country_name}. Relations improved to "
                              f"{self.get_relation_status(new_relation)}.")
        
        elif action == "Cultural Exchange":
            # Cultural exchanges improve relations with low cost
            cost = country.gdp * 0.0005  # 0.05% of GDP
            
            country.gdp -= cost
            
            # Improve relations slightly
            improvement = 0.05 + (random.random() * 0.05)  # 0.05-0.1
            new_relation = min(1.0, current_relation + improvement)
            country.international_relations[target_country_name] = new_relation
            target_country.international_relations[country.name] = new_relation
            
            result = "Success"
            log_message = (f"Cultural exchange program established with {target_country_name} "
                          f"at a cost of {self.format_currency(cost)}. Relations improved slightly to "
                          f"{self.get_relation_status(new_relation)}.")
        
        # Add result to diplomacy log
        if log_message and hasattr(self, "diplomacy_log"):
            date_str = self.app.game_state.current_date.strftime("%Y-%m-%d")
            full_log = f"[{date_str}] {result}: {log_message}\n\n" + self.diplomacy_log.get("1.0", tk.END)
            
            # Update log display
            self.diplomacy_log.configure(state=tk.NORMAL)
            self.diplomacy_log.delete("1.0", tk.END)
            self.diplomacy_log.insert(tk.END, full_log)
            self.diplomacy_log.configure(state=tk.DISABLED)
        
        # Show message
        messagebox.showinfo(f"Diplomatic Action: {result}", log_message)
        
        # Update display
        self.update_display()
    
    def update_resource_info(self, country):
        """Update the natural resource information"""
        if not hasattr(self, "resource_labels"):
            return
            
        # Update resource amounts
        for resource, amount in country.natural_resources.items():
            if resource in self.resource_labels:
                # Format the amount based on resource type
                if resource in ["oil", "natural_gas"]:
                    formatted = f"{amount:.1f} million barrels"
                elif resource in ["coal", "iron", "copper"]:
                    formatted = f"{amount:.1f} million tons"
                elif resource == "gold":
                    formatted = f"{amount:.1f} tons"
                elif resource == "uranium":
                    formatted = f"{amount:.1f} thousand tons"
                elif resource in ["arable_land", "fresh_water", "forests"]:
                    formatted = f"{amount:.1f}% of land area"
                else:
                    formatted = f"{amount:.1f} units"
                
                self.resource_labels[resource].configure(text=formatted)
        
        # Update resources chart
        if hasattr(self, "resources_chart"):
            self.update_resources_chart(country)
    
    def update_resources_chart(self, country):
        """Update the natural resources chart"""
        if not hasattr(self, "resources_chart"):
            return
            
        # Clear previous plot
        self.resources_chart.clear()
        
        # Filter for mineral/energy resources only to make chart readable
        energy_resources = {k: v for k, v in country.natural_resources.items() 
                          if k in ["oil", "natural_gas", "coal", "uranium"]}
        mineral_resources = {k: v for k, v in country.natural_resources.items()
                           if k in ["iron", "copper", "gold"]}
        
        # Normalize values for display
        energy_values = list(energy_resources.values())
        energy_labels = list(energy_resources.keys())
        
        mineral_values = list(mineral_resources.values())
        mineral_labels = list(mineral_resources.keys())
        
        # Create a 2-part chart
        if energy_values:
            ax1 = self.resources_chart
            ax1.bar(energy_labels, energy_values, color=PRIMARY_COLOR, alpha=0.7)
            ax1.set_ylabel('Amount (in millions)')
            ax1.set_title('Energy Resources')
            
            # Add mineral resources as a second axis
            if mineral_values:
                ax2 = ax1.twinx()
                ax2.bar([f"{label}_" for label in mineral_labels], mineral_values, color=SECONDARY_COLOR, alpha=0.7)
                ax2.set_ylabel('Amount (in millions)')
                
                # Combine legends
                ax1.legend(energy_labels, loc='upper left')
                ax2.legend(mineral_labels, loc='upper right')
        
        # Update the canvas
        self.resources_figure.tight_layout()
        self.resources_canvas.draw()
    
    def extract_resource(self, resource):
        """Extract more of a natural resource"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        if resource not in country.natural_resources:
            messagebox.showerror("Error", f"Resource '{resource}' not found")
            return
            
        # Calculate extraction amount and cost
        current_amount = country.natural_resources[resource]
        extraction_percent = 0.1  # Extract 10% of current amount
        extraction_amount = current_amount * extraction_percent
        
        # Cost depends on resource type
        cost_per_unit = {
            "oil": 10000000,  # $10M per million barrels
            "natural_gas": 8000000,  # $8M per million cubic feet
            "coal": 5000000,  # $5M per million tons
            "iron": 7000000,  # $7M per million tons
            "copper": 15000000,  # $15M per million tons
            "gold": 100000000,  # $100M per ton
            "uranium": 20000000,  # $20M per thousand tons
        }
        
        extraction_cost = extraction_amount * cost_per_unit.get(resource, 10000000)
        
        # Check if country can afford it
        if extraction_cost > country.government_budget * 0.05:
            messagebox.showerror(
                "Insufficient Funds",
                f"Extracting {extraction_amount:.2f} units of {resource} would cost "
                f"{self.format_currency(extraction_cost)}, which is too expensive."
            )
            return
            
        # Confirmation dialog
        confirm = messagebox.askyesno(
            "Confirm Resource Extraction",
            f"Extract {extraction_amount:.2f} units of {resource} at a cost of "
            f"{self.format_currency(extraction_cost)}?"
        )
        
        if not confirm:
            return
            
        # Perform extraction
        country.gdp -= extraction_cost
        
        # Reduce natural resource
        country.natural_resources[resource] -= extraction_amount
        
        # Add to economy
        extraction_value = extraction_amount * country.resource_markets[resource].current_price * 1000000
        country.gdp += extraction_value
        
        # Show results
        messagebox.showinfo(
            "Resource Extraction",
            f"Successfully extracted {extraction_amount:.2f} units of {resource}. "
            f"Cost: {self.format_currency(extraction_cost)}. "
            f"Value: {self.format_currency(extraction_value)}. "
            f"Net profit: {self.format_currency(extraction_value - extraction_cost)}."
        )
        
        # Update display
        self.update_display()
    
    def trade_resource(self, resource):
        """Trade a natural resource on the international market"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        if resource not in country.natural_resources or resource not in country.resource_markets:
            messagebox.showerror("Error", f"Resource '{resource}' or its market not found")
            return
            
        # Create trade dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Trade {resource.capitalize()}")
        dialog.geometry("400x300")
        dialog.transient(self)  # Make it a modal dialog
        dialog.grab_set()
        
        # Resource info
        info_frame = ctk.CTkFrame(dialog)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Current amount and price
        current_amount = country.natural_resources[resource]
        current_price = country.resource_markets[resource].current_price
        
        ctk.CTkLabel(
            info_frame,
            text=f"Available {resource.capitalize()}: {current_amount:.2f} units",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        ctk.CTkLabel(
            info_frame,
            text=f"Current Market Price: ${current_price:.2f} per unit",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Trade controls
        trade_frame = ctk.CTkFrame(dialog)
        trade_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Buy/Sell selector
        action_frame = ctk.CTkFrame(trade_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        action_var = tk.StringVar(value="buy")
        
        ctk.CTkLabel(action_frame, text="Action:").pack(side=tk.LEFT, padx=5)
        
        ctk.CTkRadioButton(
            action_frame,
            text="Buy",
            variable=action_var,
            value="buy"
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkRadioButton(
            action_frame,
            text="Sell",
            variable=action_var,
            value="sell"
        ).pack(side=tk.LEFT, padx=10)
        
        # Amount input
        amount_frame = ctk.CTkFrame(trade_frame)
        amount_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(amount_frame, text="Amount (units):").pack(side=tk.LEFT, padx=5)
        
        amount_entry = ctk.CTkEntry(amount_frame, width=100)
        amount_entry.pack(side=tk.LEFT, padx=5)
        amount_entry.insert(0, "1.0")
        
        # Total value display
        total_frame = ctk.CTkFrame(trade_frame)
        total_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(total_frame, text="Total Value:").pack(side=tk.LEFT, padx=5)
        
        total_value_label = ctk.CTkLabel(total_frame, text="$0")
        total_value_label.pack(side=tk.LEFT, padx=5)
        
        # Update total value when amount changes
        def update_total(*args):
            try:
                amount = float(amount_entry.get())
                total = amount * current_price
                total_value_label.configure(text=f"${total:,.2f}")
            except ValueError:
                total_value_label.configure(text="Invalid input")
        
        # Bind to entry changes
        amount_entry.bind("<KeyRelease>", update_total)
        update_total()  # Initial update
        
        # Execute button
        ctk.CTkButton(
            dialog,
            text="Execute Trade",
            command=lambda: self.execute_resource_trade(
                dialog, country, resource, action_var.get(), amount_entry, current_price
            )
        ).pack(pady=10)
    
    def execute_resource_trade(self, dialog, country, resource, action, amount_entry, price):
        """Execute a resource trade"""
        try:
            amount = float(amount_entry.get())
            if amount <= 0:
                raise ValueError("Amount must be positive")
                
            total_value = amount * price
            
            if action == "buy":
                # Check if country can afford it
                if total_value > country.government_budget * 0.1:
                    messagebox.showerror(
                        "Insufficient Funds",
                        f"Buying {amount:.2f} units of {resource} would cost "
                        f"${total_value:,.2f}, which is too expensive."
                    )
                    return
                    
                # Buy the resource
                country.gdp -= total_value
                country.natural_resources[resource] += amount
                
                messagebox.showinfo(
                    "Trade Successful",
                    f"Successfully purchased {amount:.2f} units of {resource} for ${total_value:,.2f}."
                )
            else:  # sell
                # Check if country has enough
                if amount > country.natural_resources[resource]:
                    messagebox.showerror(
                        "Insufficient Resources",
                        f"You only have {country.natural_resources[resource]:.2f} units of {resource} "
                        f"available to sell."
                    )
                    return
                    
                # Sell the resource
                country.gdp += total_value
                country.natural_resources[resource] -= amount
                
                messagebox.showinfo(
                    "Trade Successful",
                    f"Successfully sold {amount:.2f} units of {resource} for ${total_value:,.2f}."
                )
            
            # Close the dialog
            dialog.destroy()
            
            # Update display
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def fund_research(self, area):
        """Fund research in a specific technology area"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        # Calculate funding amount
        funding_amount = country.gdp * 0.005  # 0.5% of GDP
        
        # Confirmation dialog
        confirm = messagebox.askyesno(
            "Confirm Research Funding",
            f"Allocate {self.format_currency(funding_amount)} to {area} research?"
        )
        
        if not confirm:
            return
            
        # Apply funding
        country.gdp -= funding_amount
        
        # Research effects (simulated)
        tech_level_increase = 0.1 + (random.random() * 0.1)
        
        # Show results
        messagebox.showinfo(
            "Research Funded",
            f"Successfully allocated {self.format_currency(funding_amount)} to {area} research. "
            f"Technology level in this area increased by {tech_level_increase:.1f} points."
        )
        
        # Update display
        self.update_display()
    
    def start_infra_project(self, project, cost, time):
        """Start an infrastructure project"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        # Convert cost to actual value
        cost_value = cost * 1000000000  # Convert billions to actual value
        
        # Check if country can afford it
        if cost_value > country.government_budget * 0.2:
            messagebox.showerror(
                "Insufficient Funds",
                f"The {project} would cost {self.format_currency(cost_value)}, "
                f"which is too expensive for the current budget."
            )
            return
            
        # Confirmation dialog
        confirm = messagebox.askyesno(
            "Confirm Infrastructure Project",
            f"Start the {project} project?\n\n"
            f"Cost: {self.format_currency(cost_value)}\n"
            f"Time: {time} years"
        )
        
        if not confirm:
            return
            
        # Start the project (in a real game, this would be added to an active projects list)
        country.gdp -= cost_value
        
        # Add to infrastructure department budget
        if "infrastructure" in country.departments:
            country.departments["infrastructure"].budget += cost_value * 0.5
            country.departments["infrastructure"].build_infrastructure(int(cost_value / 1000000000), 1000000000)
        
        # Show results
        messagebox.showinfo(
            "Project Started",
            f"The {project} project has been initiated. "
            f"It will take approximately {time} years to complete."
        )
        
        # Update display
        self.update_display()
    
    def start_national_project(self, project, cost, years):
        """Start a national project"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        # Convert cost to actual value
        cost_value = cost * 1000000000  # Convert billions to actual value
        
        # Check if country can afford it
        if cost_value > country.government_budget * 0.3:
            messagebox.showerror(
                "Insufficient Funds",
                f"The {project} would cost {self.format_currency(cost_value)}, "
                f"which is too expensive for the current budget."
            )
            return
            
        # Confirmation dialog
        confirm = messagebox.askyesno(
            "Confirm National Project",
            f"Start the {project} project?\n\n"
            f"Cost: {self.format_currency(cost_value)}\n"
            f"Time: {years} years"
        )
        
        if not confirm:
            return
            
        # Start the project
        country.gdp -= cost_value
        
        # Add to active projects list
        if hasattr(self, "active_projects_list"):
            # Remove "No active projects" label if present
            for widget in self.active_projects_list.winfo_children():
                widget.destroy()
            
            # Create project entry
            project_frame = ctk.CTkFrame(self.active_projects_list)
            project_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkLabel(
                project_frame,
                text=project,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            progress_bar = ctk.CTkProgressBar(project_frame, width=150)
            progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.Y)
            progress_bar.set(0)  # Just started
            
            years_label = ctk.CTkLabel(
                project_frame,
                text=f"Time remaining: {years} years"
            )
            years_label.pack(side=tk.LEFT, padx=10)
        
        # Show results
        messagebox.showinfo(
            "Project Started",
            f"The {project} project has been initiated. "
            f"It will take approximately {years} years to complete."
        )
        
        # Update display
        self.update_display()
    
    def replace_cabinet_member(self, position):
        """Replace a cabinet member"""
        # In a real game, this would show a list of candidates with different skills
        first_names = ["John", "Sarah", "Michael", "Emma", "David", "Michelle", 
                      "Robert", "Jennifer", "William", "Elizabeth"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller",
                     "Davis", "Garcia", "Rodriguez", "Wilson"]
        
        # Generate three candidates
        candidates = []
        for _ in range(3):
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            skill = random.randint(60, 95)
            candidates.append((name, skill))
        
        # Create selection dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Select New {position}")
        dialog.geometry("400x300")
        dialog.transient(self)  # Make it a modal dialog
        dialog.grab_set()
        
        ctk.CTkLabel(
            dialog,
            text=f"Choose a new {position}:",
            font=("Helvetica", 14, "bold")
        ).pack(pady=(20, 10))
        
        # Create a frame for candidates
        candidates_frame = ctk.CTkFrame(dialog)
        candidates_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add candidates
        selected_candidate = tk.StringVar()
        
        for name, skill in candidates:
            candidate_frame = ctk.CTkFrame(candidates_frame)
            candidate_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkRadioButton(
                candidate_frame,
                text=f"{name} (Skill: {skill}/100)",
                variable=selected_candidate,
                value=name
            ).pack(side=tk.LEFT, padx=10)
        
        # Select first by default
        if candidates:
            selected_candidate.set(candidates[0][0])
        
        # Submit button
        ctk.CTkButton(
            dialog,
            text="Appoint",
            command=lambda: self.appoint_cabinet_member(dialog, position, selected_candidate.get())
        ).pack(pady=10)
    
    def appoint_cabinet_member(self, dialog, position, name):
        """Appoint the selected cabinet member"""
        if not name:
            messagebox.showerror("Error", "No candidate selected")
            return
            
        # In a real game, this would update the government cabinet
        messagebox.showinfo("Appointment", f"{name} has been appointed as the new {position}.")
        
        # Close dialog
        dialog.destroy()
    
    def implement_policy(self, policy, category):
        """Implement a government policy"""
        country = self.app.game_state.get_player_entity()
        if not country or not isinstance(country, Country):
            return
            
        # Cost and effect depend on policy and category
        costs = {
            "Economic": {
                "Minimum Wage Increase": {"cost": 0.005, "happiness": 0.05, "unemployment": 0.01},
                "Corporate Tax Reform": {"cost": 0.001, "growth": 0.01, "tax_revenue": 0.02},
                "Infrastructure Investment": {"cost": 0.02, "growth": 0.015, "infrastructure": 0.1},
                "Research & Development Grants": {"cost": 0.01, "growth": 0.02, "tax_revenue": -0.01},
                "Small Business Loans": {"cost": 0.007, "growth": 0.01, "unemployment": -0.01},
                "Trade Tariffs": {"cost": 0.002, "growth": -0.01, "tax_revenue": 0.03},
                "Universal Basic Income": {"cost": 0.03, "happiness": 0.1, "tax_revenue": -0.05},
                "Banking Regulations": {"cost": 0.001, "growth": -0.005, "stability": 0.05}
            },
            "Social": {
                "Universal Healthcare": {"cost": 0.04, "happiness": 0.15, "tax_rate": 0.03},
                "Free College Education": {"cost": 0.025, "happiness": 0.1, "tax_rate": 0.02},
                "Affordable Housing Initiative": {"cost": 0.015, "happiness": 0.07, "tax_rate": 0.01},
                "Police Reform": {"cost": 0.01, "happiness": 0.05, "stability": 0.03},
                "Immigration Reform": {"cost": 0.005, "happiness": 0.03, "population_growth": 0.002},
                "Drug Decriminalization": {"cost": 0.003, "happiness": 0.04, "stability": -0.02},
                "Gun Control Measures": {"cost": 0.002, "happiness": 0.03, "stability": 0.02},
                "Parental Leave Policy": {"cost": 0.008, "happiness": 0.06, "population_growth": 0.001}
            }
        }
        
        if category not in costs or policy not in costs[category]:
            messagebox.showinfo("Policy Implementation", "This policy is not yet available.")
            return
            
        policy_info = costs[category][policy]
        
        # Calculate cost as percentage of GDP
        cost_value = country.gdp * policy_info["cost"]
        
        # Confirmation dialog
        effects_text = "\n".join([f"{k.replace('_', ' ').title()}: {'+'if v > 0 else ''}{v*100:.1f}%" 
                                for k, v in policy_info.items() if k != "cost"])
        
        confirm = messagebox.askyesno(
            "Confirm Policy Implementation",
            f"Implement {policy}?\n\n"
            f"Cost: {self.format_currency(cost_value)}\n\n"
            f"Effects:\n{effects_text}"
        )
        
        if not confirm:
            return
            
        # Implement policy
        country.gdp -= cost_value
        
        # Apply effects
        if "happiness" in policy_info:
            country.happiness = min(1.0, country.happiness + policy_info["happiness"])
        
        if "growth" in policy_info:
            country.gdp_growth += policy_info["growth"]
        
        if "unemployment" in policy_info:
            country.unemployment_rate = max(0.01, country.unemployment_rate + policy_info["unemployment"])
        
        if "tax_revenue" in policy_info:
            # Adjust tax rates
            for tax in country.taxes.values():
                tax.rate = max(0.01, min(0.9, tax.rate * (1 + policy_info["tax_revenue"])))
        
        if "tax_rate" in policy_info:
            country.tax_rate = min(0.9, country.tax_rate + policy_info["tax_rate"])
        
        if "infrastructure" in policy_info and "infrastructure" in country.departments:
            country.departments["infrastructure"].infrastructure += int(policy_info["infrastructure"] * 100)
        
        if "stability" in policy_info:
            country.political_stability = min(1.0, max(0.1, country.political_stability + policy_info["stability"]))
        
        if "population_growth" in policy_info:
            country.birth_rate += policy_info["population_growth"]
        
        # Show results
        messagebox.showinfo(
            "Policy Implemented",
            f"The {policy} has been implemented at a cost of {self.format_currency(cost_value)}."
        )
        
        # Update display
        self.update_display()
    
    def format_currency(self, value):
        """Format a currency value based on size"""
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        else:
            return f"${value:.2f}"

class BusinessMainScreen(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.app = master
        
        # Tabs/sub-screens
        self.tabs = {}
        self.current_tab = None
        
        # Create layout
        self.create_widgets()
    
    def create_widgets(self):
        # Top info bar
        self.info_bar = ctk.CTkFrame(self, height=60)
        self.info_bar.pack(fill=tk.X, padx=10, pady=10)
        
        # Business name and logo
        self.business_label = ctk.CTkLabel(
            self.info_bar, 
            text="", 
            font=("Helvetica", 18, "bold")
        )
        self.business_label.pack(side=tk.LEFT, padx=15)
        
        # Key stats in info bar
        self.cash_label = self.create_stat_label(self.info_bar, "Cash:", "$0M")
        self.revenue_label = self.create_stat_label(self.info_bar, "Revenue:", "$0M/yr")
        self.employees_label = self.create_stat_label(self.info_bar, "Employees:", "0")
        self.profit_margin_label = self.create_stat_label(self.info_bar, "Profit Margin:", "0%")
        
        # Tabs for different sections
        self.tab_bar = ctk.CTkFrame(self)
        self.tab_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Create tab content frames
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tab buttons
        self.tabs["overview"] = self.create_tab("Overview", self.create_overview_tab)
        self.tabs["products"] = self.create_tab("Products", self.create_products_tab)
        self.tabs["finances"] = self.create_tab("Finances", self.create_finances_tab)
        self.tabs["operations"] = self.create_tab("Operations", self.create_operations_tab)
        self.tabs["marketing"] = self.create_tab("Marketing", self.create_marketing_tab)
        self.tabs["research"] = self.create_tab("Research", self.create_research_tab)
        self.tabs["competition"] = self.create_tab("Competition", self.create_competition_tab)
        
        # Create tab content frames (initially hidden)
        for tab_id, (_, content_creator) in self.tabs.items():
            content_frame = content_creator()
            content_frame.pack_forget()  # Hide initially
            self.tabs[tab_id] = (self.tabs[tab_id][0], content_frame)
        
        # Show the overview tab initially
        self.show_tab("overview")
    
    def create_stat_label(self, parent, label_text, value_text):
        """Helper to create a statistic label with value"""
        frame = ctk.CTkFrame(parent)
        frame.pack(side=tk.LEFT, padx=15)
        
        ctk.CTkLabel(
            frame,
            text=label_text,
            font=("Helvetica", 12)
        ).pack(side=tk.LEFT)
        
        value_label = ctk.CTkLabel(
            frame,
            text=value_text,
            font=("Helvetica", 12, "bold")
        )
        value_label.pack(side=tk.LEFT, padx=5)
        
        return value_label
    
    def create_tab(self, text, content_creator):
        """Create a tab button and its associated content"""
        button = ctk.CTkButton(
            self.tab_bar,
            text=text,
            height=30,
            border_width=1,
            fg_color="transparent",
            text_color=TEXT_COLOR,
            command=lambda t=text.lower(): self.show_tab(t)
        )
        button.pack(side=tk.LEFT, padx=5)
        
        return (button, content_creator)  # Content will be created when needed
    
    def show_tab(self, tab_id):
        """Switch to the specified tab"""
        # Update button styles
        for tid, (button, _) in self.tabs.items():
            if tid == tab_id:
                button.configure(fg_color=PRIMARY_COLOR, text_color="white", border_width=0)
            else:
                button.configure(fg_color="transparent", text_color=TEXT_COLOR, border_width=1)
        
        # Hide current content frame
        if self.current_tab and self.current_tab in self.tabs:
            _, content = self.tabs[self.current_tab]
            if content:
                content.pack_forget()
        
        # Show selected content frame
        _, content = self.tabs[tab_id]
        if content:
            content.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        self.current_tab = tab_id
    
    def create_overview_tab(self):
        """Create content for overview tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Split into columns
        left_col = ctk.CTkFrame(frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_col = ctk.CTkFrame(frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Business stats in left column
        stats_frame = ctk.CTkFrame(left_col)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(
            stats_frame,
            text="Business Statistics",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Create statistics table
        self.stat_labels = {}
        self.create_stat_row(stats_frame, "Cash", "$0M")
        self.create_stat_row(stats_frame, "Revenue (Quarter)", "$0M")
        self.create_stat_row(stats_frame, "Profit (Quarter)", "$0M")
        self.create_stat_row(stats_frame, "Profit Margin", "0%")
        self.create_stat_row(stats_frame, "Assets", "$0M")
        self.create_stat_row(stats_frame, "Liabilities", "$0M")
        self.create_stat_row(stats_frame, "Net Worth", "$0M")
        self.create_stat_row(stats_frame, "Employees", "0")
        self.create_stat_row(stats_frame, "Brand Value", "0")
        
        if_public_frame = ctk.CTkFrame(stats_frame)
        if_public_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.create_stat_row(if_public_frame, "Stock Price", "$0.00")
        self.create_stat_row(if_public_frame, "Market Cap", "$0M")
        self.create_stat_row(if_public_frame, "P/E Ratio", "0")
        
        # Revenue chart in right column
        chart_frame = ctk.CTkFrame(right_col)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            chart_frame,
            text="Revenue & Profit Trend",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Create a figure for the chart
        self.revenue_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.revenue_chart = self.revenue_figure.add_subplot(111)
        
        # Add the plot to the tkinter window
        self.revenue_canvas = FigureCanvasTkAgg(self.revenue_figure, master=chart_frame)
        self.revenue_canvas.draw()
        self.revenue_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Business notifications
        notifications_frame = ctk.CTkFrame(left_col)
        notifications_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(10, 5))
        
        ctk.CTkLabel(
            notifications_frame,
            text="Recent Notifications",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        self.notifications_text = ctk.CTkTextbox(notifications_frame, height=200)
        self.notifications_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.notifications_text.configure(state=tk.DISABLED)  # Read-only
        
        # Product summary in right column
        products_frame = ctk.CTkFrame(right_col)
        products_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(10, 5))
        
        ctk.CTkLabel(
            products_frame,
            text="Top Products",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        self.top_products_frame = ctk.CTkScrollableFrame(products_frame, height=200)
        self.top_products_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        return frame
    
    def create_stat_row(self, parent, label, value):
        """Create a row in the statistics table"""
        row = ctk.CTkFrame(parent)
        row.pack(fill=tk.X, padx=10, pady=2)
        
        ctk.CTkLabel(row, text=label, width=120, anchor="w").pack(side=tk.LEFT)
        value_label = ctk.CTkLabel(row, text=value, anchor="e")
        value_label.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        self.stat_labels[label] = value_label
        return value_label
    
    def create_products_tab(self):
        """Create content for products tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Create notebook/tabs for product sections
        tabview = ctk.CTkTabview(frame)
        tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs
        tabview.add("Products List")
        tabview.add("New Product")
        tabview.add("Production")
        tabview.add("Quality Control")
        
        # Products List tab
        products_list_frame = ctk.CTkFrame(tabview.tab("Products List"))
        products_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            products_list_frame,
            text="Your Products",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Create a scrollable list of products
        self.products_list = ctk.CTkScrollableFrame(products_list_frame)
        self.products_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # New Product tab
        new_product_frame = ctk.CTkFrame(tabview.tab("New Product"))
        new_product_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            new_product_frame,
            text="Develop a New Product",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Product creation form
        form_frame = ctk.CTkFrame(new_product_frame)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Product name
        name_frame = ctk.CTkFrame(form_frame)
        name_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(name_frame, text="Product Name:").pack(side=tk.LEFT, padx=10)
        self.new_product_name = ctk.CTkEntry(name_frame, width=200)
        self.new_product_name.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Category
        category_frame = ctk.CTkFrame(form_frame)
        category_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(category_frame, text="Category:").pack(side=tk.LEFT, padx=10)
        self.new_product_category = ctk.CTkEntry(category_frame, width=200)
        self.new_product_category.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Initial investment
        investment_frame = ctk.CTkFrame(form_frame)
        investment_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(investment_frame, text="Initial R&D Investment ($):").pack(side=tk.LEFT, padx=10)
        self.new_product_investment = ctk.CTkEntry(investment_frame, width=150)
        self.new_product_investment.pack(side=tk.LEFT, padx=10)
        self.new_product_investment.insert(0, "1000000")  # Default $1M
        
        # Pricing strategy
        price_frame = ctk.CTkFrame(form_frame)
        price_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(price_frame, text="Initial Price ($):").pack(side=tk.LEFT, padx=10)
        self.new_product_price = ctk.CTkEntry(price_frame, width=150)
        self.new_product_price.pack(side=tk.LEFT, padx=10)
        
        # Development time
        time_frame = ctk.CTkFrame(form_frame)
        time_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(time_frame, text="Development Time (months):").pack(side=tk.LEFT, padx=10)
        self.new_product_time = ctk.CTkEntry(time_frame, width=100)
        self.new_product_time.pack(side=tk.LEFT, padx=10)
        self.new_product_time.insert(0, "3")  # Default 3 months
        
        # Create product button
        ctk.CTkButton(
            form_frame,
            text="Start Development",
            command=self.develop_new_product,
            fg_color=PRIMARY_COLOR
        ).pack(pady=20)
        
        # Production tab
        production_frame = ctk.CTkFrame(tabview.tab("Production"))
        production_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            production_frame,
            text="Production Management",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Split into columns
        production_left = ctk.CTkFrame(production_frame)
        production_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        production_right = ctk.CTkFrame(production_frame)
        production_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Production scheduling
        schedule_frame = ctk.CTkFrame(production_left)
        schedule_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            schedule_frame,
            text="Production Schedule",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Production schedule table
        self.production_schedule = ctk.CTkScrollableFrame(schedule_frame)
        self.production_schedule.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Production capacity
        capacity_frame = ctk.CTkFrame(production_right)
        capacity_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            capacity_frame,
            text="Production Capacity",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Create capacity stats
        self.capacity_stats = {}
        self.capacity_stats["Total Factories"] = self.create_stat_row(capacity_frame, "Total Factories", "0")
        self.capacity_stats["Total Capacity"] = self.create_stat_row(capacity_frame, "Total Capacity", "0 units/month")
        self.capacity_stats["Current Utilization"] = self.create_stat_row(capacity_frame, "Current Utilization", "0%")
        
        # Build new factory button
        ctk.CTkButton(
            capacity_frame,
            text="Build New Factory",
            command=self.build_new_factory
        ).pack(pady=10)
        
        # Quality Control tab
        quality_frame = ctk.CTkFrame(tabview.tab("Quality Control"))
        quality_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            quality_frame,
            text="Quality Control",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Quality stats
        quality_stats = ctk.CTkFrame(quality_frame)
        quality_stats.pack(fill=tk.X, padx=10, pady=10)
        
        self.quality_stats = {}
        self.create_stat_row(quality_stats, "Average Product Quality", "0/10")
        self.create_stat_row(quality_stats, "Defect Rate", "0%")
        self.create_stat_row(quality_stats, "Customer Satisfaction", "0/10")
        
        # Quality improvement programs
        programs_frame = ctk.CTkFrame(quality_frame)
        programs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            programs_frame,
            text="Quality Improvement Programs",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # List of programs to implement
        programs = [
            "Six Sigma Certification",
            "Employee Training Program",
            "Automated Quality Testing",
            "Supply Chain Improvements",
            "ISO 9001 Certification"
        ]
        
        for program in programs:
            program_frame = ctk.CTkFrame(programs_frame)
            program_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(program_frame, text=program).pack(side=tk.LEFT, padx=10)
            
            # Random cost between 100K and 1M
            cost = random.randint(100, 1000) * 1000
            
            ctk.CTkLabel(
                program_frame,
                text=f"Cost: ${cost:,}"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                program_frame,
                text="Implement",
                width=100,
                command=lambda p=program, c=cost: self.implement_quality_program(p, c)
            ).pack(side=tk.RIGHT, padx=10)
        
        return frame
    
    def implement_quality_program(self, program, cost):
        print("Coming soon!")
    def create_finances_tab(self):
        """Create content for finances tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Create notebook/tabs for finance sections
        tabview = ctk.CTkTabview(frame)
        tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs
        tabview.add("Overview")
        tabview.add("Cash Management")
        tabview.add("Loans")
        tabview.add("Investments")
        tabview.add("Stock")
        
        # Overview tab
        overview_frame = ctk.CTkFrame(tabview.tab("Overview"))
        overview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split into columns
        left_overview = ctk.CTkFrame(overview_frame)
        left_overview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_overview = ctk.CTkFrame(overview_frame)
        right_overview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Financial summary
        summary_frame = ctk.CTkFrame(left_overview)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            summary_frame,
            text="Financial Summary",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Financial stats
        self.finance_stats = {}
        self.finance_stats["Revenue (Annual)"] = self.create_stat_row(summary_frame, "Revenue (Annual)", "$0M")
        self.finance_stats["Expenses (Annual)"] = self.create_stat_row(summary_frame, "Expenses (Annual)", "$0M")
        self.finance_stats["Profit (Annual)"] = self.create_stat_row(summary_frame, "Profit (Annual)", "$0M")
        self.finance_stats["Profit Margin"] = self.create_stat_row(summary_frame, "Profit Margin", "0%")
        self.finance_stats["Cash"] = self.create_stat_row(summary_frame, "Cash", "$0M")
        self.finance_stats["Assets"] = self.create_stat_row(summary_frame, "Assets", "$0M")
        self.finance_stats["Liabilities"] = self.create_stat_row(summary_frame, "Liabilities", "$0M")
        self.finance_stats["Net Worth"] = self.create_stat_row(summary_frame, "Net Worth", "$0M")
        
        # Profit chart
        profit_frame = ctk.CTkFrame(right_overview)
        profit_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            profit_frame,
            text="Profit History",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        # Create figure for profit chart
        self.profit_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.profit_chart = self.profit_figure.add_subplot(111)
        
        # Add chart to frame
        self.profit_canvas = FigureCanvasTkAgg(self.profit_figure, master=profit_frame)
        self.profit_canvas.draw()
        self.profit_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Income statement
        income_frame = ctk.CTkFrame(left_overview)
        income_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            income_frame,
            text="Income Statement",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)
        
        self.income_text = ctk.CTkTextbox(income_frame, height=200)
        self.income_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Cash Management tab
        cash_frame = ctk.CTkFrame(tabview.tab("Cash Management"))
        cash_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            cash_frame,
            text="Cash Management",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Cash flow
        flow_frame = ctk.CTkFrame(cash_frame)
        flow_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            flow_frame,
            text="Cash Flow",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Cash flow stats
        self.cash_flow_stats = {}
        self.create_stat_row(flow_frame, "Operating Cash Flow", "$0M/month")
        self.create_stat_row(flow_frame, "Investing Cash Flow", "$0M/month")
        self.create_stat_row(flow_frame, "Financing Cash Flow", "$0M/month")
        self.create_stat_row(flow_frame, "Net Cash Flow", "$0M/month")
        self.create_stat_row(flow_frame, "Cash on Hand", "$0M")
        
        # Cash actions
        actions_frame = ctk.CTkFrame(cash_frame)
        actions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            actions_frame,
            text="Cash Actions",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Buttons for cash actions
        buttons_frame = ctk.CTkFrame(actions_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Dividend action
        dividend_frame = ctk.CTkFrame(buttons_frame)
        dividend_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(dividend_frame, text="Issue Dividend:").pack(side=tk.LEFT, padx=10)
        self.dividend_amount = ctk.CTkEntry(dividend_frame, width=100)
        self.dividend_amount.pack(side=tk.LEFT, padx=10)
        self.dividend_amount.insert(0, "0.10")  # Default $0.10 per share
        ctk.CTkLabel(dividend_frame, text="per share").pack(side=tk.LEFT)
        
        ctk.CTkButton(
            dividend_frame,
            text="Issue Dividend",
            command=self.issue_dividend,
            width=100
        ).pack(side=tk.RIGHT, padx=10)
        
        # Share repurchase
        repurchase_frame = ctk.CTkFrame(buttons_frame)
        repurchase_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(repurchase_frame, text="Share Repurchase:").pack(side=tk.LEFT, padx=10)
        self.repurchase_amount = ctk.CTkEntry(repurchase_frame, width=100)
        self.repurchase_amount.pack(side=tk.LEFT, padx=10)
        self.repurchase_amount.insert(0, "1000")  # Default 1000 shares
        ctk.CTkLabel(repurchase_frame, text="shares").pack(side=tk.LEFT)
        
        ctk.CTkButton(
            repurchase_frame,
            text="Repurchase",
            command=self.repurchase_shares,
            width=100
        ).pack(side=tk.RIGHT, padx=10)
        
        # Emergency fundraising
        fundraise_frame = ctk.CTkFrame(buttons_frame)
        fundraise_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(fundraise_frame, text="Emergency Fundraising:").pack(side=tk.LEFT, padx=10)
        
        ctk.CTkButton(
            fundraise_frame,
            text="Issue Bonds",
            command=self.issue_bonds,
            width=100
        ).pack(side=tk.RIGHT, padx=10)
        
        ctk.CTkButton(
            fundraise_frame,
            text="Issue Shares",
            command=self.issue_new_shares,
            width=100
        ).pack(side=tk.RIGHT, padx=10)
        
        # Loans tab
        loans_frame = ctk.CTkFrame(tabview.tab("Loans"))
        loans_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            loans_frame,
            text="Loans Management",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Current loans
        current_loans = ctk.CTkFrame(loans_frame)
        current_loans.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            current_loans,
            text="Current Loans",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Loans table
        self.loans_table = ctk.CTkScrollableFrame(current_loans, height=150)
        self.loans_table.pack(fill=tk.BOTH, padx=10, pady=10)
        
        ctk.CTkLabel(
            self.loans_table,
            text="No active loans"
        ).pack(pady=10)
        
        # New loan application
        new_loan = ctk.CTkFrame(loans_frame)
        new_loan.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            new_loan,
            text="Apply for a New Loan",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Loan amount
        amount_frame = ctk.CTkFrame(new_loan)
        amount_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(amount_frame, text="Loan Amount ($):").pack(side=tk.LEFT, padx=10)
        self.loan_amount = ctk.CTkEntry(amount_frame, width=150)
        self.loan_amount.pack(side=tk.LEFT, padx=10)
        self.loan_amount.insert(0, "1000000")  # Default $1M
        
        # Loan term
        term_frame = ctk.CTkFrame(new_loan)
        term_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(term_frame, text="Loan Term (years):").pack(side=tk.LEFT, padx=10)
        self.loan_term = ctk.CTkOptionMenu(term_frame, values=["1", "3", "5", "10", "20"])
        self.loan_term.pack(side=tk.LEFT, padx=10)
        
        # Apply button
        ctk.CTkButton(
            new_loan,
            text="Apply for Loan",
            command=self.apply_for_loan
        ).pack(pady=10)
        
        # Investments tab
        investments_frame = ctk.CTkFrame(tabview.tab("Investments"))
        investments_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            investments_frame,
            text="Investments",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Current investments
        current_investments = ctk.CTkFrame(investments_frame)
        current_investments.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            current_investments,
            text="Current Investments",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Investments table
        self.investments_table = ctk.CTkScrollableFrame(current_investments, height=200)
        self.investments_table.pack(fill=tk.BOTH, padx=10, pady=10)
        
        ctk.CTkLabel(
            self.investments_table,
            text="No active investments"
        ).pack(pady=10)
        
        # New investment
        new_investment = ctk.CTkFrame(investments_frame)
        new_investment.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            new_investment,
            text="Make a New Investment",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Investment options
        options_frame = ctk.CTkScrollableFrame(new_investment)
        options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        investment_types = [
            {"name": "Research & Development", "return": "8-15%", "risk": "Medium", "cost": "$5M"},
            {"name": "New Factory", "return": "10-12%", "risk": "Low", "cost": "$20M"},
            {"name": "Marketing Campaign", "return": "15-25%", "risk": "High", "cost": "$2M"},
            {"name": "Competitor Acquisition", "return": "20-30%", "risk": "Very High", "cost": "$50M"},
            {"name": "International Expansion", "return": "12-18%", "risk": "Medium-High", "cost": "$15M"}
        ]
        
        for inv in investment_types:
            inv_frame = ctk.CTkFrame(options_frame)
            inv_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(
                inv_frame,
                text=inv["name"],
                font=("Helvetica", 12, "bold")
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                inv_frame,
                text=f"Return: {inv['return']} | Risk: {inv['risk']} | Cost: {inv['cost']}"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                inv_frame,
                text="Invest",
                command=lambda i=inv: self.make_investment(i),
                width=80
            ).pack(side=tk.RIGHT, padx=10)
        
        # Stock tab
        stock_frame = ctk.CTkFrame(tabview.tab("Stock"))
        stock_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            stock_frame,
            text="Stock Management",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Current stock status
        status_frame = ctk.CTkFrame(stock_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Stock status headers
        self.stock_status = {}
        
        if_public_frame = ctk.CTkFrame(status_frame)
        if_public_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_stat_row(if_public_frame, "Public Status", "Private")
        self.create_stat_row(if_public_frame, "Stock Price", "N/A")
        self.create_stat_row(if_public_frame, "Market Cap", "N/A")
        self.create_stat_row(if_public_frame, "Shares Outstanding", "N/A")
        self.create_stat_row(if_public_frame, "P/E Ratio", "N/A")
        self.create_stat_row(if_public_frame, "Dividend Yield", "N/A")
        
        # Stock price chart
        chart_frame = ctk.CTkFrame(stock_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure for stock chart
        self.stock_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.stock_chart = self.stock_figure.add_subplot(111)
        
        # Add chart to frame
        self.stock_canvas = FigureCanvasTkAgg(self.stock_figure, master=chart_frame)
        self.stock_canvas.draw()
        self.stock_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # IPO section
        ipo_frame = ctk.CTkFrame(stock_frame)
        ipo_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            ipo_frame,
            text="Initial Public Offering (IPO)",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # IPO controls
        ipo_controls = ctk.CTkFrame(ipo_frame)
        ipo_controls.pack(fill=tk.X, padx=10, pady=10)
        
        # Stock exchange
        exchange_frame = ctk.CTkFrame(ipo_controls)
        exchange_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(exchange_frame, text="Stock Exchange:").pack(side=tk.LEFT, padx=10)
        self.ipo_exchange = ctk.CTkOptionMenu(
            exchange_frame, 
            values=["NYSE", "NASDAQ", "LSE", "TSE", "HKSE"]
        )
        self.ipo_exchange.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Initial share price
        price_frame = ctk.CTkFrame(ipo_controls)
        price_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(price_frame, text="Initial Share Price ($):").pack(side=tk.LEFT, padx=10)
        self.ipo_price = ctk.CTkEntry(price_frame, width=100)
        self.ipo_price.pack(side=tk.LEFT, padx=10)
        self.ipo_price.insert(0, "20.00")  # Default $20 per share
        
        # Number of shares
        shares_frame = ctk.CTkFrame(ipo_controls)
        shares_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(shares_frame, text="Shares to Issue:").pack(side=tk.LEFT, padx=10)
        self.ipo_shares = ctk.CTkEntry(shares_frame, width=150)
        self.ipo_shares.pack(side=tk.LEFT, padx=10)
        self.ipo_shares.insert(0, "1000000")  # Default 1M shares
        
        # Calculate button
        calc_frame = ctk.CTkFrame(ipo_controls)
        calc_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkButton(
            calc_frame,
            text="Calculate Proceeds",
            command=self.calculate_ipo,
            width=150
        ).pack(side=tk.LEFT, padx=10)
        
        self.ipo_proceeds = ctk.CTkLabel(calc_frame, text="Proceeds: $0")
        self.ipo_proceeds.pack(side=tk.LEFT, padx=10)
        
        # Go public button
        ctk.CTkButton(
            ipo_controls,
            text="Go Public (IPO)",
            command=self.go_public,
            fg_color=PRIMARY_COLOR,
            hover_color="#2980b9"
        ).pack(pady=10)
        
        return frame
    
    def create_operations_tab(self):
        """Create content for operations tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Create notebook/tabs for operations sections
        tabview = ctk.CTkTabview(frame)
        tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs
        tabview.add("Facilities")
        tabview.add("Supply Chain")
        tabview.add("Human Resources")
        tabview.add("Expansion")
        
        # Facilities tab
        facilities_frame = ctk.CTkFrame(tabview.tab("Facilities"))
        facilities_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            facilities_frame,
            text="Facilities Management",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Current facilities
        current_facilities = ctk.CTkFrame(facilities_frame)
        current_facilities.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            current_facilities,
            text="Your Facilities",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Facilities stats
        self.facilities_stats = {}
        self.create_stat_row(current_facilities, "Factories", "0")
        self.create_stat_row(current_facilities, "Offices", "0")
        self.create_stat_row(current_facilities, "Retail Stores", "0")
        self.create_stat_row(current_facilities, "Warehouses", "0")
        self.create_stat_row(current_facilities, "R&D Centers", "0")
        
        # Facilities map/list
        facilities_list = ctk.CTkScrollableFrame(facilities_frame)
        facilities_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.facilities_list = facilities_list
        
        # Facilities actions
        actions_frame = ctk.CTkFrame(facilities_frame)
        actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Build new facility button
        ctk.CTkButton(
            actions_frame,
            text="Build New Facility",
            command=self.build_new_facility
        ).pack(side=tk.LEFT, padx=10)
        
        # Upgrade facility button
        ctk.CTkButton(
            actions_frame,
            text="Upgrade Existing Facility",
            command=self.upgrade_facility
        ).pack(side=tk.LEFT, padx=10)
        
        # Close facility button
        ctk.CTkButton(
            actions_frame,
            text="Close Facility",
            command=self.close_facility,
            fg_color=DANGER_COLOR,
            hover_color="#c0392b"
        ).pack(side=tk.RIGHT, padx=10)
        
        # Supply Chain tab
        supply_frame = ctk.CTkFrame(tabview.tab("Supply Chain"))
        supply_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            supply_frame,
            text="Supply Chain Management",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Supply chain overview
        overview_frame = ctk.CTkFrame(supply_frame)
        overview_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            overview_frame,
            text="Supply Chain Overview",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Supply chain stats
        self.supply_stats = {}
        self.create_stat_row(overview_frame, "Suppliers", "0")
        self.create_stat_row(overview_frame, "Average Lead Time", "0 days")
        self.create_stat_row(overview_frame, "Supply Chain Cost", "$0M/year")
        self.create_stat_row(overview_frame, "Inventory Value", "$0M")
        self.create_stat_row(overview_frame, "Inventory Turnover", "0 times/year")
        
        # Suppliers list
        suppliers_frame = ctk.CTkFrame(supply_frame)
        suppliers_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            suppliers_frame,
            text="Suppliers",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        self.suppliers_list = ctk.CTkScrollableFrame(suppliers_frame)
        self.suppliers_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Supply chain actions
        sc_actions = ctk.CTkFrame(supply_frame)
        sc_actions.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkButton(
            sc_actions,
            text="Add New Supplier",
            command=self.add_supplier
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkButton(
            sc_actions,
            text="Optimize Supply Chain",
            command=self.optimize_supply_chain
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkButton(
            sc_actions,
            text="Manage Inventory",
            command=self.manage_inventory
        ).pack(side=tk.RIGHT, padx=10)
        
        # Human Resources tab
        hr_frame = ctk.CTkFrame(tabview.tab("Human Resources"))
        hr_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            hr_frame,
            text="Human Resources",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # HR overview
        hr_overview = ctk.CTkFrame(hr_frame)
        hr_overview.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            hr_overview,
            text="Workforce Overview",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # HR stats
        self.hr_stats = {}
        self.create_stat_row(hr_overview, "Total Employees", "0")
        self.create_stat_row(hr_overview, "Average Salary", "$0/year")
        self.create_stat_row(hr_overview, "Employee Satisfaction", "0/10")
        self.create_stat_row(hr_overview, "Turnover Rate", "0%")
        self.create_stat_row(hr_overview, "Productivity", "0/10")
        
        # Hiring form
        hiring_frame = ctk.CTkFrame(hr_frame)
        hiring_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            hiring_frame,
            text="Hiring",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Employee count
        count_frame = ctk.CTkFrame(hiring_frame)
        count_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(count_frame, text="Number of Employees:").pack(side=tk.LEFT, padx=10)
        self.hiring_count = ctk.CTkEntry(count_frame, width=100)
        self.hiring_count.pack(side=tk.LEFT, padx=10)
        self.hiring_count.insert(0, "10")
        
        # Employee type
        type_frame = ctk.CTkFrame(hiring_frame)
        type_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(type_frame, text="Employee Type:").pack(side=tk.LEFT, padx=10)
        self.hiring_type = ctk.CTkOptionMenu(
            type_frame,
            values=[
                "Production Worker",
                "Office Staff",
                "Management",
                "Research & Development",
                "Sales & Marketing"
            ]
        )
        self.hiring_type.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Salary
        salary_frame = ctk.CTkFrame(hiring_frame)
        salary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(salary_frame, text="Annual Salary ($):").pack(side=tk.LEFT, padx=10)
        self.hiring_salary = ctk.CTkEntry(salary_frame, width=100)
        self.hiring_salary.pack(side=tk.LEFT, padx=10)
        self.hiring_salary.insert(0, "50000")  # Default $50K
        
        # Hire button
        ctk.CTkButton(
            hiring_frame,
            text="Hire Employees",
            command=self.hire_employees
        ).pack(pady=10)
        
        # Employee benefits
        benefits_frame = ctk.CTkFrame(hr_frame)
        benefits_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            benefits_frame,
            text="Employee Benefits",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        benefits = [
            ("Health Insurance", 2000),
            ("Retirement Plan", 3000),
            ("Paid Time Off", 1500),
            ("Bonuses", 5000),
            ("Education Reimbursement", 2500)
        ]
        
        for benefit, cost in benefits:
            benefit_row = ctk.CTkFrame(benefits_frame)
            benefit_row.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(benefit_row, text=benefit).pack(side=tk.LEFT, padx=10)
            ctk.CTkLabel(benefit_row, text=f"Cost: ${cost}/employee/year").pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                benefit_row,
                text="Implement",
                width=100,
                command=lambda b=benefit, c=cost: self.implement_benefit(b, c)
            ).pack(side=tk.RIGHT, padx=10)
        
        # Expansion tab
        expansion_frame = ctk.CTkFrame(tabview.tab("Expansion"))
        expansion_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            expansion_frame,
            text="Business Expansion",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Current markets
        markets_frame = ctk.CTkFrame(expansion_frame)
        markets_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            markets_frame,
            text="Current Markets",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Markets list
        self.markets_list = ctk.CTkScrollableFrame(markets_frame, height=150)
        self.markets_list.pack(fill=tk.BOTH, padx=10, pady=10)
        
        # Expansion opportunities
        opportunities_frame = ctk.CTkFrame(expansion_frame)
        opportunities_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            opportunities_frame,
            text="Expansion Opportunities",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Country expansion
        country_expansion = ctk.CTkFrame(opportunities_frame)
        country_expansion.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            country_expansion,
            text="International Expansion",
            font=("Helvetica", 12, "bold")
        ).pack(pady=5)
        
        # Country selection
        country_frame = ctk.CTkFrame(country_expansion)
        country_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(country_frame, text="Select Country:").pack(side=tk.LEFT, padx=10)
        self.expansion_country = ctk.CTkOptionMenu(
            country_frame,
            values=[
                "United States", 
                "China", 
                "Germany", 
                "Japan", 
                "United Kingdom",
                "Canada",
                "Australia",
                "Brazil",
                "India",
                "South Korea"
            ]
        )
        self.expansion_country.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Facility counts
        facilities_frame = ctk.CTkFrame(country_expansion)
        facilities_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(facilities_frame, text="Factories:").pack(side=tk.LEFT, padx=10)
        self.expansion_factories = ctk.CTkEntry(facilities_frame, width=60)
        self.expansion_factories.pack(side=tk.LEFT, padx=10)
        self.expansion_factories.insert(0, "1")
        
        ctk.CTkLabel(facilities_frame, text="Stores:").pack(side=tk.LEFT, padx=10)
        self.expansion_stores = ctk.CTkEntry(facilities_frame, width=60)
        self.expansion_stores.pack(side=tk.LEFT, padx=10)
        self.expansion_stores.insert(0, "2")
        
        ctk.CTkLabel(facilities_frame, text="Offices:").pack(side=tk.LEFT, padx=10)
        self.expansion_offices = ctk.CTkEntry(facilities_frame, width=60)
        self.expansion_offices.pack(side=tk.LEFT, padx=10)
        self.expansion_offices.insert(0, "1")
        
        # Investment amount
        investment_frame = ctk.CTkFrame(country_expansion)
        investment_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(investment_frame, text="Investment Amount:").pack(side=tk.LEFT, padx=10)
        
        self.expansion_investment = ctk.CTkLabel(investment_frame, text="$0")
        self.expansion_investment.pack(side=tk.LEFT, padx=10)
        
        # Calculate button
        ctk.CTkButton(
            investment_frame,
            text="Calculate Cost",
            command=self.calculate_expansion_cost,
            width=120
        ).pack(side=tk.RIGHT, padx=10)
        
        # Expand button
        ctk.CTkButton(
            country_expansion,
            text="Expand to Country",
            command=self.expand_to_country,
            fg_color=PRIMARY_COLOR
        ).pack(pady=10)
        
        # Diversification
        diversification = ctk.CTkFrame(opportunities_frame)
        diversification.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            diversification,
            text="Business Diversification",
            font=("Helvetica", 12, "bold")
        ).pack(pady=5)
        
        sectors = [
            ("Technology", "High growth potential, high competition", 50000000),
            ("Retail", "Stable, lower margins", 20000000),
            ("Manufacturing", "Capital intensive, moderate growth", 80000000),
            ("Healthcare", "Regulated, stable growth", 60000000),
            ("Financial Services", "High margins, highly regulated", 40000000)
        ]
        
        for sector, desc, cost in sectors:
            sector_frame = ctk.CTkFrame(diversification)
            sector_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(
                sector_frame,
                text=sector,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                sector_frame,
                text=desc
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                sector_frame,
                text=f"Cost: ${cost/1000000}M"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                sector_frame,
                text="Diversify",
                width=100,
                command=lambda s=sector, c=cost: self.diversify_business(s, c)
            ).pack(side=tk.RIGHT, padx=10)
        
        return frame
    
    def create_marketing_tab(self):
        """Create content for marketing tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Create notebook/tabs for marketing sections
        tabview = ctk.CTkTabview(frame)
        tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs
        tabview.add("Campaigns")
        tabview.add("Market Research")
        tabview.add("Pricing")
        tabview.add("Brand Management")
        
        # Campaigns tab
        campaigns_frame = ctk.CTkFrame(tabview.tab("Campaigns"))
        campaigns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            campaigns_frame,
            text="Marketing Campaigns",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Active campaigns
        active_frame = ctk.CTkFrame(campaigns_frame)
        active_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            active_frame,
            text="Active Campaigns",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        self.campaigns_list = ctk.CTkScrollableFrame(active_frame)
        self.campaigns_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # New campaign
        new_campaign = ctk.CTkFrame(campaigns_frame)
        new_campaign.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            new_campaign,
            text="Launch New Campaign",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Campaign type
        type_frame = ctk.CTkFrame(new_campaign)
        type_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(type_frame, text="Campaign Type:").pack(side=tk.LEFT, padx=10)
        self.campaign_type = ctk.CTkOptionMenu(
            type_frame,
            values=[
                "TV Advertising",
                "Social Media",
                "Print Media",
                "Radio",
                "Billboard",
                "Online Display Ads",
                "Influencer Marketing"
            ]
        )
        self.campaign_type.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Target product
        product_frame = ctk.CTkFrame(new_campaign)
        product_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(product_frame, text="Target Product:").pack(side=tk.LEFT, padx=10)
        self.campaign_product = ctk.CTkOptionMenu(
            product_frame,
            values=["All Products"]  # Will be populated with actual products
        )
        self.campaign_product.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Budget
        budget_frame = ctk.CTkFrame(new_campaign)
        budget_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(budget_frame, text="Budget ($):").pack(side=tk.LEFT, padx=10)
        self.campaign_budget = ctk.CTkEntry(budget_frame, width=150)
        self.campaign_budget.pack(side=tk.LEFT, padx=10)
        self.campaign_budget.insert(0, "1000000")  # Default $1M
        
        # Duration
        duration_frame = ctk.CTkFrame(new_campaign)
        duration_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(duration_frame, text="Duration (months):").pack(side=tk.LEFT, padx=10)
        self.campaign_duration = ctk.CTkEntry(duration_frame, width=100)
        self.campaign_duration.pack(side=tk.LEFT, padx=10)
        self.campaign_duration.insert(0, "3")  # Default 3 months
        
        # Launch button
        ctk.CTkButton(
            new_campaign,
            text="Launch Campaign",
            command=self.launch_campaign,
            fg_color=PRIMARY_COLOR
        ).pack(pady=10)
        
        # Market Research tab
        research_frame = ctk.CTkFrame(tabview.tab("Market Research"))
        research_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            research_frame,
            text="Market Research",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Research insights
        insights_frame = ctk.CTkFrame(research_frame)
        insights_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            insights_frame,
            text="Market Insights",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Market stats
        self.market_stats = {}
        self.create_stat_row(insights_frame, "Market Size", "$0B")
        self.create_stat_row(insights_frame, "Your Market Share", "0%")
        self.create_stat_row(insights_frame, "Market Growth Rate", "0%/year")
        self.create_stat_row(insights_frame, "Brand Recognition", "0%")
        self.create_stat_row(insights_frame, "Customer Satisfaction", "0/10")
        
        # Commission research
        commission_frame = ctk.CTkFrame(research_frame)
        commission_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            commission_frame,
            text="Commission Research",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        research_types = [
            ("Consumer Survey", "Understand customer preferences", 500000),
            ("Competitor Analysis", "Research competitor strategies", 750000),
            ("Market Trends", "Identify emerging market trends", 600000),
            ("Product Testing", "Get feedback on specific products", 450000),
            ("Price Sensitivity", "Determine optimal pricing", 550000)
        ]
        
        for r_type, desc, cost in research_types:
            research_row = ctk.CTkFrame(commission_frame)
            research_row.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(
                research_row,
                text=r_type,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                research_row,
                text=desc
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                research_row,
                text=f"Cost: ${cost:,}"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                research_row,
                text="Commission",
                width=100,
                command=lambda t=r_type, c=cost: self.commission_research(t, c)
            ).pack(side=tk.RIGHT, padx=10)
        
        # Pricing tab
        pricing_frame = ctk.CTkFrame(tabview.tab("Pricing"))
        pricing_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            pricing_frame,
            text="Product Pricing",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Products pricing list
        pricing_list = ctk.CTkFrame(pricing_frame)
        pricing_list.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            pricing_list,
            text="Product Prices",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        self.pricing_list = ctk.CTkScrollableFrame(pricing_list)
        self.pricing_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Price optimization
        optimization_frame = ctk.CTkFrame(pricing_frame)
        optimization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            optimization_frame,
            text="Price Optimization",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Price strategies
        strategies = [
            ("Premium Pricing", "Set prices higher than competition", "High-quality products"),
            ("Value Pricing", "Set competitive prices", "Mass market products"),
            ("Penetration Pricing", "Set below-market prices", "New market entry"),
            ("Skimming", "Start high, then gradually reduce", "Innovative products"),
            ("Loss Leader", "Price below cost to attract customers", "Drive complementary sales")
        ]
        
        for strategy, desc, best_for in strategies:
            strategy_row = ctk.CTkFrame(optimization_frame)
            strategy_row.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(
                strategy_row,
                text=strategy,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                strategy_row,
                text=desc
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                strategy_row,
                text=f"Best for: {best_for}"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                strategy_row,
                text="Apply",
                width=80,
                command=lambda s=strategy: self.apply_pricing_strategy(s)
            ).pack(side=tk.RIGHT, padx=10)
        
        # Brand Management tab
        brand_frame = ctk.CTkFrame(tabview.tab("Brand Management"))
        brand_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            brand_frame,
            text="Brand Management",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Brand stats
        brand_stats = ctk.CTkFrame(brand_frame)
        brand_stats.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            brand_stats,
            text="Brand Statistics",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        self.brand_stats = {}
        self.create_stat_row(brand_stats, "Brand Value", "$0M")
        self.create_stat_row(brand_stats, "Brand Recognition", "0%")
        self.create_stat_row(brand_stats, "Customer Loyalty", "0/10")
        self.create_stat_row(brand_stats, "Public Perception", "Neutral")
        
        # Brand initiatives
        initiatives = ctk.CTkFrame(brand_frame)
        initiatives.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            initiatives,
            text="Brand Initiatives",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # List of brand initiatives
        brand_initiatives = [
            ("Corporate Social Responsibility", "Improve public perception", 2000000),
            ("Brand Refresh", "Update brand image and messaging", 5000000),
            ("Customer Loyalty Program", "Increase repeat business", 1000000),
            ("Sponsorships", "Increase brand visibility", 3000000),
            ("Public Relations Campaign", "Shape public narrative", 1500000)
        ]
        
        for initiative, desc, cost in brand_initiatives:
            initiative_row = ctk.CTkFrame(initiatives)
            initiative_row.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(
                initiative_row,
                text=initiative,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                initiative_row,
                text=desc
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                initiative_row,
                text=f"Cost: ${cost/1000000:.1f}M"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                initiative_row,
                text="Implement",
                width=100,
                command=lambda i=initiative, c=cost: self.implement_brand_initiative(i, c)
            ).pack(side=tk.RIGHT, padx=10)
        
        return frame
    
    def create_research_tab(self):
        """Create content for research tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Create notebook/tabs for research sections
        tabview = ctk.CTkTabview(frame)
        tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs
        tabview.add("R&D Projects")
        tabview.add("Innovation")
        tabview.add("Patents")
        tabview.add("Technology")
        
        # R&D Projects tab
        rd_frame = ctk.CTkFrame(tabview.tab("R&D Projects"))
        rd_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            rd_frame,
            text="Research & Development Projects",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Active projects
        active_frame = ctk.CTkFrame(rd_frame)
        active_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            active_frame,
            text="Active R&D Projects",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        self.rd_projects = ctk.CTkScrollableFrame(active_frame)
        self.rd_projects.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # New R&D project
        new_project = ctk.CTkFrame(rd_frame)
        new_project.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            new_project,
            text="Start New R&D Project",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Project name
        name_frame = ctk.CTkFrame(new_project)
        name_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(name_frame, text="Project Name:").pack(side=tk.LEFT, padx=10)
        self.rd_name = ctk.CTkEntry(name_frame, width=200)
        self.rd_name.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Project type
        type_frame = ctk.CTkFrame(new_project)
        type_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(type_frame, text="Project Type:").pack(side=tk.LEFT, padx=10)
        self.rd_type = ctk.CTkOptionMenu(
            type_frame,
            values=[
                "Product Improvement",
                "New Product Development",
                "Cost Reduction",
                "Manufacturing Process",
                "Materials Research",
                "Software Development"
            ]
        )
        self.rd_type.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Budget
        budget_frame = ctk.CTkFrame(new_project)
        budget_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(budget_frame, text="Budget ($):").pack(side=tk.LEFT, padx=10)
        self.rd_budget = ctk.CTkEntry(budget_frame, width=150)
        self.rd_budget.pack(side=tk.LEFT, padx=10)
        self.rd_budget.insert(0, "5000000")  # Default $5M
        
        # Duration
        duration_frame = ctk.CTkFrame(new_project)
        duration_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(duration_frame, text="Expected Duration (months):").pack(side=tk.LEFT, padx=10)
        self.rd_duration = ctk.CTkEntry(duration_frame, width=100)
        self.rd_duration.pack(side=tk.LEFT, padx=10)
        self.rd_duration.insert(0, "12")  # Default 12 months
        
        # Success probability
        prob_frame = ctk.CTkFrame(new_project)
        prob_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(prob_frame, text="Success Probability (%):").pack(side=tk.LEFT, padx=10)
        self.rd_probability = ctk.CTkEntry(prob_frame, width=100)
        self.rd_probability.pack(side=tk.LEFT, padx=10)
        self.rd_probability.insert(0, "70")  # Default 70%
        
        # Start project button
        ctk.CTkButton(
            new_project,
            text="Start R&D Project",
            command=self.start_rd_project,
            fg_color=PRIMARY_COLOR
        ).pack(pady=10)
        
        # Innovation tab
        innovation_frame = ctk.CTkFrame(tabview.tab("Innovation"))
        innovation_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            innovation_frame,
            text="Innovation Management",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Innovation stats
        stats_frame = ctk.CTkFrame(innovation_frame)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            stats_frame,
            text="Innovation Metrics",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        self.innovation_stats = {}
        self.create_stat_row(stats_frame, "Innovation Index", "0/100")
        self.create_stat_row(stats_frame, "R&D Spending", "$0M/year")
        self.create_stat_row(stats_frame, "R&D as % of Revenue", "0%")
        self.create_stat_row(stats_frame, "New Products (Past Year)", "0")
        self.create_stat_row(stats_frame, "Patents Filed (Past Year)", "0")
        
        # Innovation initiatives
        initiatives_frame = ctk.CTkFrame(innovation_frame)
        initiatives_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            initiatives_frame,
            text="Innovation Initiatives",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        initiatives = [
            ("Innovation Lab", "Create dedicated facility for experimentation", 10000000),
            ("Open Innovation", "Collaborate with startups and academia", 5000000),
            ("Idea Incubator", "Internal program for employee ideas", 2000000),
            ("Innovation Contest", "Hold competition for breakthrough ideas", 1000000),
            ("Research Partnerships", "Partner with research institutions", 7000000)
        ]
        
        for initiative, desc, cost in initiatives:
            initiative_row = ctk.CTkFrame(initiatives_frame)
            initiative_row.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(
                initiative_row,
                text=initiative,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                initiative_row,
                text=desc
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                initiative_row,
                text=f"Cost: ${cost/1000000:.1f}M"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                initiative_row,
                text="Launch",
                width=80,
                command=lambda i=initiative, c=cost: self.launch_innovation_initiative(i, c)
            ).pack(side=tk.RIGHT, padx=10)
        
        # Patents tab
        patents_frame = ctk.CTkFrame(tabview.tab("Patents"))
        patents_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            patents_frame,
            text="Patent Management",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Patent portfolio
        portfolio_frame = ctk.CTkFrame(patents_frame)
        portfolio_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            portfolio_frame,
            text="Patent Portfolio",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        self.patents_list = ctk.CTkScrollableFrame(portfolio_frame)
        self.patents_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File new patent
        file_patent = ctk.CTkFrame(patents_frame)
        file_patent.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            file_patent,
            text="File New Patent",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Patent title
        title_frame = ctk.CTkFrame(file_patent)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(title_frame, text="Patent Title:").pack(side=tk.LEFT, padx=10)
        self.patent_title = ctk.CTkEntry(title_frame, width=250)
        self.patent_title.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Patent category
        category_frame = ctk.CTkFrame(file_patent)
        category_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(category_frame, text="Category:").pack(side=tk.LEFT, padx=10)
        self.patent_category = ctk.CTkOptionMenu(
            category_frame,
            values=[
                "Product Design",
                "Software",
                "Manufacturing Process",
                "Materials",
                "Electronics",
                "Mechanical",
                "Chemical",
                "Biotechnology"
            ]
        )
        self.patent_category.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Filing cost
        cost_frame = ctk.CTkFrame(file_patent)
        cost_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(cost_frame, text="Filing Cost:").pack(side=tk.LEFT, padx=10)
        self.filing_cost = ctk.CTkLabel(cost_frame, text="$50,000")
        self.filing_cost.pack(side=tk.LEFT, padx=10)
        
        # Patent value estimate
        value_frame = ctk.CTkFrame(file_patent)
        value_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(value_frame, text="Estimated Value:").pack(side=tk.LEFT, padx=10)
        self.patent_value = ctk.CTkLabel(value_frame, text="$500,000 - $2,000,000")
        self.patent_value.pack(side=tk.LEFT, padx=10)
        
        # File button
        ctk.CTkButton(
            file_patent,
            text="File Patent Application",
            command=self.file_patent
        ).pack(pady=10)
        
        # License patents section
        license_frame = ctk.CTkFrame(patents_frame)
        license_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            license_frame,
            text="License Patent Technologies",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        license_options = [
            ("Advanced AI Algorithms", "Artificial Intelligence", 2000000, 500000),
            ("Quantum Computing Interface", "Computing", 5000000, 1000000),
            ("Nano-material Manufacturing", "Materials", 3000000, 750000),
            ("Renewable Energy Storage", "Energy", 4000000, 800000),
            ("Advanced Cooling System", "Thermal Management", 1500000, 300000)
        ]
        
        for tech, category, cost, annual in license_options:
            license_row = ctk.CTkFrame(license_frame)
            license_row.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(
                license_row,
                text=tech,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                license_row,
                text=f"Category: {category}"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                license_row,
                text=f"License: ${cost:,} + ${annual:,}/year"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                license_row,
                text="License",
                width=80,
                command=lambda t=tech, c=cost, a=annual: self.license_patent(t, c, a)
            ).pack(side=tk.RIGHT, padx=10)
        
        # Technology tab
        tech_frame = ctk.CTkFrame(tabview.tab("Technology"))
        tech_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            tech_frame,
            text="Technology Management",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Tech stack
        stack_frame = ctk.CTkFrame(tech_frame)
        stack_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(
            stack_frame,
            text="Technology Stack",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        # Tech stats
        self.tech_stats = {}
        self.create_stat_row(stack_frame, "Technology Level", "0/10")
        self.create_stat_row(stack_frame, "IT Infrastructure", "0/10")
        self.create_stat_row(stack_frame, "Automation Level", "0/10")
        self.create_stat_row(stack_frame, "Digital Transformation", "0/10")
        self.create_stat_row(stack_frame, "Tech Investment", "$0M/year")
        
        # Tech upgrades
        upgrades_frame = ctk.CTkFrame(tech_frame)
        upgrades_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            upgrades_frame,
            text="Technology Upgrades",
            font=("Helvetica", 14)
        ).pack(pady=5)
        
        tech_upgrades = [
            ("Enterprise Resource Planning (ERP)", "Streamline operations", 5000000),
            ("Advanced Manufacturing Automation", "Increase production efficiency", 8000000),
            ("Artificial Intelligence Systems", "Improve decision making", 7000000),
            ("Cybersecurity Enhancement", "Protect digital assets", 3000000),
            ("Cloud Infrastructure", "Scalable computing resources", 4000000),
            ("Big Data Analytics", "Better understand markets and operations", 6000000)
        ]
        
        for upgrade, desc, cost in tech_upgrades:
            upgrade_row = ctk.CTkFrame(upgrades_frame)
            upgrade_row.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(
                upgrade_row,
                text=upgrade,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                upgrade_row,
                text=desc
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                upgrade_row,
                text=f"Cost: ${cost/1000000:.1f}M"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                upgrade_row,
                text="Implement",
                width=100,
                command=lambda u=upgrade, c=cost: self.implement_tech_upgrade(u, c)
            ).pack(side=tk.RIGHT, padx=10)
        
        return frame
    
    def create_competition_tab(self):
        """Create content for competition tab"""
        frame = ctk.CTkFrame(self.content_frame)
        
        # Split into columns
        left_col = ctk.CTkFrame(frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_col = ctk.CTkFrame(frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Competitors list in left column
        competitors_frame = ctk.CTkFrame(left_col)
        competitors_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            competitors_frame,
            text="Competitors",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        self.competitors_list = ctk.CTkScrollableFrame(competitors_frame)
        self.competitors_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Market share chart in right column (top)
        market_frame = ctk.CTkFrame(right_col)
        market_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(
            market_frame,
            text="Market Share",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Create figure for market share
        self.market_figure = plt.Figure(figsize=(5, 3.5), dpi=100)
        self.market_chart = self.market_figure.add_subplot(111)
        
        # Add chart to frame
        self.market_canvas = FigureCanvasTkAgg(self.market_figure, master=market_frame)
        self.market_canvas.draw()
        self.market_canvas.get_tk_widget().pack(fill=tk.X, padx=10, pady=10, expand=True)
        
        # Competitive actions available
        actions_frame = ctk.CTkFrame(right_col)
        actions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            actions_frame,
            text="Competitive Actions",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        # Select competitor
        competitor_frame = ctk.CTkFrame(actions_frame)
        competitor_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(competitor_frame, text="Select Competitor:").pack(side=tk.LEFT, padx=10)
        self.selected_competitor = ctk.CTkOptionMenu(
            competitor_frame,
            values=["Select a competitor"]  # Will be populated with actual competitors
        )
        self.selected_competitor.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Competitor actions
        actions_list = ctk.CTkFrame(actions_frame)
        actions_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # List of possible actions
        competitive_actions = [
            ("Price War", "Reduce prices to take market share", "High risk, potential revenue loss"),
            ("Aggressive Marketing", "Outspend competitor on advertising", "Medium risk, guaranteed expense"),
            ("Product Innovation", "Develop superior features", "Low risk, long-term benefit"),
            ("Talent Acquisition", "Hire key employees from competitor", "Medium risk, potential legal issues"),
            ("Market Expansion", "Enter competitor's core markets", "Medium risk, dilutes focus")
        ]
        
        for action, desc, risk in competitive_actions:
            action_row = ctk.CTkFrame(actions_list)
            action_row.pack(fill=tk.X, padx=10, pady=5)
            
            ctk.CTkLabel(
                action_row,
                text=action,
                font=("Helvetica", 12)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                action_row,
                text=desc
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                action_row,
                text=risk,
                text_color=WARNING_COLOR
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                action_row,
                text="Execute",
                width=80,
                command=lambda a=action: self.execute_competitive_action(a)
            ).pack(side=tk.RIGHT, padx=10)
        
        # Acquisition section
        acquisition_frame = ctk.CTkFrame(left_col)
        acquisition_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ctk.CTkLabel(
            acquisition_frame,
            text="Acquisitions",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        ctk.CTkLabel(
            acquisition_frame,
            text="Acquire smaller competitors to grow market share"
        ).pack()
        
        # Acquisition button
        ctk.CTkButton(
            acquisition_frame,
            text="Identify Acquisition Targets",
            command=self.show_acquisition_targets,
            fg_color=PRIMARY_COLOR,
            hover_color="#2980b9"
        ).pack(pady=10)
        
        return frame
    
    def update_display(self):
        """Update the UI with current data"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Update business name in info bar
        self.business_label.configure(text=business.name)
        
        # Format financial values
        cash_display = self.format_currency(business.cash)
        self.cash_label.configure(text=cash_display)
        
        # Get latest revenue
        if business.revenue_history:
            latest_date = max(business.revenue_history.keys())
            latest_revenue = business.revenue_history[latest_date]
            revenue_display = self.format_currency(latest_revenue * 4)  # Annualized from quarterly
            self.revenue_label.configure(text=revenue_display)
        else:
            self.revenue_label.configure(text="$0/yr")
        
        # Employees
        self.employees_label.configure(text=str(business.employees))
        
        # Profit margin
        margin = business.profit_margin * 100
        self.profit_margin_label.configure(text=f"{margin:.1f}%")
        
        # If overview tab is active, update detailed stats
        if self.current_tab == "overview" and hasattr(self, "stat_labels"):
            self.stat_labels["Cash"].configure(text=cash_display)
            
            # Quarterly revenue and profit
            if business.revenue_history:
                latest_date = max(business.revenue_history.keys())
                quarterly_revenue = business.revenue_history[latest_date]
                self.stat_labels["Revenue (Quarter)"].configure(text=self.format_currency(quarterly_revenue))
                
                quarterly_profit = business.profit_history.get(latest_date, 0)
                self.stat_labels["Profit (Quarter)"].configure(text=self.format_currency(quarterly_profit))
                
                self.stat_labels["Profit Margin"].configure(text=f"{margin:.1f}%")
            
            # Assets and liabilities
            self.stat_labels["Assets"].configure(text=self.format_currency(business.total_assets))
            self.stat_labels["Liabilities"].configure(text=self.format_currency(business.total_liabilities))
            self.stat_labels["Net Worth"].configure(text=self.format_currency(business.net_worth))
            
            # Employees and brand value
            self.stat_labels["Employees"].configure(text=str(business.employees))
            self.stat_labels["Brand Value"].configure(text=f"{business.brand_value:.1f}")
            
            # Public company stats if applicable
            if business.is_public:
                self.stat_labels["Stock Price"].configure(text=f"${business.share_price:.2f}")
                self.stat_labels["Market Cap"].configure(text=self.format_currency(business.market_cap))
                
                # Calculate P/E ratio
                if business.profit_history:
                    latest_date = max(business.profit_history.keys())
                    annual_profit = business.profit_history[latest_date] * 4  # Annualized from quarterly
                    if annual_profit > 0:
                        pe_ratio = business.market_cap / annual_profit
                        self.stat_labels["P/E Ratio"].configure(text=f"{pe_ratio:.1f}")
                    else:
                        self.stat_labels["P/E Ratio"].configure(text="N/A")
            
            # Update revenue & profit chart
            self.update_revenue_chart(business)
            
            # Update top products list
            self.update_top_products(business)
            
            # Update notifications
            self.update_notifications(business)
        
        # Update products tab if active
        if self.current_tab == "products" and hasattr(self, "products_list"):
            self.update_products_list(business)
            
            # Update production capacity stats
            if hasattr(self, "capacity_stats"):
                total_factories = sum(business.factories.values())
                self.capacity_stats["Total Factories"].configure(text=str(total_factories))
                
                # Rough capacity estimate based on factories
                capacity = total_factories * 10000
                self.capacity_stats["Total Capacity"].configure(text=f"{capacity:,} units/month")
                
                # Utilization based on current production vs capacity
                total_production = sum(p.units_produced for p in business.products)
                if capacity > 0:
                    utilization = (total_production / capacity) * 100
                    self.capacity_stats["Current Utilization"].configure(text=f"{utilization:.1f}%")
                else:
                    self.capacity_stats["Current Utilization"].configure(text="N/A")
        
        # Update finances tab if active
        if self.current_tab == "finances":
            # Update finance stats
            if hasattr(self, "finance_stats"):
                # Annual figures (estimate from latest quarter x4)
                if business.revenue_history:
                    latest_date = max(business.revenue_history.keys())
                    annual_revenue = business.revenue_history[latest_date] * 4
                    self.finance_stats["Revenue (Annual)"].configure(text=self.format_currency(annual_revenue))
                    
                    # Rough estimate of expenses
                    if latest_date in business.profit_history:
                        annual_profit = business.profit_history[latest_date] * 4
                        annual_expenses = annual_revenue - annual_profit
                        self.finance_stats["Expenses (Annual)"].configure(text=self.format_currency(annual_expenses))
                        self.finance_stats["Profit (Annual)"].configure(text=self.format_currency(annual_profit))
                        
                        # Profit margin
                        if annual_revenue > 0:
                            profit_margin = (annual_profit / annual_revenue) * 100
                            self.finance_stats["Profit Margin"].configure(text=f"{profit_margin:.1f}%")
                
                # Balance sheet items
                self.finance_stats["Cash"].configure(text=self.format_currency(business.cash))
                self.finance_stats["Assets"].configure(text=self.format_currency(business.total_assets))
                self.finance_stats["Liabilities"].configure(text=self.format_currency(business.total_liabilities))
                self.finance_stats["Net Worth"].configure(text=self.format_currency(business.net_worth))
            
            # Update profit chart
            if hasattr(self, "profit_chart"):
                self.update_profit_chart(business)
            
            # Update income statement
            if hasattr(self, "income_text"):
                self.update_income_statement(business)
            
            # Update stock information if public
            if hasattr(self, "stock_status"):
                if business.is_public:
                    self.stock_status["Public Status"].configure(text="Public")
                    self.stock_status["Stock Price"].configure(text=f"${business.share_price:.2f}")
                    self.stock_status["Market Cap"].configure(text=self.format_currency(business.market_cap))
                    self.stock_status["Shares Outstanding"].configure(text=f"{business.shares_outstanding:,}")
                    
                    # Calculate P/E ratio
                    if business.profit_history:
                        latest_date = max(business.profit_history.keys())
                        annual_profit = business.profit_history[latest_date] * 4  # Annualized from quarterly
                        if annual_profit > 0:
                            pe_ratio = business.market_cap / annual_profit
                            self.stock_status["P/E Ratio"].configure(text=f"{pe_ratio:.1f}")
                        else:
                            self.stock_status["P/E Ratio"].configure(text="N/A")
                            
                    # No dividend yet
                    self.stock_status["Dividend Yield"].configure(text="0.0%")
                else:
                    self.stock_status["Public Status"].configure(text="Private")
                    self.stock_status["Stock Price"].configure(text="N/A")
                    self.stock_status["Market Cap"].configure(text="N/A")
                    self.stock_status["Shares Outstanding"].configure(text="N/A")
                    self.stock_status["P/E Ratio"].configure(text="N/A")
                    self.stock_status["Dividend Yield"].configure(text="N/A")
                
                # Update stock price chart if public
                if business.is_public:
                    self.update_stock_chart(business)
    
    def update_revenue_chart(self, business):
        """Update the revenue and profit chart with historical data"""
        if not hasattr(self, "revenue_chart"):
            return
            
        # Clear previous plot
        self.revenue_chart.clear()
        
        # Get revenue and profit history
        if business.revenue_history and business.profit_history:
            # Extract data
            dates = sorted(business.revenue_history.keys())
            revenues = [business.revenue_history[date] for date in dates]
            profits = [business.profit_history.get(date, 0) for date in dates]
            
            # Convert dates to datetime objects for plotting
            date_objects = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
            
            # Create the plot
            self.revenue_chart.plot(date_objects, revenues, marker='o', linestyle='-', color=PRIMARY_COLOR, label='Revenue')
            self.revenue_chart.plot(date_objects, profits, marker='s', linestyle='-', color=SUCCESS_COLOR, label='Profit')
            
            # Add zero line to highlight profit/loss
            self.revenue_chart.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Format the plot
            self.revenue_chart.set_title("Quarterly Performance")
            self.revenue_chart.set_xlabel("Quarter")
            
            # Format y-axis with million/billion labels
            max_value = max(max(revenues), abs(min(profits)) if profits else 0)
            if max_value >= 1e9:
                self.revenue_chart.set_ylabel("Amount (Billions $)")
                self.revenue_chart.yaxis.set_major_formatter(lambda x, pos: f"${x/1e9:.1f}B")
            elif max_value >= 1e6:
                self.revenue_chart.set_ylabel("Amount (Millions $)")
                self.revenue_chart.yaxis.set_major_formatter(lambda x, pos: f"${x/1e6:.1f}M")
            else:
                self.revenue_chart.set_ylabel("Amount ($)")
                self.revenue_chart.yaxis.set_major_formatter(lambda x, pos: f"${x:.0f}")
            
            # Format x-axis dates
            self.revenue_chart.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            self.revenue_chart.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(self.revenue_chart.xaxis.get_majorticklabels(), rotation=45)
            
            # Add legend
            self.revenue_chart.legend()
            
            # Add grid
            self.revenue_chart.grid(True, linestyle='--', alpha=0.7)
        else:
            # No data yet
            self.revenue_chart.text(0.5, 0.5, "No financial history data available yet", 
                                   horizontalalignment='center', verticalalignment='center')
        
        # Update the canvas
        self.revenue_figure.tight_layout()
        self.revenue_canvas.draw()
    
    def update_top_products(self, business):
        """Update the top products display"""
        if not hasattr(self, "top_products_frame"):
            return
            
        # Clear existing products
        for widget in self.top_products_frame.winfo_children():
            widget.destroy()
        
        if not business.products:
            # No products message
            ctk.CTkLabel(
                self.top_products_frame,
                text="No products yet. Develop some products!"
            ).pack(pady=10)
            return
            
        # Sort products by revenue (units sold * price)
        sorted_products = sorted(
            business.products, 
            key=lambda p: p.units_sold * p.price,
            reverse=True
        )
        
        # Display top 5 products
        for i, product in enumerate(sorted_products[:5]):
            product_frame = ctk.CTkFrame(self.top_products_frame)
            product_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Product name and category
            ctk.CTkLabel(
                product_frame,
                text=f"{i+1}. {product.name}",
                font=("Helvetica", 12, "bold")
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                product_frame,
                text=f"({product.category})"
            ).pack(side=tk.LEFT)
            
            # Sales data
            revenue = product.units_sold * product.price
            margin = ((product.price - product.cost) / product.price) * 100 if product.price > 0 else 0
            
            ctk.CTkLabel(
                product_frame,
                text=f"Revenue: {self.format_currency(revenue)} | "
                     f"Margin: {margin:.1f}%"
            ).pack(side=tk.RIGHT, padx=10)
    
    def update_notifications(self, business):
        """Update the notifications display"""
        if not hasattr(self, "notifications_text"):
            return
            
        # Enable editing
        self.notifications_text.configure(state=tk.NORMAL)
        
        # Clear existing text
        self.notifications_text.delete("1.0", tk.END)
        
        # Example notifications - in a real game, these would be generated based on events
        current_date = self.app.game_state.current_date.strftime("%Y-%m-%d")
        
        notifications = [
            f"[{current_date}] Current cash position: {self.format_currency(business.cash)}",
            f"[{current_date}] You have {len(business.products)} products in your portfolio."
        ]
        
        # Add notifications based on financial health
        if business.cash < 1000000:
            notifications.append(f"[{current_date}] WARNING: Cash reserves are critically low!")
        
        if business.profit_history:
            latest_date = max(business.profit_history.keys())
            latest_profit = business.profit_history[latest_date]
            
            if latest_profit < 0:
                notifications.append(f"[{current_date}] ALERT: Your business is operating at a loss!")
            elif latest_profit > 1000000:
                notifications.append(f"[{current_date}] GOOD NEWS: Profits are strong this quarter!")
        
        # Market notifications
        notifications.append(f"[{current_date}] Market research shows growing demand in your sector.")
        
        # Add all notifications to the text box
        for note in notifications:
            self.notifications_text.insert(tk.END, note + "\n\n")
        
        # Disable editing
        self.notifications_text.configure(state=tk.DISABLED)
    
    def update_products_list(self, business):
        """Update the product list display"""
        # Clear existing products
        for widget in self.products_list.winfo_children():
            widget.destroy()
        
        if not business.products:
            ctk.CTkLabel(
                self.products_list,
                text="No products yet. Develop some products!"
            ).pack(pady=10)
            return
            
        # Add each product
        for product in business.products:
            product_frame = ctk.CTkFrame(self.products_list)
            product_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Product header
            header = ctk.CTkFrame(product_frame)
            header.pack(fill=tk.X, padx=5, pady=2)
            
            ctk.CTkLabel(
                header,
                text=product.name,
                font=("Helvetica", 14, "bold")
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                header,
                text=f"Category: {product.category}"
            ).pack(side=tk.LEFT, padx=10)
            
            # Product details
            details = ctk.CTkFrame(product_frame)
            details.pack(fill=tk.X, padx=5, pady=2)
            
            # Left column of details
            left_details = ctk.CTkFrame(details)
            left_details.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, expand=True)
            
            ctk.CTkLabel(
                left_details,
                text=f"Price: ${product.price:,.2f}"
            ).pack(anchor="w", padx=10, pady=2)
            
            ctk.CTkLabel(
                left_details,
                text=f"Cost: ${product.cost:,.2f}"
            ).pack(anchor="w", padx=10, pady=2)
            
            margin = ((product.price - product.cost) / product.price) * 100 if product.price > 0 else 0
            ctk.CTkLabel(
                left_details,
                text=f"Margin: {margin:.1f}%"
            ).pack(anchor="w", padx=10, pady=2)
            
            # Right column of details
            right_details = ctk.CTkFrame(details)
            right_details.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, expand=True)
            
            ctk.CTkLabel(
                right_details,
                text=f"Quality: {product.quality * 10:.1f}/10"
            ).pack(anchor="w", padx=10, pady=2)
            
            ctk.CTkLabel(
                right_details,
                text=f"Brand Awareness: {product.brand_awareness * 100:.1f}%"
            ).pack(anchor="w", padx=10, pady=2)
            
            ctk.CTkLabel(
                right_details,
                text=f"In Stock: {product.units_produced:,} units"
            ).pack(anchor="w", padx=10, pady=2)
            
            # Product actions
            actions = ctk.CTkFrame(product_frame)
            actions.pack(fill=tk.X, padx=5, pady=5)
            
            # Pricing controls
            price_frame = ctk.CTkFrame(actions)
            price_frame.pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(price_frame, text="New Price:").pack(side=tk.LEFT, padx=5)
            price_entry = ctk.CTkEntry(price_frame, width=100)
            price_entry.pack(side=tk.LEFT, padx=5)
            price_entry.insert(0, str(product.price))
            
            ctk.CTkButton(
                price_frame,
                text="Update",
                width=80,
                command=lambda p=product, e=price_entry: self.update_product_price(p, e)
            ).pack(side=tk.LEFT, padx=5)
            
            # Production controls
            production_frame = ctk.CTkFrame(actions)
            production_frame.pack(side=tk.RIGHT, padx=10)
            
            ctk.CTkLabel(production_frame, text="Produce:").pack(side=tk.LEFT, padx=5)
            production_entry = ctk.CTkEntry(production_frame, width=100)
            production_entry.pack(side=tk.LEFT, padx=5)
            production_entry.insert(0, "1000")
            
            ctk.CTkButton(
                production_frame,
                text="Produce",
                width=80,
                command=lambda p=product, e=production_entry: self.produce_units(p, e)
            ).pack(side=tk.LEFT, padx=5)
            
            # Improvement buttons
            improve_frame = ctk.CTkFrame(product_frame)
            improve_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkButton(
                improve_frame,
                text="Improve Quality",
                width=150,
                command=lambda p=product: self.improve_product_quality(p)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                improve_frame,
                text="Marketing Push",
                width=150,
                command=lambda p=product: self.market_product(p)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                improve_frame,
                text="Reduce Costs",
                width=150,
                command=lambda p=product: self.reduce_product_costs(p)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                improve_frame,
                text="Discontinue",
                width=100,
                fg_color=DANGER_COLOR,
                hover_color="#c0392b",
                command=lambda p=product: self.discontinue_product(p)
            ).pack(side=tk.RIGHT, padx=10)
        
        # When we're in production schedule tab, update the scheduler
        if hasattr(self, "production_schedule"):
            self.update_production_schedule(business)
    
    def update_production_schedule(self, business):
        """Update the production scheduling interface"""
        # Clear existing items
        for widget in self.production_schedule.winfo_children():
            widget.destroy()
        
        if not business.products:
            ctk.CTkLabel(
                self.production_schedule,
                text="No products to schedule"
            ).pack(pady=10)
            return
            
        # Header
        header = ctk.CTkFrame(self.production_schedule)
        header.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(
            header,
            text="Product",
            font=("Helvetica", 12, "bold"),
            width=150
        ).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkLabel(
            header,
            text="Current Stock",
            font=("Helvetica", 12, "bold"),
            width=100
        ).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkLabel(
            header,
            text="Monthly Production",
            font=("Helvetica", 12, "bold"),
            width=150
        ).pack(side=tk.LEFT, padx=5)
        
        # For each product
        for product in business.products:
            product_row = ctk.CTkFrame(self.production_schedule)
            product_row.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkLabel(
                product_row,
                text=product.name,
                width=150
            ).pack(side=tk.LEFT, padx=5)
            
            ctk.CTkLabel(
                product_row,
                text=f"{product.units_produced:,}",
                width=100
            ).pack(side=tk.LEFT, padx=5)
            
            # Production amount entry
            production_entry = ctk.CTkEntry(product_row, width=100)
            production_entry.pack(side=tk.LEFT, padx=5)
            production_entry.insert(0, "1000")  # Default 1000 units
            
            ctk.CTkButton(
                product_row,
                text="Set Monthly",
                width=100,
                command=lambda p=product, e=production_entry: self.set_monthly_production(p, e)
            ).pack(side=tk.LEFT, padx=5)
    
    def update_profit_chart(self, business):
        """Update the profit history chart"""
        if not hasattr(self, "profit_chart"):
            return
            
        # Clear previous plot
        self.profit_chart.clear()
        
        # Get profit history
        if business.profit_history:
            # Extract data
            dates = sorted(business.profit_history.keys())
            profits = [business.profit_history[date] for date in dates]
            
            # Convert dates to datetime objects for plotting
            date_objects = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
            
            # Create the plot
            self.profit_chart.plot(date_objects, profits, marker='o', linestyle='-', color=SUCCESS_COLOR)
            
            # Add zero line to highlight profit/loss
            self.profit_chart.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Format the plot
            self.profit_chart.set_title("Profit History")
            self.profit_chart.set_xlabel("Quarter")
            
            # Format y-axis with million/billion labels
            max_profit = max(profits) if profits else 0
            min_profit = min(profits) if profits else 0
            max_abs = max(abs(max_profit), abs(min_profit))
            
            if max_abs >= 1e9:
                self.profit_chart.set_ylabel("Profit (Billions $)")
                self.profit_chart.yaxis.set_major_formatter(lambda x, pos: f"${x/1e9:.1f}B")
            elif max_abs >= 1e6:
                self.profit_chart.set_ylabel("Profit (Millions $)")
                self.profit_chart.yaxis.set_major_formatter(lambda x, pos: f"${x/1e6:.1f}M")
            else:
                self.profit_chart.set_ylabel("Profit ($)")
                self.profit_chart.yaxis.set_major_formatter(lambda x, pos: f"${x:.0f}")
            
            # Format x-axis dates
            self.profit_chart.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            self.profit_chart.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(self.profit_chart.xaxis.get_majorticklabels(), rotation=45)
            
            # Add grid
            self.profit_chart.grid(True, linestyle='--', alpha=0.7)
        else:
            # No data yet
            self.profit_chart.text(0.5, 0.5, "No profit history data available yet", 
                                  horizontalalignment='center', verticalalignment='center')
        
        # Update the canvas
        self.profit_figure.tight_layout()
        self.profit_canvas.draw()
    
    def update_income_statement(self, business):
        """Update the income statement text display"""
        if not hasattr(self, "income_text"):
            return
            
        # Enable editing
        self.income_text.configure(state=tk.NORMAL)
        
        # Clear existing text
        self.income_text.delete("1.0", tk.END)
        
        # Create income statement
        if business.revenue_history:
            latest_date = max(business.revenue_history.keys())
            quarterly_revenue = business.revenue_history[latest_date]
            quarterly_profit = business.profit_history.get(latest_date, 0)
            
            # Calculate expenses (simplified)
            quarterly_expenses = quarterly_revenue - quarterly_profit
            
            # Format the statement
            statement = f"QUARTERLY INCOME STATEMENT\n"
            statement += f"Period ending: {latest_date}\n\n"
            statement += f"Revenue:                  {self.format_currency(quarterly_revenue)}\n"
            statement += f"Cost of Goods Sold:       {self.format_currency(quarterly_expenses * 0.6)}\n"
            statement += f"Gross Profit:             {self.format_currency(quarterly_revenue - quarterly_expenses * 0.6)}\n\n"
            statement += f"Operating Expenses:\n"
            statement += f"  R&D:                    {self.format_currency(quarterly_expenses * 0.1)}\n"
            statement += f"  Sales & Marketing:      {self.format_currency(quarterly_expenses * 0.15)}\n"
            statement += f"  General & Admin:        {self.format_currency(quarterly_expenses * 0.15)}\n"
            statement += f"Total Operating Expenses: {self.format_currency(quarterly_expenses * 0.4)}\n\n"
            statement += f"Operating Income:         {self.format_currency(quarterly_profit)}\n"
            statement += f"Interest Expense:         {self.format_currency(0)}\n"
            statement += f"Income Before Tax:        {self.format_currency(quarterly_profit)}\n"
            statement += f"Income Tax:               {self.format_currency(quarterly_profit * 0.21)}\n"
            statement += f"Net Income:               {self.format_currency(quarterly_profit * 0.79)}\n\n"
            
            profit_margin = (quarterly_profit / quarterly_revenue) * 100 if quarterly_revenue > 0 else 0
            statement += f"Profit Margin:            {profit_margin:.1f}%\n"
        else:
            statement = "No financial data available yet."
        
        # Insert the statement
        self.income_text.insert(tk.END, statement)
        
        # Disable editing
        self.income_text.configure(state=tk.DISABLED)
    
    def update_stock_chart(self, business):
        """Update the stock price history chart"""
        if not hasattr(self, "stock_chart") or not business.is_public:
            return
            
        # Clear previous plot
        self.stock_chart.clear()
        
        # Create simulated stock price history (in a real game, this would be stored)
        # For now, we'll generate data based on the share_price and some randomness
        current_price = business.share_price
        days = 90  # Show 90 days of history
        
        # Generate dates
        end_date = self.app.game_state.current_date
        dates = [(end_date - datetime.timedelta(days=i)) for i in range(days, 0, -1)]
        dates.append(end_date)
        
        # Generate prices with some randomness
        # Start from a slightly different price and gradually move toward the current price
        start_price = current_price * (0.8 + 0.4 * random.random())  # Start between 80% and 120% of current price
        
        # Generate daily fluctuations that work toward the final price
        prices = []
        for i in range(len(dates) - 1):
            progress = i / (len(dates) - 1)  # 0 to 1
            target = start_price * (1 - progress) + current_price * progress
            daily_fluctuation = 0.01 * random.uniform(-1, 1)  # +/- 1% random daily change
            prices.append(target * (1 + daily_fluctuation))
        
        # Add current price
        prices.append(current_price)
        
        # Plot the data
        self.stock_chart.plot(dates, prices, marker='', linestyle='-', color=PRIMARY_COLOR)
        
        # Format the plot
        self.stock_chart.set_title("Stock Price History")
        self.stock_chart.set_xlabel("Date")
        self.stock_chart.set_ylabel("Price ($)")
        
        # Format x-axis dates
        self.stock_chart.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        self.stock_chart.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(self.stock_chart.xaxis.get_majorticklabels(), rotation=45)
        
        # Format y-axis with dollar values
        self.stock_chart.yaxis.set_major_formatter(lambda x, pos: f"${x:.2f}")
        
        # Add grid
        self.stock_chart.grid(True, linestyle='--', alpha=0.7)
        
        # Update the canvas
        self.stock_figure.tight_layout()
        self.stock_canvas.draw()
    
    def develop_new_product(self):
        """Start development of a new product"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Get values from form
        name = self.new_product_name.get().strip()
        category = self.new_product_category.get().strip()
        
        try:
            investment = float(self.new_product_investment.get())
            if investment <= 0:
                raise ValueError("Investment must be positive")
                
            price = float(self.new_product_price.get()) if self.new_product_price.get() else 0
            
            development_time = int(self.new_product_time.get())
            if development_time <= 0:
                raise ValueError("Development time must be positive")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return
        
        # Check if business has enough cash
        if investment > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"You need ${investment:,.2f} but only have {self.format_currency(business.cash)}")
            return
        
        # Validation
        if not name:
            messagebox.showerror("Input Error", "Product name is required")
            return
            
        if not category:
            messagebox.showerror("Input Error", "Product category is required")
            return
        
        # In a real game, you would create a pending product that becomes available after the development time
        # For now, we'll create it immediately
        product = business.develop_new_product(name, category, investment)
        
        if product:
            # Set the initial price if provided
            if price > 0:
                product.set_price(price)
            
            messagebox.showinfo("Product Development", 
                             f"Successfully started development of {name}. "
                             f"It will be available in {development_time} months.")
            
            # Update the UI
            self.update_display()
            
            # Clear form fields
            self.new_product_name.delete(0, tk.END)
            self.new_product_category.delete(0, tk.END)
            self.new_product_price.delete(0, tk.END)
            
            # Update product dropdown in marketing campaign form if it exists
            if hasattr(self, "campaign_product"):
                products = ["All Products"] + [p.name for p in business.products]
                self.campaign_product.configure(values=products)
        else:
            messagebox.showerror("Error", "Failed to start product development")
    
    def update_product_price(self, product, price_entry):
        """Update the price of a product"""
        try:
            new_price = float(price_entry.get())
            if new_price < 0:
                raise ValueError("Price cannot be negative")
                
            product.set_price(new_price)
            messagebox.showinfo("Price Updated", f"{product.name} price updated to ${new_price:,.2f}")
            
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def produce_units(self, product, units_entry):
        """Produce units of a product"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        try:
            units = int(units_entry.get())
            if units <= 0:
                raise ValueError("Units must be positive")
                
            # Check if we have production capacity
            total_factories = sum(business.factories.values())
            if total_factories == 0:
                messagebox.showerror("No Factories", 
                                   "You need to build factories before you can produce units")
                return
                
            # Check if we have enough cash for production costs
            production_cost = units * product.cost
            if production_cost > business.cash:
                messagebox.showerror("Insufficient Funds", 
                                   f"Production requires {self.format_currency(production_cost)} "
                                   f"but you only have {self.format_currency(business.cash)}")
                return
                
            # Produce the units
            business.cash -= production_cost
            product.produce(units)
            
            messagebox.showinfo("Production", 
                             f"Successfully produced {units:,} units of {product.name} "
                             f"at a cost of {self.format_currency(production_cost)}")
            
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def improve_product_quality(self, product):
        """Invest in product quality improvement"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Calculate cost based on current quality (higher quality = more expensive to improve)
        base_cost = 500000  # $500K base cost
        quality_multiplier = 1 + (product.quality * 5)  # Higher quality means higher cost
        improvement_cost = base_cost * quality_multiplier
        
        # Ask for confirmation
        result = messagebox.askyesno("Improve Quality", 
                                    f"Improve quality of {product.name}?\n\n"
                                    f"Current quality: {product.quality * 10:.1f}/10\n"
                                    f"Cost: {self.format_currency(improvement_cost)}\n\n"
                                    f"This will increase quality and allow you to charge higher prices.")
        
        if not result:
            return
            
        # Check if business has enough cash
        if improvement_cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"Quality improvement requires {self.format_currency(improvement_cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Apply the improvement
        business.cash -= improvement_cost
        
        # Calculate quality increase (diminishing returns at higher quality)
        increase = 0.05 * (1 - product.quality * 0.5)  # Higher quality = smaller improvement
        product.quality = min(1.0, product.quality + increase)
        
        messagebox.showinfo("Quality Improved", 
                         f"Successfully improved {product.name} quality to {product.quality * 10:.1f}/10")
        
        self.update_display()
    
    def market_product(self, product):
        """Invest in product marketing"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Calculate cost
        base_cost = 200000  # $200K base cost
        awareness_multiplier = 1 + (product.brand_awareness * 2)  # Higher awareness = more expensive
        marketing_cost = base_cost * awareness_multiplier
        
        # Ask for confirmation
        result = messagebox.askyesno("Marketing Campaign", 
                                    f"Launch marketing campaign for {product.name}?\n\n"
                                    f"Current brand awareness: {product.brand_awareness * 100:.1f}%\n"
                                    f"Cost: {self.format_currency(marketing_cost)}\n\n"
                                    f"This will increase brand awareness and sales potential.")
        
        if not result:
            return
            
        # Check if business has enough cash
        if marketing_cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"Marketing campaign requires {self.format_currency(marketing_cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Apply the marketing
        business.cash -= marketing_cost
        product.invest_in_marketing(marketing_cost)
        
        messagebox.showinfo("Marketing Campaign", 
                         f"Successfully launched marketing campaign for {product.name}. "
                         f"Brand awareness increased to {product.brand_awareness * 100:.1f}%")
        
        self.update_display()
    
    def reduce_product_costs(self, product):
        """Invest in cost reduction for a product"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Calculate the investment cost
        investment = product.cost * 5000  # Arbitrary formula
        
        # Ask for confirmation
        result = messagebox.askyesno("Reduce Costs", 
                                    f"Invest in cost reduction for {product.name}?\n\n"
                                    f"Current cost: ${product.cost:,.2f} per unit\n"
                                    f"Investment: {self.format_currency(investment)}\n\n"
                                    f"This will reduce production costs and increase profit margins.")
        
        if not result:
            return
            
        # Check if business has enough cash
        if investment > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"Cost reduction requires {self.format_currency(investment)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Apply the investment
        business.cash -= investment
        
        # Calculate cost reduction (diminishing returns for already low costs)
        min_cost = product.cost * 0.2  # Can't reduce below 20% of original cost
        reduction_percent = 0.1 * (1 - (min_cost / product.cost))  # Smaller reduction as we approach min cost
        new_cost = max(min_cost, product.cost * (1 - reduction_percent))
        
        product.cost = new_cost
        
        messagebox.showinfo("Cost Reduced", 
                         f"Successfully reduced production costs for {product.name} "
                         f"to ${product.cost:,.2f} per unit")
        
        self.update_display()
    
    def discontinue_product(self, product):
        """Discontinue a product"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Ask for confirmation
        result = messagebox.askyesno("Discontinue Product", 
                                    f"Are you sure you want to discontinue {product.name}?\n\n"
                                    f"This action cannot be undone. Any unsold inventory "
                                    f"({product.units_produced:,} units) will be written off.")
        
        if not result:
            return
            
        # Remove the product
        business.products = [p for p in business.products if p.id != product.id]
        
        messagebox.showinfo("Product Discontinued", 
                         f"{product.name} has been discontinued.")
        
        self.update_display()
        
        # Update product dropdown in marketing campaign form if it exists
        if hasattr(self, "campaign_product"):
            products = ["All Products"] + [p.name for p in business.products]
            self.campaign_product.configure(values=products)
    
    def set_monthly_production(self, product, units_entry):
        """Set the monthly production schedule for a product"""
        try:
            units = int(units_entry.get())
            if units < 0:
                raise ValueError("Units must be non-negative")
                
            # In a real game, this would set up automated production
            messagebox.showinfo("Production Schedule", 
                             f"Monthly production for {product.name} set to {units:,} units")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def build_new_factory(self):
        """Build a new production facility"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Create a dialog to select location and type
        dialog = ctk.CTkToplevel(self)
        dialog.title("Build New Factory")
        dialog.geometry("400x300")
        dialog.transient(self)  # Make it a modal dialog
        dialog.grab_set()
        
        # Factory location
        location_frame = ctk.CTkFrame(dialog)
        location_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(location_frame, text="Country:").pack(side=tk.LEFT, padx=10)
        
        # Get list of countries from business existing factories or add headquarter country
        countries = list(business.factories.keys())
        if business.headquarters_country and business.headquarters_country not in countries:
            countries.append(business.headquarters_country)
            
        # Add some other major countries as options
        additional_countries = ["United States", "China", "Germany", "Japan", "United Kingdom"]
        for country in additional_countries:
            if country not in countries:
                countries.append(country)
        
        country_var = tk.StringVar(value=business.headquarters_country)
        country_menu = ctk.CTkOptionMenu(
            location_frame,
            values=countries,
            variable=country_var
        )
        country_menu.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Factory type
        type_frame = ctk.CTkFrame(dialog)
        type_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(type_frame, text="Factory Type:").pack(side=tk.LEFT, padx=10)
        factory_type = ctk.CTkOptionMenu(
            type_frame,
            values=[
                "Production Facility",
                "Assembly Plant",
                "High-Tech Manufacturing"
            ]
        )
        factory_type.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Factory size/capacity
        size_frame = ctk.CTkFrame(dialog)
        size_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(size_frame, text="Size:").pack(side=tk.LEFT, padx=10)
        factory_size = ctk.CTkOptionMenu(
            size_frame,
            values=[
                "Small (capacity: 5,000 units/month)",
                "Medium (capacity: 10,000 units/month)",
                "Large (capacity: 25,000 units/month)"
            ]
        )
        factory_size.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Factory cost calculation
        cost_frame = ctk.CTkFrame(dialog)
        cost_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(cost_frame, text="Estimated Cost:").pack(side=tk.LEFT, padx=10)
        cost_label = ctk.CTkLabel(cost_frame, text="$10,000,000")
        cost_label.pack(side=tk.LEFT, padx=10)
        
        # Calculate cost based on selections
        def update_cost(*args):
            size_text = factory_size.get()
            country = country_var.get()
            type_text = factory_type.get()
            
            # Base costs by size
            if "Small" in size_text:
                base_cost = 10000000  # $10M
            elif "Medium" in size_text:
                base_cost = 20000000  # $20M
            else:
                base_cost = 40000000  # $40M
            
            # Adjust for country
            country_multipliers = {
                "United States": 1.2,
                "Germany": 1.3,
                "Japan": 1.25,
                "United Kingdom": 1.15,
                "China": 0.8,
                "India": 0.7,
                "Mexico": 0.75,
                "Brazil": 0.85
            }
            
            country_mult = country_multipliers.get(country, 1.0)
            
            # Adjust for factory type
            if "High-Tech" in type_text:
                type_mult = 1.5
            elif "Assembly" in type_text:
                type_mult = 0.9
            else:
                type_mult = 1.0
            
            # Calculate final cost
            final_cost = base_cost * country_mult * type_mult
            cost_label.configure(text=f"${final_cost:,.2f}")
            
            # Store cost for build button
            dialog.final_cost = final_cost
        
        # Bind the cost calculation to selection changes
        factory_size.configure(command=update_cost)
        factory_type.configure(command=update_cost)
        country_menu.configure(command=update_cost)
        
        # Initial cost calculation
        update_cost()
        
        # Build button
        ctk.CTkButton(
            dialog,
            text="Build Factory",
            command=lambda: self.confirm_factory_build(
                dialog, 
                country_var.get(), 
                factory_type.get(), 
                factory_size.get(), 
                dialog.final_cost
            )
        ).pack(pady=10)
    
    def confirm_factory_build(self, dialog, country, factory_type, factory_size, cost):
        """Confirm and execute factory building"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            dialog.destroy()
            return
        
        # Check if business has enough cash
        if cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"Building this factory requires {self.format_currency(cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Extract size from text
        if "Small" in factory_size:
            size = "Small"
            capacity = 5000
        elif "Medium" in factory_size:
            size = "Medium"
            capacity = 10000
        else:
            size = "Large"
            capacity = 25000
        
        # Build the factory
        business.cash -= cost
        business.total_assets += cost * 0.9  # 90% of cost becomes an asset
        
        # Add to factories dictionary
        if country in business.factories:
            business.factories[country] += 1
        else:
            business.factories[country] = 1
        
        # Close dialog
        dialog.destroy()
        
        # Show success message
        messagebox.showinfo("Factory Built", 
                         f"Successfully built a {size} {factory_type} in {country} "
                         f"with a capacity of {capacity:,} units/month.")
        
        self.update_display()
    
    def issue_dividend(self):
        """Issue a dividend to shareholders"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Check if business is public
        if not business.is_public:
            messagebox.showerror("Not Public", 
                              "Your company needs to be publicly traded to issue dividends.")
            return
            
        try:
            # Get dividend amount
            amount_per_share = float(self.dividend_amount.get())
            if amount_per_share <= 0:
                raise ValueError("Dividend amount must be positive")
                
            # Calculate total dividend payment
            total_payment = amount_per_share * business.shares_outstanding
            
            # Check if business has enough cash
            if total_payment > business.cash:
                messagebox.showerror("Insufficient Funds", 
                                   f"Dividend payment requires {self.format_currency(total_payment)} "
                                   f"but you only have {self.format_currency(business.cash)}")
                return
                
            # Ask for confirmation
            result = messagebox.askyesno("Confirm Dividend", 
                                       f"Issue dividend of ${amount_per_share:.2f} per share?\n\n"
                                       f"Total payment: {self.format_currency(total_payment)}\n"
                                       f"Shares outstanding: {business.shares_outstanding:,}")
            
            if not result:
                return
                
            # Issue the dividend
            business.cash -= total_payment
            
            messagebox.showinfo("Dividend Issued", 
                             f"Successfully issued dividend of ${amount_per_share:.2f} per share "
                             f"for a total of {self.format_currency(total_payment)}.")
            
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def repurchase_shares(self):
        """Repurchase company shares from the market"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Check if business is public
        if not business.is_public:
            messagebox.showerror("Not Public", 
                              "Your company needs to be publicly traded to repurchase shares.")
            return
            
        try:
            # Get number of shares to repurchase
            shares = int(self.repurchase_amount.get())
            if shares <= 0:
                raise ValueError("Number of shares must be positive")
                
            # Check if trying to repurchase more shares than outstanding
            if shares > business.shares_outstanding:
                raise ValueError(f"Cannot repurchase more than the {business.shares_outstanding:,} outstanding shares")
                
            # Calculate total cost
            total_cost = shares * business.share_price * 1.05  # 5% premium
            
            # Check if business has enough cash
            if total_cost > business.cash:
                messagebox.showerror("Insufficient Funds", 
                                   f"Share repurchase requires {self.format_currency(total_cost)} "
                                   f"but you only have {self.format_currency(business.cash)}")
                return
                
            # Ask for confirmation
            result = messagebox.askyesno("Confirm Repurchase", 
                                       f"Repurchase {shares:,} shares at ${business.share_price * 1.05:.2f} per share?\n\n"
                                       f"Total cost: {self.format_currency(total_cost)}\n"
                                       f"This includes a 5% market premium.")
            
            if not result:
                return
                
            # Execute the repurchase
            business.cash -= total_cost
            business.shares_outstanding -= shares
            
            # Update market cap
            business.market_cap = business.shares_outstanding * business.share_price
            
            messagebox.showinfo("Shares Repurchased", 
                             f"Successfully repurchased {shares:,} shares "
                             f"at a total cost of {self.format_currency(total_cost)}.")
            
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def issue_bonds(self):
        """Issue corporate bonds to raise capital"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Create a dialog for bond issuance
        dialog = ctk.CTkToplevel(self)
        dialog.title("Issue Corporate Bonds")
        dialog.geometry("400x350")
        dialog.transient(self)  # Make it a modal dialog
        dialog.grab_set()
        
        ctk.CTkLabel(
            dialog,
            text="Issue Corporate Bonds",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(20, 10))
        
        # Amount to raise
        amount_frame = ctk.CTkFrame(dialog)
        amount_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ctk.CTkLabel(amount_frame, text="Amount to Raise:").pack(side=tk.LEFT, padx=10)
        amount_entry = ctk.CTkEntry(amount_frame, width=150)
        amount_entry.pack(side=tk.LEFT, padx=10)
        amount_entry.insert(0, "10000000")  # Default $10M
        ctk.CTkLabel(amount_frame, text="$").pack(side=tk.LEFT)
        
        # Bond term
        term_frame = ctk.CTkFrame(dialog)
        term_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ctk.CTkLabel(term_frame, text="Term (years):").pack(side=tk.LEFT, padx=10)
        term_var = tk.StringVar(value="5")
        terms = [("3 years", "3"), ("5 years", "5"), ("7 years", "7"), ("10 years", "10")]
        
        for text, value in terms:
            ctk.CTkRadioButton(
                term_frame,
                text=text,
                variable=term_var,
                value=value
            ).pack(side=tk.LEFT, padx=5)
        
        # Interest rate (based on credit rating and term)
        rate_frame = ctk.CTkFrame(dialog)
        rate_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ctk.CTkLabel(rate_frame, text="Interest Rate:").pack(side=tk.LEFT, padx=10)
        
        # Calculate initial rate
        base_rate = 0.03  # 3% base rate
        
        # Adjust for credit rating
        rating_to_premium = {
            "AAA": 0.0, "AA": 0.005, "A": 0.01, "BBB": 0.02,
            "BB": 0.04, "B": 0.07, "CCC": 0.12, "CC": 0.18, "C": 0.25, "D": 0.4
        }
        
        rating_premium = rating_to_premium.get(business.credit_rating, 0.05)
        
        # Adjust for term (longer term = higher rate)
        term_premium = (int(term_var.get()) / 10) * 0.02
        
        initial_rate = base_rate + rating_premium + term_premium
        
        rate_label = ctk.CTkLabel(rate_frame, text=f"{initial_rate*100:.2f}%")
        rate_label.pack(side=tk.LEFT, padx=10)
        
        # Update rate when term changes
        def update_rate(*args):
            term_value = int(term_var.get())
            term_premium = (term_value / 10) * 0.02
            new_rate = base_rate + rating_premium + term_premium
            rate_label.configure(text=f"{new_rate*100:.2f}%")
            dialog.interest_rate = new_rate
        
        term_var.trace("w", update_rate)
        dialog.interest_rate = initial_rate
        
        # Annual interest payment
        payment_frame = ctk.CTkFrame(dialog)
        payment_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ctk.CTkLabel(payment_frame, text="Annual Interest Payment:").pack(side=tk.LEFT, padx=10)
        payment_label = ctk.CTkLabel(payment_frame, text="$0")
        payment_label.pack(side=tk.LEFT, padx=10)
        
        # Calculate payment when amount or rate changes
        def update_payment(*args):
            try:
                amount = float(amount_entry.get())
                payment = amount * dialog.interest_rate
                payment_label.configure(text=f"${payment:,.2f}")
            except ValueError:
                payment_label.configure(text="Invalid amount")
        
        amount_entry.bind("<KeyRelease>", update_payment)
        term_var.trace("w", update_payment)
        update_payment()  # Initial calculation
        
        # Issue button
        ctk.CTkButton(
            dialog,
            text="Issue Bonds",
            fg_color=PRIMARY_COLOR,
            command=lambda: self.confirm_bond_issue(
                dialog, 
                amount_entry,
                int(term_var.get()),
                dialog.interest_rate
            )
        ).pack(pady=20)
        
        # Cancel button
        ctk.CTkButton(
            dialog,
            text="Cancel",
            fg_color=SECONDARY_COLOR,
            command=dialog.destroy
        ).pack(pady=(0, 10))
    
    def confirm_bond_issue(self, dialog, amount_entry, term, interest_rate):
        """Confirm and process corporate bond issuance"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            dialog.destroy()
            return
            
        try:
            amount = float(amount_entry.get())
            if amount <= 0:
                raise ValueError("Amount must be positive")
                
            # Execute bond issuance
            success = business.take_loan(amount, interest_rate, term)
            
            if success:
                dialog.destroy()
                
                messagebox.showinfo("Bonds Issued", 
                                 f"Successfully issued ${amount:,.2f} in {term}-year corporate bonds "
                                 f"at {interest_rate*100:.2f}% interest rate.")
                
                self.update_display()
            else:
                messagebox.showerror("Issue Failed", 
                                   "Could not issue bonds. Please check your company's financial status.")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def issue_new_shares(self):
        """Issue new shares to raise capital"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Check if business is public
        if not business.is_public:
            messagebox.showerror("Not Public", 
                              "Your company needs to be publicly traded to issue new shares.")
            return
        
        # Create a dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Issue New Shares")
        dialog.geometry("400x300")
        dialog.transient(self)  # Make it a modal dialog
        dialog.grab_set()
        
        ctk.CTkLabel(
            dialog,
            text="Issue New Shares",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(20, 10))
        
        # Number of shares
        shares_frame = ctk.CTkFrame(dialog)
        shares_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ctk.CTkLabel(shares_frame, text="Number of Shares:").pack(side=tk.LEFT, padx=10)
        shares_entry = ctk.CTkEntry(shares_frame, width=150)
        shares_entry.pack(side=tk.LEFT, padx=10)
        
        # Default to 5% of outstanding shares
        default_shares = int(business.shares_outstanding * 0.05)
        shares_entry.insert(0, str(default_shares))
        
        # Current stats
        stats_frame = ctk.CTkFrame(dialog)
        stats_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ctk.CTkLabel(
            stats_frame,
            text=f"Current Share Price: ${business.share_price:.2f}"
        ).pack(anchor="w", padx=10, pady=2)
        
        ctk.CTkLabel(
            stats_frame,
            text=f"Shares Outstanding: {business.shares_outstanding:,}"
        ).pack(anchor="w", padx=10, pady=2)
        
        ctk.CTkLabel(
            stats_frame,
            text=f"Market Cap: {self.format_currency(business.market_cap)}"
        ).pack(anchor="w", padx=10, pady=2)
        
        # Discount rate (shares are typically issued at a discount to market price)
        discount_frame = ctk.CTkFrame(dialog)
        discount_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ctk.CTkLabel(discount_frame, text="Issue Price Discount:").pack(side=tk.LEFT, padx=10)
        
        discount_var = tk.DoubleVar(value=5.0)  # Default 5% discount
        discount_slider = ctk.CTkSlider(
            discount_frame,
            from_=0,
            to=15,
            number_of_steps=15,
            variable=discount_var
        )
        discount_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        discount_label = ctk.CTkLabel(discount_frame, text="5.0%")
        discount_label.pack(side=tk.LEFT, padx=10)
        
        # Issues stats
        result_frame = ctk.CTkFrame(dialog)
        result_frame.pack(fill=tk.X, padx=20, pady=10)
        
        issue_price_label = ctk.CTkLabel(
            result_frame,
            text=f"Issue Price: ${business.share_price * 0.95:.2f}"
        )
        issue_price_label.pack(anchor="w", padx=10, pady=2)
        
        proceeds_label = ctk.CTkLabel(
            result_frame,
            text=f"Expected Proceeds: ${default_shares * business.share_price * 0.95:,.2f}"
        )
        proceeds_label.pack(anchor="w", padx=10, pady=2)
        
        dilution_label = ctk.CTkLabel(
            result_frame,
            text=f"Dilution: {(default_shares / business.shares_outstanding) * 100:.2f}%"
        )
        dilution_label.pack(anchor="w", padx=10, pady=2)
        
        # Update calculations when inputs change
        def update_calculations(*args):
            try:
                shares = int(shares_entry.get())
                discount = discount_var.get() / 100
                issue_price = business.share_price * (1 - discount)
                proceeds = shares * issue_price
                dilution = (shares / business.shares_outstanding) * 100
                
                issue_price_label.configure(text=f"Issue Price: ${issue_price:.2f}")
                proceeds_label.configure(text=f"Expected Proceeds: ${proceeds:,.2f}")
                dilution_label.configure(text=f"Dilution: {dilution:.2f}%")
                
                discount_label.configure(text=f"{discount_var.get():.1f}%")
            except ValueError:
                issue_price_label.configure(text="Issue Price: Invalid input")
                proceeds_label.configure(text="Expected Proceeds: Invalid input")
                dilution_label.configure(text="Dilution: Invalid input")
        
        shares_entry.bind("<KeyRelease>", update_calculations)
        discount_slider.configure(command=lambda value: update_calculations())
        
        # Issue button
        ctk.CTkButton(
            dialog,
            text="Issue Shares",
            fg_color=PRIMARY_COLOR,
            command=lambda: self.confirm_share_issue(
                dialog,
                shares_entry,
                discount_var.get() / 100
            )
        ).pack(pady=10)
    
    def confirm_share_issue(self, dialog, shares_entry, discount):
        """Confirm and process new share issuance"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            dialog.destroy()
            return
            
        try:
            shares = int(shares_entry.get())
            if shares <= 0:
                raise ValueError("Number of shares must be positive")
                
            # Calculate price and proceeds
            issue_price = business.share_price * (1 - discount)
            proceeds = shares * issue_price
            
            # Issue the shares
            business.cash += proceeds
            business.total_assets += proceeds
            business.shares_outstanding += shares
            
            # Update market cap
            # In a real market, share price might drop due to dilution
            # Simple approximation: maintain total market cap and recalculate per share
            original_market_cap = business.market_cap
            new_share_price = original_market_cap / business.shares_outstanding
            
            # Adjust share price (blend of original and diluted price)
            business.share_price = (business.share_price * 0.7) + (new_share_price * 0.3)
            
            # Update market cap with new price
            business.market_cap = business.shares_outstanding * business.share_price
            
            dialog.destroy()
            
            messagebox.showinfo("Shares Issued", 
                             f"Successfully issued {shares:,} new shares at ${issue_price:.2f} per share, "
                             f"raising {self.format_currency(proceeds)}.")
            
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def apply_for_loan(self):
        """Apply for a business loan"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        try:
            # Get values
            loan_amount = float(self.loan_amount.get())
            if loan_amount <= 0:
                raise ValueError("Loan amount must be positive")
                
            loan_term = int(self.loan_term.get())
            
            # Calculate interest based on business credit rating and term
            base_rate = 0.03  # 3% base
            
            # Adjust for credit rating
            rating_premium = {
                "AAA": 0.0, "AA": 0.005, "A": 0.01, "BBB": 0.02,
                "BB": 0.04, "B": 0.07, "CCC": 0.12, "CC": 0.18, "C": 0.25, "D": 0.4
            }
            
            credit_adjustment = rating_premium.get(business.credit_rating, 0.05)
            
            # Adjust for term (longer = higher rate)
            term_adjustment = (loan_term / 10) * 0.01
            
            # Final rate
            interest_rate = base_rate + credit_adjustment + term_adjustment
            
            # Ask for confirmation
            annual_payment = loan_amount * (interest_rate)
            
            confirm = messagebox.askyesno("Confirm Loan", 
                                       f"Apply for a ${loan_amount:,.2f} loan?\n\n"
                                       f"Term: {loan_term} years\n"
                                       f"Interest Rate: {interest_rate*100:.2f}%\n"
                                       f"Annual Interest: ${annual_payment:,.2f}")
            
            if not confirm:
                return
                
            # Check approval (simplified)
            debt_to_assets_ratio = (business.debt + loan_amount) / business.total_assets
            
            if debt_to_assets_ratio > 0.7:
                messagebox.showerror("Loan Denied", 
                                   f"Loan application denied. Your debt-to-assets ratio would be too high "
                                   f"({debt_to_assets_ratio:.1%}). Try a smaller loan amount.")
                return
                
            # Approve the loan
            success = business.take_loan(loan_amount, interest_rate, loan_term)
            
            if success:
                messagebox.showinfo("Loan Approved", 
                                 f"Your loan for ${loan_amount:,.2f} has been approved at "
                                 f"{interest_rate*100:.2f}% interest rate!")
                
                # Make a record in the loans table
                if hasattr(self, "loans_table"):
                    # Clear "no loans" message if it's the first loan
                    for widget in self.loans_table.winfo_children():
                        widget.destroy()
                        
                    # Add loan to the table
                    loan_row = ctk.CTkFrame(self.loans_table)
                    loan_row.pack(fill=tk.X, padx=5, pady=5)
                    
                    ctk.CTkLabel(
                        loan_row,
                        text=f"${loan_amount:,.2f}",
                        width=120
                    ).pack(side=tk.LEFT, padx=5)
                    
                    ctk.CTkLabel(
                        loan_row,
                        text=f"{interest_rate*100:.2f}%",
                        width=80
                    ).pack(side=tk.LEFT, padx=5)
                    
                    ctk.CTkLabel(
                        loan_row,
                        text=f"{loan_term} years",
                        width=80
                    ).pack(side=tk.LEFT, padx=5)
                    
                    # Calculate maturity date
                    current_date = self.app.game_state.current_date
                    maturity_date = current_date + datetime.timedelta(days=loan_term*365)
                    
                    ctk.CTkLabel(
                        loan_row,
                        text=maturity_date.strftime("%Y-%m-%d"),
                        width=100
                    ).pack(side=tk.LEFT, padx=5)
                    
                    ctk.CTkButton(
                        loan_row,
                        text="Repay",
                        width=80,
                        command=lambda amt=loan_amount: self.repay_loan(amt)
                    ).pack(side=tk.RIGHT, padx=5)
                
                self.update_display()
            else:
                messagebox.showerror("Error", "Failed to process the loan")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def repay_loan(self, amount):
        """Repay a loan early"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Check if business has enough cash
        if amount > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"Loan repayment requires ${amount:,.2f} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Repayment", 
                                    f"Repay ${amount:,.2f} loan early?\n\n"
                                    f"This will reduce your debt and interest expenses.")
        
        if not confirm:
            return
            
        # Repay the loan
        success = business.pay_debt(amount)
        
        if success:
            messagebox.showinfo("Loan Repaid", 
                             f"Successfully repaid ${amount:,.2f} loan.")
                             
            # Remove from loans table (simplified - in a real game, you'd need to track loan IDs)
            if hasattr(self, "loans_table"):
                # Instead of trying to find the exact loan, just refresh the whole table
                self.update_display()
        else:
            messagebox.showerror("Error", "Failed to repay loan")
    
    def make_investment(self, investment):
        """Make a business investment"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Extract investment info
        name = investment["name"]
        investment_type = "Strategic Investment"
        cost = self.extract_cost(investment["cost"])  # Extract numerical value from cost string
        
        # Check if business has enough cash
        if cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"This investment requires {self.format_currency(cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        return_range = investment["return"]
        risk = investment["risk"]
        
        confirm = messagebox.askyesno("Confirm Investment", 
                                    f"Invest in {name}?\n\n"
                                    f"Cost: {self.format_currency(cost)}\n"
                                    f"Expected Return: {return_range}\n"
                                    f"Risk Level: {risk}")
        
        if not confirm:
            return
            
        # Make the investment
        business.cash -= cost
        business.total_assets += cost  # The investment becomes an asset
        
        # In a real game, you'd track this investment and its performance over time
        # For now, just add to the investments table
        if hasattr(self, "investments_table"):
            # Clear "no investments" message if it's the first investment
            for widget in self.investments_table.winfo_children():
                widget.destroy()
                
            # Add investment to the table
            investment_row = ctk.CTkFrame(self.investments_table)
            investment_row.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkLabel(
                investment_row,
                text=name,
                width=180
            ).pack(side=tk.LEFT, padx=5)
            
            ctk.CTkLabel(
                investment_row,
                text=investment_type,
                width=120
            ).pack(side=tk.LEFT, padx=5)
            
            ctk.CTkLabel(
                investment_row,
                text=self.format_currency(cost),
                width=100
            ).pack(side=tk.LEFT, padx=5)
            
            # Current date
            date = self.app.game_state.current_date.strftime("%Y-%m-%d")
            ctk.CTkLabel(
                investment_row,
                text=date,
                width=100
            ).pack(side=tk.LEFT, padx=5)
            
            ctk.CTkLabel(
                investment_row,
                text="Active",
                text_color=SUCCESS_COLOR,
                width=80
            ).pack(side=tk.LEFT, padx=5)
        
        messagebox.showinfo("Investment Made", 
                         f"Successfully invested {self.format_currency(cost)} in {name}.")
        
        self.update_display()
    
    def extract_cost(self, cost_str):
        """Extract numerical cost from string like '$5M' or '$20M'"""
        if isinstance(cost_str, (int, float)):
            return cost_str
            
        # Remove $ and other non-digit characters except decimal point
        digits = ''.join(c for c in cost_str if c.isdigit() or c == '.')
        
        if not digits:
            return 0
            
        value = float(digits)
        
        # Check for million/billion indicators
        if 'M' in cost_str:
            value *= 1000000
        elif 'B' in cost_str:
            value *= 1000000000
            
        return value
    
    def calculate_ipo(self):
        """Calculate the proceeds from an IPO"""
        try:
            price = float(self.ipo_price.get())
            shares = int(self.ipo_shares.get())
            
            if price <= 0 or shares <= 0:
                raise ValueError("Price and shares must be positive")
                
            proceeds = price * shares
            self.ipo_proceeds.configure(text=f"Proceeds: ${proceeds:,.2f}")
        except ValueError as e:
            self.ipo_proceeds.configure(text="Invalid input")
    
    def go_public(self):
        """Take the company public with an IPO"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Check if already public
        if business.is_public:
            messagebox.showerror("Already Public", "Your company is already publicly traded.")
            return
            
        try:
            # Get values
            exchange = self.ipo_exchange.get()
            price = float(self.ipo_price.get())
            shares = int(self.ipo_shares.get())
            
            if price <= 0:
                raise ValueError("Share price must be positive")
                
            if shares <= 0:
                raise ValueError("Number of shares must be positive")
                
            # Check minimum requirements for IPO
            if business.total_assets < 10000000:
                messagebox.showerror("IPO Requirements", 
                                   "Your business needs at least $10 million in assets for an IPO.")
                return
                
            if business.employees < 50:
                messagebox.showerror("IPO Requirements", 
                                   "Your business needs at least 50 employees for an IPO.")
                return
                
            # Calculate proceeds
            proceeds = price * shares
            
            # Ask for confirmation
            confirm = messagebox.askyesno("Confirm IPO", 
                                       f"Take {business.name} public on {exchange}?\n\n"
                                       f"Initial share price: ${price:.2f}\n"
                                       f"Shares to issue: {shares:,}\n"
                                       f"Expected proceeds: ${proceeds:,.2f}\n\n"
                                       f"This will dilute ownership but raise capital for growth.")
            
            if not confirm:
                return
                
            # Execute IPO
            success = business.go_public(exchange, price, shares)
            
            if success:
                messagebox.showinfo("IPO Complete", 
                                 f"Congratulations! {business.name} is now a publicly traded company "
                                 f"on the {exchange}.\n\n"
                                 f"Raised: ${proceeds:,.2f}\n"
                                 f"Initial market cap: ${business.market_cap:,.2f}")
                
                self.update_display()
            else:
                messagebox.showerror("IPO Failed", "Failed to complete the IPO.")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def launch_campaign(self):
        """Launch a marketing campaign"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        try:
            # Get values
            campaign_type = self.campaign_type.get()
            target_product = self.campaign_product.get()
            budget = float(self.campaign_budget.get())
            duration = int(self.campaign_duration.get())
            
            if budget <= 0:
                raise ValueError("Budget must be positive")
                
            if duration <= 0:
                raise ValueError("Duration must be positive")
                
            # Check if business has enough cash
            if budget > business.cash:
                messagebox.showerror("Insufficient Funds", 
                                   f"This campaign requires {self.format_currency(budget)} "
                                   f"but you only have {self.format_currency(business.cash)}")
                return
                
            # Launch the campaign
            business.cash -= budget
            
            # Find target product(s)
            if target_product == "All Products":
                # Apply to all products
                for product in business.products:
                    product.invest_in_marketing(budget / len(business.products))
            else:
                # Find the specific product
                for product in business.products:
                    if product.name == target_product:
                        product.invest_in_marketing(budget)
                        break
            
            # Increase brand value
            business.brand_value += (budget / 10000000)  # Arbitrary formula
            
            messagebox.showinfo("Campaign Launched", 
                             f"Successfully launched a {campaign_type} campaign "
                             f"targeting {target_product} with a budget of {self.format_currency(budget)} "
                             f"for {duration} months.")
            
            # Add to campaigns list if it exists
            if hasattr(self, "campaigns_list"):
                # Clear out "No campaigns" message if present
                for widget in self.campaigns_list.winfo_children():
                    if isinstance(widget, ctk.CTkLabel) and "No active campaigns" in widget.cget("text"):
                        widget.destroy()
                
                # Add campaign to the list
                campaign_frame = ctk.CTkFrame(self.campaigns_list)
                campaign_frame.pack(fill=tk.X, padx=5, pady=5)
                
                campaign_info = ctk.CTkFrame(campaign_frame)
                campaign_info.pack(fill=tk.X, padx=5, pady=2)
                
                ctk.CTkLabel(
                    campaign_info,
                    text=campaign_type,
                    font=("Helvetica", 12, "bold")
                ).pack(side=tk.LEFT, padx=10)
                
                ctk.CTkLabel(
                    campaign_info,
                    text=f"Target: {target_product}"
                ).pack(side=tk.LEFT, padx=10)
                
                stats_frame = ctk.CTkFrame(campaign_frame)
                stats_frame.pack(fill=tk.X, padx=5, pady=2)
                
                ctk.CTkLabel(
                    stats_frame,
                    text=f"Budget: {self.format_currency(budget)}"
                ).pack(side=tk.LEFT, padx=10)
                
                # Calculate end date
                start_date = self.app.game_state.current_date
                end_date = start_date + datetime.timedelta(days=duration*30)
                
                ctk.CTkLabel(
                    stats_frame,
                    text=f"Duration: {duration} months (ends {end_date.strftime('%Y-%m-%d')})"
                ).pack(side=tk.LEFT, padx=10)
                
                # Progress bar (simplified - in a real game, this would update over time)
                progress_frame = ctk.CTkFrame(campaign_frame)
                progress_frame.pack(fill=tk.X, padx=5, pady=5)
                
                ctk.CTkLabel(progress_frame, text="Progress:").pack(side=tk.LEFT, padx=10)
                
                progress = ctk.CTkProgressBar(progress_frame, width=200)
                progress.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
                progress.set(0.1)  # Just started
            
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def commission_research(self, research_type, cost):
        """Commission market research"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Check if business has enough cash
        if cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"This research costs {self.format_currency(cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Research", 
                                    f"Commission {research_type} for {self.format_currency(cost)}?")
        
        if not confirm:
            return
            
        # Conduct the research
        business.cash -= cost
        
        # In a real game, this would generate detailed market insights
        # For now, just show a summary of findings
        
        # Generate random market insights
        if research_type == "Consumer Survey":
            insights = [
                "73% of consumers prefer eco-friendly packaging",
                "Price is the #1 factor for 42% of purchase decisions",
                "Brand loyalty is declining among younger consumers",
                "Online reviews influence 68% of purchasing decisions",
                "Product quality ranks as the most important factor overall"
            ]
        elif research_type == "Competitor Analysis":
            insights = [
                "Main competitor is planning a major product launch next quarter",
                "Competitor prices are 7% higher on average",
                "Smaller competitors are gaining market share in niche segments",
                "Largest competitor is struggling with supply chain issues",
                "Industry consolidation is likely in the next 12-18 months"
            ]
        elif research_type == "Market Trends":
            insights = [
                "The market is growing at 8.3% annually, above the industry average",
                "Subscription-based models are gaining popularity",
                "Environmental sustainability is becoming a key purchasing factor",
                "Direct-to-consumer sales are disrupting traditional channels",
                "International markets present significant growth opportunities"
            ]
        elif research_type == "Product Testing":
            insights = [
                "Users rate your core product features 7.8/10 on average",
                "Reliability issues were reported by 12% of testers",
                "Price point is perceived as 'fair' by 65% of consumers",
                "Design aesthetics received the highest ratings",
                "Packaging needs improvement according to 38% of feedback"
            ]
        elif research_type == "Price Sensitivity":
            insights = [
                f"Optimal price point for main product is ${business.products[0].price * 1.15:.2f} (15% higher)",
                "Price elasticity is lower than industry average",
                "Premium pricing strategy is viable for 30% of your product line",
                "Bulk discounts would increase sales volume by an estimated 22%",
                "Price is less important than quality for your target demographic"
            ]
        else:
            insights = [
                "Market research complete",
                "Several actionable insights identified",
                "Full report available to management team"
            ]
        
        # Create a research results dialog
        results_dialog = ctk.CTkToplevel(self)
        results_dialog.title(f"Research Results: {research_type}")
        results_dialog.geometry("500x400")
        results_dialog.transient(self)  # Make it a modal dialog
        results_dialog.grab_set()
        
        ctk.CTkLabel(
            results_dialog,
            text=f"{research_type} Results",
            font=("Helvetica", 18, "bold")
        ).pack(pady=(20, 10))
        
        ctk.CTkLabel(
            results_dialog,
            text="Key Findings:",
            font=("Helvetica", 14)
        ).pack(pady=(10, 5), anchor="w", padx=20)
        
        # Add insights
        insights_frame = ctk.CTkScrollableFrame(results_dialog, height=200)
        insights_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        for i, insight in enumerate(insights):
            insight_frame = ctk.CTkFrame(insights_frame)
            insight_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkLabel(
                insight_frame,
                text=f"{i+1}.",
                width=30
            ).pack(side=tk.LEFT, padx=5)
            
            ctk.CTkLabel(
                insight_frame,
                text=insight,
                wraplength=400,
                justify="left"
            ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Recommendations
        ctk.CTkLabel(
            results_dialog,
            text="Recommendations:",
            font=("Helvetica", 14)
        ).pack(pady=(10, 5), anchor="w", padx=20)
        
        recommendation = "Based on these findings, consider adjusting your marketing strategy and product development priorities."
        
        ctk.CTkLabel(
            results_dialog,
            text=recommendation,
            wraplength=460
        ).pack(pady=5, padx=20)
        
        # Close button
        ctk.CTkButton(
            results_dialog,
            text="Apply Insights",
            command=results_dialog.destroy
        ).pack(pady=20)
        
        # Update UI
        self.update_display()
    
    def apply_pricing_strategy(self, strategy):
        """Apply a pricing strategy to products"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business) or not business.products:
            return
            
        # Create a dialog to select which products to apply to
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Apply {strategy}")
        dialog.geometry("400x400")
        dialog.transient(self)  # Make it a modal dialog
        dialog.grab_set()
        
        ctk.CTkLabel(
            dialog,
            text=f"Apply {strategy} Strategy",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(20, 10))
        
        # Strategy description
        descriptions = {
            "Premium Pricing": "Set prices higher to signal quality and exclusivity. Best for high-quality products with strong branding.",
            "Value Pricing": "Set competitive prices that offer good value. Best for mainstream products in competitive markets.",
            "Penetration Pricing": "Set prices below market to gain market share. Best for new products or entering new markets.",
            "Skimming": "Start with high prices and gradually reduce them. Best for innovative products with limited competition.",
            "Loss Leader": "Price below cost to attract customers. Best for products that drive additional purchases."
        }
        
        ctk.CTkLabel(
            dialog,
            text=descriptions.get(strategy, ""),
            wraplength=360,
            justify="left"
        ).pack(pady=10, padx=20)
        
        # Select which products to apply to
        ctk.CTkLabel(
            dialog,
            text="Select Products:",
            font=("Helvetica", 14)
        ).pack(pady=(10, 5), anchor="w", padx=20)
        
        # Scrollable product selection
        products_frame = ctk.CTkScrollableFrame(dialog, height=150)
        products_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Checkboxes for each product
        product_vars = {}
        for product in business.products:
            var = tk.BooleanVar(value=False)
            product_vars[product.id] = var
            
            product_row = ctk.CTkFrame(products_frame)
            product_row.pack(fill=tk.X, padx=5, pady=2)
            
            ctk.CTkCheckBox(
                product_row,
                text=f"{product.name} (Current: ${product.price:.2f})",
                variable=var
            ).pack(side=tk.LEFT, padx=10)
        
        # Apply percentage adjustment
        adjustment_frame = ctk.CTkFrame(dialog)
        adjustment_frame.pack(fill=tk.X, padx=20, pady=10)
        
        if strategy == "Premium Pricing":
            default_adjustment = 25  # +25%
        elif strategy == "Value Pricing":
            default_adjustment = 0  # no change
        elif strategy == "Penetration Pricing":
            default_adjustment = -15  # -15%
        elif strategy == "Skimming":
            default_adjustment = 40  # +40%
        elif strategy == "Loss Leader":
            default_adjustment = -30  # -30%
        else:
            default_adjustment = 0
            
        ctk.CTkLabel(adjustment_frame, text="Price Adjustment:").pack(side=tk.LEFT, padx=10)
        
        adjustment_var = tk.DoubleVar(value=default_adjustment)
        adjustment_entry = ctk.CTkEntry(adjustment_frame, width=80, textvariable=adjustment_var)
        adjustment_entry.pack(side=tk.LEFT, padx=10)
        
        ctk.CTkLabel(adjustment_frame, text="%").pack(side=tk.LEFT)
        
        # Preview changes
        preview_frame = ctk.CTkFrame(dialog)
        preview_frame.pack(fill=tk.X, padx=20, pady=10)
        
        preview_text = ctk.CTkTextbox(preview_frame, height=80)
        preview_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Update preview when adjustment changes
        def update_preview(*args):
            try:
                adjustment = adjustment_var.get() / 100  # Convert percentage to multiplier
                
                selected_products = [p for p in business.products if product_vars[p.id].get()]
                
                preview_text.configure(state=tk.NORMAL)
                preview_text.delete("1.0", tk.END)
                
                for product in selected_products:
                    new_price = product.price * (1 + adjustment)
                    preview_text.insert(tk.END, f"{product.name}: ${product.price:.2f} → ${new_price:.2f}\n")
                
                if not selected_products:
                    preview_text.insert(tk.END, "No products selected")
                    
                preview_text.configure(state=tk.DISABLED)
            except:
                preview_text.configure(state=tk.NORMAL)
                preview_text.delete("1.0", tk.END)
                preview_text.insert(tk.END, "Invalid adjustment value")
                preview_text.configure(state=tk.DISABLED)
        
        adjustment_var.trace("w", update_preview)
        
        # Initial preview
        update_preview()
        
        # Apply button
        ctk.CTkButton(
            dialog,
            text="Apply Pricing Strategy",
            command=lambda: self.execute_pricing_strategy(
                dialog,
                strategy,
                {prod_id: var.get() for prod_id, var in product_vars.items()},
                adjustment_var.get() / 100
            ),
            fg_color=PRIMARY_COLOR
        ).pack(pady=(10, 5))
        
        # Cancel button
        ctk.CTkButton(
            dialog,
            text="Cancel",
            command=dialog.destroy,
            fg_color=SECONDARY_COLOR
        ).pack(pady=(0, 10))
    
    def execute_pricing_strategy(self, dialog, strategy, selected_products, adjustment):
        """Execute the pricing strategy on selected products"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            dialog.destroy()
            return
            
        # Get selected products
        products_to_update = [p for p in business.products if selected_products.get(p.id, False)]
        
        if not products_to_update:
            messagebox.showwarning("No Products", "No products were selected to update.")
            return
            
        # Apply pricing changes
        for product in products_to_update:
            new_price = product.price * (1 + adjustment)
            if new_price <= 0:
                messagebox.showerror("Invalid Price", 
                                   f"The adjustment would result in a negative or zero price for {product.name}.")
                return
                
            product.set_price(new_price)
        
        dialog.destroy()
        
        messagebox.showinfo("Pricing Strategy Applied", 
                         f"Successfully applied {strategy} pricing strategy to {len(products_to_update)} products.")
        
        self.update_display()
    
    def implement_brand_initiative(self, initiative, cost):
        """Implement a brand initiative"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Check if business has enough cash
        if cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"This initiative costs {self.format_currency(cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Initiative", 
                                    f"Implement {initiative} for {self.format_currency(cost)}?")
        
        if not confirm:
            return
            
        # Implement the initiative
        business.cash -= cost
        
        # Apply effects based on initiative type
        if initiative == "Corporate Social Responsibility":
            brand_increase = 0.3 + (random.random() * 0.2)  # 0.3-0.5
            message = "Successfully launched CSR program, improving public perception and brand value."
        elif initiative == "Brand Refresh":
            brand_increase = 0.4 + (random.random() * 0.3)  # 0.4-0.7
            message = "Brand refresh completed, modernizing your company's image and increasing recognition."
        elif initiative == "Customer Loyalty Program":
            brand_increase = 0.2 + (random.random() * 0.2)  # 0.2-0.4
            message = "Loyalty program launched, improving customer retention and lifetime value."
        elif initiative == "Sponsorships":
            brand_increase = 0.3 + (random.random() * 0.2)  # 0.3-0.5
            message = "Sponsorship deals secured, increasing brand visibility and positive associations."
        elif initiative == "Public Relations Campaign":
            brand_increase = 0.2 + (random.random() * 0.3)  # 0.2-0.5
            message = "PR campaign successful in shaping public narrative around your brand."
        else:
            brand_increase = 0.2
            message = "Brand initiative successfully implemented."
        
        # Increase brand value
        business.brand_value += brand_increase
        
        # Also increase product brand awareness
        for product in business.products:
            product.brand_awareness = min(1.0, product.brand_awareness + (brand_increase * 0.1))
        
        messagebox.showinfo("Initiative Implemented", message)
        
        self.update_display()
    
    def start_rd_project(self):
        """Start a new R&D project"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        try:
            # Get values
            name = self.rd_name.get().strip()
            project_type = self.rd_type.get()
            budget = float(self.rd_budget.get())
            duration = int(self.rd_duration.get())
            probability = float(self.rd_probability.get())
            
            if not name:
                raise ValueError("Project name is required")
                
            if budget <= 0:
                raise ValueError("Budget must be positive")
                
            if duration <= 0:
                raise ValueError("Duration must be positive")
                
            if probability <= 0 or probability > 100:
                raise ValueError("Success probability must be between 1 and 100")
                
            # Check if business has enough cash
            if budget > business.cash:
                messagebox.showerror("Insufficient Funds", 
                                   f"This project requires {self.format_currency(budget)} "
                                   f"but you only have {self.format_currency(business.cash)}")
                return
                
            # Ask for confirmation
            confirm = messagebox.askyesno("Confirm R&D Project", 
                                       f"Start {name} ({project_type}) project?\n\n"
                                       f"Budget: {self.format_currency(budget)}\n"
                                       f"Duration: {duration} months\n"
                                       f"Success probability: {probability}%")
            
            if not confirm:
                return
                
            # Start the project
            business.cash -= budget
            
            # In a real game, you'd track ongoing R&D projects
            # For now, add to active projects list if it exists
            if hasattr(self, "rd_projects"):
                # Clear "no projects" message if it's the first project
                for widget in self.rd_projects.winfo_children():
                    if isinstance(widget, ctk.CTkLabel) and "No active" in widget.cget("text"):
                        widget.destroy()
                
                # Add project to the list
                project_frame = ctk.CTkFrame(self.rd_projects)
                project_frame.pack(fill=tk.X, padx=5, pady=5)
                
                project_info = ctk.CTkFrame(project_frame)
                project_info.pack(fill=tk.X, padx=5, pady=2)
                
                ctk.CTkLabel(
                    project_info,
                    text=name,
                    font=("Helvetica", 12, "bold")
                ).pack(side=tk.LEFT, padx=10)
                
                ctk.CTkLabel(
                    project_info,
                    text=f"Type: {project_type}"
                ).pack(side=tk.LEFT, padx=10)
                
                ctk.CTkLabel(
                    project_info,
                    text=f"Budget: {self.format_currency(budget)}"
                ).pack(side=tk.RIGHT, padx=10)
                
                # Progress info
                progress_info = ctk.CTkFrame(project_frame)
                progress_info.pack(fill=tk.X, padx=5, pady=2)
                
                # Random progress between 5-15%
                initial_progress = 0.05 + (random.random() * 0.1)
                
                ctk.CTkLabel(
                    progress_info,
                    text=f"Progress: {initial_progress*100:.1f}%"
                ).pack(side=tk.LEFT, padx=10)
                
                ctk.CTkLabel(
                    progress_info,
                    text=f"Success Probability: {probability}%"
                ).pack(side=tk.LEFT, padx=10)
                
                # Calculate completion date
                start_date = self.app.game_state.current_date
                end_date = start_date + datetime.timedelta(days=duration*30)
                
                ctk.CTkLabel(
                    progress_info,
                    text=f"Expected completion: {end_date.strftime('%Y-%m-%d')}"
                ).pack(side=tk.RIGHT, padx=10)
                
                # Progress bar
                progress_bar = ctk.CTkProgressBar(project_frame)
                progress_bar.pack(fill=tk.X, padx=10, pady=5)
                progress_bar.set(initial_progress)
                
                # Actions
                actions_frame = ctk.CTkFrame(project_frame)
                actions_frame.pack(fill=tk.X, padx=5, pady=2)
                
                ctk.CTkButton(
                    actions_frame,
                    text="Increase Funding",
                    width=150,
                    command=lambda: self.increase_rd_funding(name, budget * 0.25)
                ).pack(side=tk.LEFT, padx=10)
                
                ctk.CTkButton(
                    actions_frame,
                    text="Cancel Project",
                    width=150,
                    fg_color=DANGER_COLOR,
                    hover_color="#c0392b",
                    command=lambda f=project_frame: f.destroy()
                ).pack(side=tk.RIGHT, padx=10)
            
            # Clear the form
            self.rd_name.delete(0, tk.END)
            
            messagebox.showinfo("R&D Project Started", 
                             f"Successfully started {name} R&D project. "
                             f"Expected completion in {duration} months.")
            
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def increase_rd_funding(self, project_name, additional_funding):
        """Increase funding for an R&D project"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Check if business has enough cash
        if additional_funding > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"This funding increase requires {self.format_currency(additional_funding)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Funding Increase", 
                                    f"Increase funding for {project_name} by {self.format_currency(additional_funding)}?\n\n"
                                    f"This will improve the project's chances of success and potentially reduce completion time.")
        
        if not confirm:
            return
            
        # Apply additional funding
        business.cash -= additional_funding
        
        # In a real game this would update the project's properties
        messagebox.showinfo("Funding Increased", 
                         f"Successfully increased funding for {project_name} by {self.format_currency(additional_funding)}.\n\n"
                         f"Success probability increased and estimated completion time reduced.")
        
        self.update_display()
    
    def file_patent(self):
        """File a new patent application"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Get values
        patent_title = self.patent_title.get().strip()
        patent_category = self.patent_category.get()
        
        # Fixed filing cost for now
        filing_cost = 50000
        
        if not patent_title:
            messagebox.showerror("Input Error", "Patent title is required")
            return
            
        # Check if business has enough cash
        if filing_cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"Patent filing costs ${filing_cost:,} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Patent Filing", 
                                    f"File patent for '{patent_title}'?\n\n"
                                    f"Category: {patent_category}\n"
                                    f"Filing Cost: ${filing_cost:,}\n\n"
                                    f"This will protect your intellectual property and potentially "
                                    f"create licensing opportunities.")
        
        if not confirm:
            return
            
        # File the patent
        business.cash -= filing_cost
        
        # In a real game, you'd track patents and their status
        # For now, add to patents list if it exists
        if hasattr(self, "patents_list"):
            # Clear "no patents" message if it's the first patent
            for widget in self.patents_list.winfo_children():
                if isinstance(widget, ctk.CTkLabel) and "No patents" in widget.cget("text"):
                    widget.destroy()
                    
            # Add patent to the list
            patent_frame = ctk.CTkFrame(self.patents_list)
            patent_frame.pack(fill=tk.X, padx=5, pady=5)
            
            patent_info = ctk.CTkFrame(patent_frame)
            patent_info.pack(fill=tk.X, padx=5, pady=2)
            
            ctk.CTkLabel(
                patent_info,
                text=patent_title,
                font=("Helvetica", 12, "bold")
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                patent_info,
                text=f"Category: {patent_category}"
            ).pack(side=tk.LEFT, padx=10)
            
            # Filing date
            filing_date = self.app.game_state.current_date.strftime("%Y-%m-%d")
            
            # Status (pending initially)
            status_frame = ctk.CTkFrame(patent_frame)
            status_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ctk.CTkLabel(
                status_frame,
                text=f"Filed: {filing_date}"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                status_frame,
                text="Status: Pending",
                text_color=WARNING_COLOR
            ).pack(side=tk.LEFT, padx=10)
            
            # Estimated approval date (1-2 years later)
            approval_days = random.randint(365, 730)
            approval_date = self.app.game_state.current_date + datetime.timedelta(days=approval_days)
            
            ctk.CTkLabel(
                status_frame,
                text=f"Est. Approval: {approval_date.strftime('%Y-%m-%d')}"
            ).pack(side=tk.RIGHT, padx=10)
            
            # Actions
            actions_frame = ctk.CTkFrame(patent_frame)
            actions_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkButton(
                actions_frame,
                text="Expedite ($25,000)",
                width=150,
                command=lambda: self.expedite_patent(patent_title)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                actions_frame,
                text="Abandon",
                width=100,
                fg_color=DANGER_COLOR,
                hover_color="#c0392b",
                command=lambda f=patent_frame: f.destroy()
            ).pack(side=tk.RIGHT, padx=10)
        
        messagebox.showinfo("Patent Filed", 
                         f"Patent application for '{patent_title}' filed successfully. "
                         f"Approval process typically takes 1-2 years.")
        
        # Clear the form
        self.patent_title.delete(0, tk.END)
        
        self.update_display()
    
    def expedite_patent(self, patent_title):
        """Expedite a patent application process"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Fixed expedite cost
        expedite_cost = 25000
        
        # Check if business has enough cash
        if expedite_cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"Expediting costs ${expedite_cost:,} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Expedite", 
                                    f"Expedite patent application for '{patent_title}'?\n\n"
                                    f"Cost: ${expedite_cost:,}\n\n"
                                    f"This will reduce the approval time by approximately 6 months.")
        
        if not confirm:
            return
            
        # Apply expedite
        business.cash -= expedite_cost
        
        messagebox.showinfo("Patent Expedited", 
                         f"Patent application for '{patent_title}' has been expedited. "
                         f"The approval time has been reduced by approximately 6 months.")
        
        self.update_display()
    
    def license_patent(self, technology, license_cost, annual_fee):
        """License a patent technology from another company"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Check if business has enough cash for the initial license
        if license_cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"Licensing requires ${license_cost:,} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm License", 
                                    f"License '{technology}' technology?\n\n"
                                    f"Initial License: ${license_cost:,}\n"
                                    f"Annual Fee: ${annual_fee:,}\n\n"
                                    f"This will give you access to advanced technology "
                                    f"that can improve your products.")
        
        if not confirm:
            return
            
        # License the technology
        business.cash -= license_cost
        
        # In a real game, you'd track ongoing licenses and their annual fees
        # For now, add to patents list if it exists
        if hasattr(self, "patents_list"):
            # Add license to the list
            license_frame = ctk.CTkFrame(self.patents_list)
            license_frame.pack(fill=tk.X, padx=5, pady=5)
            
            license_info = ctk.CTkFrame(license_frame)
            license_info.pack(fill=tk.X, padx=5, pady=2)
            
            ctk.CTkLabel(
                license_info,
                text=f"Licensed: {technology}",
                font=("Helvetica", 12, "bold")
            ).pack(side=tk.LEFT, padx=10)
            
            # License date
            license_date = self.app.game_state.current_date.strftime("%Y-%m-%d")
            
            ctk.CTkLabel(
                license_info,
                text=f"Licensed on: {license_date}"
            ).pack(side=tk.RIGHT, padx=10)
            
            # Terms
            terms_frame = ctk.CTkFrame(license_frame)
            terms_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ctk.CTkLabel(
                terms_frame,
                text=f"Initial License: ${license_cost:,}"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                terms_frame,
                text=f"Annual Fee: ${annual_fee:,}"
            ).pack(side=tk.LEFT, padx=10)
            
            # Next payment date (1 year later)
            next_payment = self.app.game_state.current_date + datetime.timedelta(days=365)
            
            ctk.CTkLabel(
                terms_frame,
                text=f"Next payment: {next_payment.strftime('%Y-%m-%d')}"
            ).pack(side=tk.RIGHT, padx=10)
            
            # Actions
            actions_frame = ctk.CTkFrame(license_frame)
            actions_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkButton(
                actions_frame,
                text="Apply to Products",
                width=150,
                command=lambda: self.apply_licensed_tech(technology)
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkButton(
                actions_frame,
                text="Terminate License",
                width=150,
                fg_color=DANGER_COLOR,
                hover_color="#c0392b",
                command=lambda f=license_frame: f.destroy()
            ).pack(side=tk.RIGHT, padx=10)
        
        messagebox.showinfo("Technology Licensed", 
                         f"Successfully licensed '{technology}' technology. "
                         f"You can now apply this technology to your products.")
        
        self.update_display()
    
    def apply_licensed_tech(self, technology):
        """Apply licensed technology to products"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business) or not business.products:
            return
            
        # Create a dialog to select which products to apply to
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Apply {technology}")
        dialog.geometry("400x400")
        dialog.transient(self)  # Make it a modal dialog
        dialog.grab_set()
        
        ctk.CTkLabel(
            dialog,
            text=f"Apply {technology} to Products",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(20, 10))
        
        # Technology description and benefits
        if "AI" in technology:
            description = "This advanced AI technology will improve product intelligence and automation capabilities."
            quality_boost = 0.15
        elif "Quantum" in technology:
            description = "Quantum computing interfaces will dramatically increase processing capabilities."
            quality_boost = 0.2
        elif "Nano" in technology:
            description = "Nano-materials will improve product durability and performance while reducing weight."
            quality_boost = 0.12
        elif "Energy" in technology:
            description = "Advanced energy storage will improve battery life and efficiency."
            quality_boost = 0.18
        elif "Cooling" in technology:
            description = "Thermal management technology will improve product reliability and performance."
            quality_boost = 0.1
        else:
            description = "This technology will improve product capabilities."
            quality_boost = 0.1
        
        ctk.CTkLabel(
            dialog,
            text=description,
            wraplength=360,
            justify="left"
        ).pack(pady=(5, 15), padx=20)
        
        ctk.CTkLabel(
            dialog,
            text=f"Quality Improvement: +{quality_boost*10:.1f} points",
            text_color=SUCCESS_COLOR,
            font=("Helvetica", 12, "bold")
        ).pack(pady=5)
        
        # Select which products to apply to
        ctk.CTkLabel(
            dialog,
            text="Select Products:",
            font=("Helvetica", 14)
        ).pack(pady=(15, 5), anchor="w", padx=20)
        
        # Scrollable product selection
        products_frame = ctk.CTkScrollableFrame(dialog, height=150)
        products_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Checkboxes for each product
        product_vars = {}
        for product in business.products:
            var = tk.BooleanVar(value=False)
            product_vars[product.id] = var
            
            product_row = ctk.CTkFrame(products_frame)
            product_row.pack(fill=tk.X, padx=5, pady=2)
            
            ctk.CTkCheckBox(
                product_row,
                text=f"{product.name} (Current quality: {product.quality*10:.1f}/10)",
                variable=var
            ).pack(side=tk.LEFT, padx=10)
        
        # Integration cost per product
        integration_cost = 200000  # $200K per product
        
        cost_label = ctk.CTkLabel(
            dialog,
            text=f"Integration Cost: ${integration_cost:,} per product"
        )
        cost_label.pack(pady=10)
        
        # Apply button
        ctk.CTkButton(
            dialog,
            text="Apply Technology",
            command=lambda: self.execute_tech_integration(
                dialog,
                technology,
                {prod_id: var.get() for prod_id, var in product_vars.items()},
                quality_boost,
                integration_cost
            ),
            fg_color=PRIMARY_COLOR
        ).pack(pady=(10, 5))
        
        # Cancel button
        ctk.CTkButton(
            dialog,
            text="Cancel",
            command=dialog.destroy,
            fg_color=SECONDARY_COLOR
        ).pack(pady=(0, 10))
    
    def execute_tech_integration(self, dialog, technology, selected_products, quality_boost, cost_per_product):
        """Execute the technology integration on selected products"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            dialog.destroy()
            return
            
        # Get selected products
        products_to_update = [p for p in business.products if selected_products.get(p.id, False)]
        
        if not products_to_update:
            messagebox.showwarning("No Products", "No products were selected to update.")
            return
            
        # Calculate total cost
        total_cost = len(products_to_update) * cost_per_product
        
        # Check if business has enough cash
        if total_cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"Integration requires {self.format_currency(total_cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Apply the integration
        business.cash -= total_cost
        
        for product in products_to_update:
            # Improve product quality
            product.quality = min(1.0, product.quality + quality_boost)
            
            # Small cost reduction benefit (5-10%)
            cost_reduction = 0.05 + (random.random() * 0.05)
            product.cost *= (1 - cost_reduction)
        
        dialog.destroy()
        
        messagebox.showinfo("Technology Integrated", 
                         f"Successfully integrated {technology} into {len(products_to_update)} products, "
                         f"improving their quality and performance.")
        
        self.update_display()
    
    def implement_tech_upgrade(self, upgrade, cost):
        """Implement a technology upgrade"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Check if business has enough cash
        if cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"This upgrade costs {self.format_currency(cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Upgrade", 
                                    f"Implement {upgrade} for {self.format_currency(cost)}?\n\n"
                                    f"This will improve your company's technological capabilities "
                                    f"and operational efficiency.")
        
        if not confirm:
            return
            
        # Implement the upgrade
        business.cash -= cost
        
        # Different benefits based on upgrade type
        if "ERP" in upgrade:
            message = "Enterprise Resource Planning system implementation will streamline operations and improve efficiency."
            efficiency_gain = 0.15
        elif "Automation" in upgrade:
            message = "Manufacturing automation will increase production efficiency and reduce labor costs."
            efficiency_gain = 0.2
        elif "AI" in upgrade:
            message = "AI systems will improve decision making and operational optimization."
            efficiency_gain = 0.18
        elif "Cybersecurity" in upgrade:
            message = "Enhanced security systems will protect your digital assets and reduce risk."
            efficiency_gain = 0.1
        elif "Cloud" in upgrade:
            message = "Cloud infrastructure will improve scalability and reduce IT costs."
            efficiency_gain = 0.12
        elif "Data" in upgrade:
            message = "Big Data Analytics will provide better insights for strategic decisions."
            efficiency_gain = 0.15
        else:
            message = "Technology upgrade successfully implemented, improving company capabilities."
            efficiency_gain = 0.1
        
        # In a real game, these would update various aspects of the business
        # For now, just show a success message
        messagebox.showinfo("Upgrade Implemented", message)
        
        self.update_display()
    
    def launch_innovation_initiative(self, initiative, cost):
        """Launch an innovation initiative"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Check if business has enough cash
        if cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"This initiative costs {self.format_currency(cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Initiative", 
                                    f"Launch {initiative} for {self.format_currency(cost)}?\n\n"
                                    f"This will enhance your company's innovation capabilities.")
        
        if not confirm:
            return
            
        # Launch the initiative
        business.cash -= cost
        
        # Different benefits based on initiative type
        if "Lab" in initiative:
            message = "Innovation Lab established, providing dedicated facilities for experimental research."
        elif "Open Innovation" in initiative:
            message = "Open Innovation program launched, creating a network with startups and academia."
        elif "Incubator" in initiative:
            message = "Idea Incubator program established to nurture employee innovations."
        elif "Contest" in initiative:
            message = "Innovation Contest launched, encouraging breakthrough ideas from all sources."
        elif "Partnerships" in initiative:
            message = "Research Partnerships formed with leading research institutions."
        else:
            message = "Innovation initiative successfully launched."
        
        # In a real game, these would provide different boosts to innovation metrics
        messagebox.showinfo("Initiative Launched", message)
        
        self.update_display()
    
    def build_new_facility(self):
        """Build a new business facility"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Create a dialog for facility selection
        dialog = ctk.CTkToplevel(self)
        dialog.title("Build New Facility")
        dialog.geometry("500x400")
        dialog.transient(self)  # Make it a modal dialog
        dialog.grab_set()
        
        ctk.CTkLabel(
            dialog,
            text="Build New Facility",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(20, 10))
        
        # Facility type
        type_frame = ctk.CTkFrame(dialog)
        type_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ctk.CTkLabel(type_frame, text="Facility Type:").pack(side=tk.LEFT, padx=10)
        facility_type = ctk.CTkOptionMenu(
            type_frame,
            values=[
                "Factory",
                "R&D Center",
                "Office",
                "Warehouse",
                "Retail Store"
            ]
        )
        facility_type.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Location
        location_frame = ctk.CTkFrame(dialog)
        location_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ctk.CTkLabel(location_frame, text="Country:").pack(side=tk.LEFT, padx=10)
        
        # Get list of countries from business existing locations or add headquarter country
        countries = list(business.factories.keys()) + list(business.retail_stores.keys())
        if business.headquarters_country and business.headquarters_country not in countries:
            countries.append(business.headquarters_country)
            
        # Add some other major countries as options
        additional_countries = ["United States", "China", "Germany", "Japan", "United Kingdom"]
        for country in additional_countries:
            if country not in countries:
                countries.append(country)
        
        country_var = tk.StringVar(value=business.headquarters_country)
        country_menu = ctk.CTkOptionMenu(
            location_frame,
            values=countries,
            variable=country_var
        )
        country_menu.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Size
        size_frame = ctk.CTkFrame(dialog)
        size_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ctk.CTkLabel(size_frame, text="Size:").pack(side=tk.LEFT, padx=10)
        facility_size = ctk.CTkOptionMenu(
            size_frame,
            values=[
                "Small",
                "Medium",
                "Large"
            ]
        )
        facility_size.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Cost calculation
        cost_frame = ctk.CTkFrame(dialog)
        cost_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ctk.CTkLabel(cost_frame, text="Estimated Cost:").pack(side=tk.LEFT, padx=10)
        cost_label = ctk.CTkLabel(cost_frame, text="$10,000,000")
        cost_label.pack(side=tk.LEFT, padx=10)
        
        # Calculate cost based on selections
        def update_cost(*args):
            fac_type = facility_type.get()
            size = facility_size.get()
            country = country_var.get()
            
            # Base costs by type
            if fac_type == "Factory":
                base_cost = 10000000  # $10M
            elif fac_type == "R&D Center":
                base_cost = 8000000   # $8M
            elif fac_type == "Office":
                base_cost = 5000000   # $5M
            elif fac_type == "Warehouse":
                base_cost = 3000000   # $3M
            else:  # Retail Store
                base_cost = 1000000   # $1M
                
            # Adjust for size
            if size == "Small":
                size_mult = 0.7
            elif size == "Medium":
                size_mult = 1.0
            else:  # Large
                size_mult = 1.8
                
            # Adjust for country
            country_multipliers = {
                "United States": 1.2,
                "Germany": 1.3,
                "Japan": 1.25,
                "United Kingdom": 1.15,
                "China": 0.8,
                "India": 0.7,
                "Mexico": 0.75,
                "Brazil": 0.85
            }
            
            country_mult = country_multipliers.get(country, 1.0)
            
            # Calculate final cost
            final_cost = base_cost * size_mult * country_mult
            cost_label.configure(text=f"${final_cost:,.2f}")
            
            # Store cost for build button
            dialog.final_cost = final_cost
        
        # Bind the cost calculation to selection changes
        facility_type.configure(command=update_cost)
        facility_size.configure(command=update_cost)
        country_menu.configure(command=update_cost)
        
        # Initial cost calculation
        update_cost()
        
        # Estimated completion time
        time_frame = ctk.CTkFrame(dialog)
        time_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ctk.CTkLabel(time_frame, text="Construction Time:").pack(side=tk.LEFT, padx=10)
        
        # Different types have different construction times
        def update_time(*args):
            fac_type = facility_type.get()
            size = facility_size.get()
            
            # Base times by type (in months)
            if fac_type == "Factory":
                base_time = 12  # 12 months
            elif fac_type == "R&D Center":
                base_time = 9   # 9 months
            elif fac_type == "Office":
                base_time = 8   # 8 months
            elif fac_type == "Warehouse":
                base_time = 6   # 6 months
            else:  # Retail Store
                base_time = 3   # 3 months
                
            # Adjust for size
            if size == "Small":
                size_mult = 0.8
            elif size == "Medium":
                size_mult = 1.0
            else:  # Large
                size_mult = 1.4
                
            # Calculate final time
            time_months = round(base_time * size_mult)
            time_label.configure(text=f"{time_months} months")
            
            # Store time for build button
            dialog.construction_time = time_months
        
        # Bind the time calculation to selection changes
        facility_type.configure(command=update_time)
        facility_size.configure(command=update_time)
        
        time_label = ctk.CTkLabel(time_frame, text="12 months")
        time_label.pack(side=tk.LEFT, padx=10)
        
        # Initial time calculation
        update_time()
        
        # Build button
        ctk.CTkButton(
            dialog,
            text="Build Facility",
            command=lambda: self.confirm_facility_build(
                dialog, 
                facility_type.get(), 
                country_var.get(), 
                facility_size.get(), 
                dialog.final_cost,
                dialog.construction_time
            ),
            fg_color=PRIMARY_COLOR
        ).pack(pady=10)
    
    def confirm_facility_build(self, dialog, facility_type, country, size, cost, construction_time):
        """Confirm and execute facility building"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            dialog.destroy()
            return
            
        # Check if business has enough cash
        if cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"Building this facility requires {self.format_currency(cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Build the facility
        business.cash -= cost
        business.total_assets += cost * 0.9  # 90% of cost becomes an asset
        
        # Add to appropriate dictionary based on facility type
        if facility_type == "Factory":
            if country in business.factories:
                business.factories[country] += 1
            else:
                business.factories[country] = 1
        elif facility_type == "Retail Store":
            if country in business.retail_stores:
                business.retail_stores[country] += 1
            else:
                business.retail_stores[country] = 1
        
        # In a real game, you'd track the other facility types too
        
        # Close dialog
        dialog.destroy()
        
        # Show success message
        messagebox.showinfo("Facility Built", 
                         f"Successfully started construction of a {size} {facility_type} in {country}. "
                         f"It will be operational in {construction_time} months.")
        
        self.update_display()
    
    def upgrade_facility(self):
        """Upgrade an existing facility"""
        messagebox.showinfo("Facility Upgrade", "Facility upgrade feature coming in the next update!")
    
    def close_facility(self):
        """Close an existing facility"""
        messagebox.showinfo("Close Facility", "Facility closure feature coming in the next update!")
    
    def add_supplier(self):
        """Add a new supplier"""
        messagebox.showinfo("Add Supplier", "Supplier management feature coming in the next update!")
    
    def optimize_supply_chain(self):
        """Optimize the supply chain"""
        messagebox.showinfo("Optimize Supply Chain", "Supply chain optimization feature coming in the next update!")
    
    def manage_inventory(self):
        """Manage inventory levels"""
        messagebox.showinfo("Manage Inventory", "Inventory management feature coming in the next update!")
    
    def hire_employees(self):
        """Hire new employees"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        try:
            # Get hiring details
            count = int(self.hiring_count.get())
            employee_type = self.hiring_type.get()
            salary = float(self.hiring_salary.get())
            
            if count <= 0:
                raise ValueError("Number of employees must be positive")
                
            if salary <= 0:
                raise ValueError("Salary must be positive")
                
            # Calculate total annual cost
            annual_cost = count * salary
            
            # Check if business can afford it (use 25% of cash as a limit)
            if annual_cost > business.cash * 0.25:
                messagebox.showerror("Insufficient Funds", 
                                   f"Annual cost of ${annual_cost:,.2f} for these employees "
                                   f"exceeds 25% of your available cash ({self.format_currency(business.cash * 0.25)}).")
                return
                
            # Ask for confirmation
            confirm = messagebox.askyesno("Confirm Hiring", 
                                       f"Hire {count} {employee_type} employees?\n\n"
                                       f"Salary: ${salary:,.2f} each\n"
                                       f"Annual Cost: ${annual_cost:,.2f}")
            
            if not confirm:
                return
                
            # Hire employees
            success = business.hire_employees(count, salary)
            
            if success:
                messagebox.showinfo("Employees Hired", 
                                 f"Successfully hired {count} {employee_type} employees.")
                
                self.update_display()
            else:
                messagebox.showerror("Hiring Failed", "Failed to complete the hiring process.")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def implement_benefit(self, benefit, cost_per_employee):
        """Implement an employee benefit"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Calculate total cost
        total_cost = cost_per_employee * business.employees
        
        # Check if business has enough cash
        if total_cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"This benefit costs {self.format_currency(total_cost)} "
                               f"for your {business.employees:,} employees, "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Benefit", 
                                    f"Implement {benefit} for all employees?\n\n"
                                    f"Cost per Employee: ${cost_per_employee:,.2f}\n"
                                    f"Total Cost: ${total_cost:,.2f}\n\n"
                                    f"This will improve employee satisfaction and productivity.")
        
        if not confirm:
            return
            
        # Implement the benefit
        business.cash -= total_cost
        
        messagebox.showinfo("Benefit Implemented", 
                         f"Successfully implemented {benefit} for all employees, "
                         f"improving satisfaction and productivity.")
        
        self.update_display()
    
    def calculate_expansion_cost(self):
        """Calculate the cost of international expansion"""
        try:
            # Get values
            factories = int(self.expansion_factories.get())
            stores = int(self.expansion_stores.get())
            offices = int(self.expansion_offices.get())
            country = self.expansion_country.get()
            
            # Calculate costs
            factory_cost = factories * 10000000  # $10M per factory
            store_cost = stores * 1000000        # $1M per store
            office_cost = offices * 3000000      # $3M per office
            
            # Adjust for country
            country_multipliers = {
                "United States": 1.2,
                "Germany": 1.3,
                "Japan": 1.25,
                "United Kingdom": 1.15,
                "China": 0.8,
                "India": 0.7,
                "Mexico": 0.75,
                "Brazil": 0.85
            }
            
            country_mult = country_multipliers.get(country, 1.0)
            
            # Calculate total cost
            total_cost = (factory_cost + store_cost + office_cost) * country_mult
            
            # Update label
            self.expansion_investment.configure(text=self.format_currency(total_cost))
            
            # Store for later use
            self.last_expansion_cost = total_cost
        except ValueError:
            self.expansion_investment.configure(text="Invalid input")
    
    def expand_to_country(self):
        """Expand business to a new country"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        try:
            # Calculate cost if not already done
            if not hasattr(self, "last_expansion_cost"):
                self.calculate_expansion_cost()
            
            # Get values
            factories = int(self.expansion_factories.get())
            stores = int(self.expansion_stores.get())
            offices = int(self.expansion_offices.get())
            country = self.expansion_country.get()
            
            # Check if business has enough cash
            if self.last_expansion_cost > business.cash:
                messagebox.showerror("Insufficient Funds", 
                                   f"This expansion requires {self.format_currency(self.last_expansion_cost)} "
                                   f"but you only have {self.format_currency(business.cash)}")
                return
                
            # Check that at least one facility is being built
            if factories + stores + offices == 0:
                messagebox.showerror("No Facilities", "You need to build at least one facility.")
                return
                
            # Ask for confirmation
            confirm = messagebox.askyesno("Confirm Expansion", 
                                       f"Expand to {country}?\n\n"
                                       f"Factories: {factories}\n"
                                       f"Retail Stores: {stores}\n"
                                       f"Offices: {offices}\n"
                                       f"Total Cost: {self.format_currency(self.last_expansion_cost)}")
            
            if not confirm:
                return
                
            # Execute expansion
            success = business.expand_to_country(
                country, factories, stores, 
                factory_cost=10000000, store_cost=1000000
            )
            
            if success:
                messagebox.showinfo("Expansion Successful", 
                                 f"Successfully expanded operations to {country}.")
                
                # Add to markets list if it exists
                if hasattr(self, "markets_list"):
                    # Remove "no markets" message if it's the first market
                    for widget in self.markets_list.winfo_children():
                        if isinstance(widget, ctk.CTkLabel) and "No markets" in widget.cget("text"):
                            widget.destroy()
                    
                    # Add market to the list
                    market_frame = ctk.CTkFrame(self.markets_list)
                    market_frame.pack(fill=tk.X, padx=5, pady=5)
                    
                    ctk.CTkLabel(
                        market_frame,
                        text=country,
                        font=("Helvetica", 12, "bold"),
                        width=150
                    ).pack(side=tk.LEFT, padx=10)
                    
                    facilities_text = []
                    if factories > 0:
                        facilities_text.append(f"{factories} factories")
                    if stores > 0:
                        facilities_text.append(f"{stores} stores")
                    if offices > 0:
                        facilities_text.append(f"{offices} offices")
                        
                    facilities_str = ", ".join(facilities_text)
                    
                    ctk.CTkLabel(
                        market_frame,
                        text=facilities_str
                    ).pack(side=tk.LEFT, padx=10)
                    
                    # Market share (randomly generated)
                    market_share = round(random.uniform(0.5, 3.0), 1)
                    
                    ctk.CTkLabel(
                        market_frame,
                        text=f"Market Share: {market_share}%"
                    ).pack(side=tk.RIGHT, padx=10)
                
                self.update_display()
            else:
                messagebox.showerror("Expansion Failed", "Failed to complete the expansion.")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def diversify_business(self, sector, cost):
        """Diversify business into a new sector"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Check if business has enough cash
        if cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"This diversification requires {self.format_currency(cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Check if already in this sector
        if sector in business.sectors:
            messagebox.showinfo("Already Present", f"Your business is already operating in the {sector} sector.")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Diversification", 
                                    f"Diversify into {sector} sector?\n\n"
                                    f"Cost: {self.format_currency(cost)}\n\n"
                                    f"This will open new markets and revenue streams but requires "
                                    f"significant investment and management attention.")
        
        if not confirm:
            return
            
        # Execute diversification
        business.cash -= cost
        business.sectors.append(sector)
        
        messagebox.showinfo("Diversification Successful", 
                         f"Successfully diversified into the {sector} sector. "
                         f"You can now develop products in this area.")
        
        self.update_display()
    
    def show_acquisition_targets(self):
        """Show potential acquisition targets"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
        
        # Create a dialog with acquisition targets
        dialog = ctk.CTkToplevel(self)
        dialog.title("Acquisition Targets")
        dialog.geometry("700x500")
        dialog.transient(self)  # Make it a modal dialog
        dialog.grab_set()
        
        ctk.CTkLabel(
            dialog,
            text="Potential Acquisition Targets",
            font=("Helvetica", 18, "bold")
        ).pack(pady=(20, 10))
        
        # Create scrollable list of targets
        targets_frame = ctk.CTkScrollableFrame(dialog)
        targets_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Generate some fictional acquisition targets
        acquisition_targets = [
            {
                "name": "TechInnovate Solutions",
                "industry": "Technology",
                "employees": 120,
                "revenue": 15000000,
                "profit": 2000000,
                "price": 45000000,
                "synergy": "High"
            },
            {
                "name": "GlobalRetail Systems",
                "industry": "Retail",
                "employees": 350,
                "revenue": 30000000,
                "profit": 3500000,
                "price": 80000000,
                "synergy": "Medium"
            },
            {
                "name": "MicroManufacturing Co.",
                "industry": "Manufacturing",
                "employees": 200,
                "revenue": 25000000,
                "profit": 1800000,
                "price": 40000000,
                "synergy": "High"
            },
            {
                "name": "DataInsight Analytics",
                "industry": "Technology",
                "employees": 75,
                "revenue": 8000000,
                "profit": 1200000,
                "price": 30000000,
                "synergy": "Medium"
            },
            {
                "name": "EcoProducts Inc.",
                "industry": "Consumer Goods",
                "employees": 150,
                "revenue": 18000000,
                "profit": 1500000,
                "price": 35000000,
                "synergy": "Low"
            }
        ]
        
        # Add each target to the list
        for target in acquisition_targets:
            target_frame = ctk.CTkFrame(targets_frame)
            target_frame.pack(fill=tk.X, padx=5, pady=10)
            
            # Company name and industry
            header_frame = ctk.CTkFrame(target_frame)
            header_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkLabel(
                header_frame,
                text=target["name"],
                font=("Helvetica", 14, "bold")
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                header_frame,
                text=f"Industry: {target['industry']}"
            ).pack(side=tk.LEFT, padx=10)
            
            # Company statistics
            stats_frame = ctk.CTkFrame(target_frame)
            stats_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ctk.CTkLabel(
                stats_frame,
                text=f"Employees: {target['employees']}"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                stats_frame,
                text=f"Annual Revenue: {self.format_currency(target['revenue'])}"
            ).pack(side=tk.LEFT, padx=10)
            
            ctk.CTkLabel(
                stats_frame,
                text=f"Annual Profit: {self.format_currency(target['profit'])}"
            ).pack(side=tk.LEFT, padx=10)
            
            # Acquisition details
            details_frame = ctk.CTkFrame(target_frame)
            details_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctk.CTkLabel(
                details_frame,
                text=f"Asking Price: {self.format_currency(target['price'])}"
            ).pack(side=tk.LEFT, padx=10)
            
            # Calculate rough multiples
            revenue_multiple = target["price"] / target["revenue"]
            pe_ratio = target["price"] / target["profit"] if target["profit"] > 0 else float('inf')
            
            ctk.CTkLabel(
                details_frame,
                text=f"Valuation: {revenue_multiple:.1f}x Revenue, {pe_ratio:.1f}x Earnings"
            ).pack(side=tk.LEFT, padx=10)
            
            # Synergy indicator
            synergy_color = SUCCESS_COLOR if target["synergy"] == "High" else \
                          WARNING_COLOR if target["synergy"] == "Medium" else \
                          DANGER_COLOR
            
            ctk.CTkLabel(
                details_frame,
                text=f"Synergy: {target['synergy']}",
                text_color=synergy_color
            ).pack(side=tk.RIGHT, padx=10)
            
            # Acquire button
            ctk.CTkButton(
                target_frame,
                text="Acquire Company",
                width=150,
                command=lambda t=target: self.acquire_company(dialog, t)
            ).pack(pady=10)
    
    def acquire_company(self, dialog, target):
        """Acquire a company"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            dialog.destroy()
            return
        
        # Check if business has enough cash
        if target["price"] > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"This acquisition requires {self.format_currency(target['price'])} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Acquisition", 
                                    f"Acquire {target['name']} for {self.format_currency(target['price'])}?\n\n"
                                    f"This will add {target['employees']} employees and "
                                    f"{self.format_currency(target['revenue'])} in annual revenue to your business.")
        
        if not confirm:
            return
            
        # Execute the acquisition
        business.cash -= target["price"]
        business.employees += target["employees"]
        business.total_assets += target["price"] * 0.8  # 80% of price becomes assets
        
        # Add industry to sectors if not already present
        if target["industry"] not in business.sectors:
            business.sectors.append(target["industry"])
        
        # Close dialog
        dialog.destroy()
        
        # Show success message
        messagebox.showinfo("Acquisition Complete", 
                         f"Successfully acquired {target['name']}. "
                         f"Integration process will take several months.")
        
        self.update_display()
    
    def execute_competitive_action(self, action):
        """Execute a competitive action against a competitor"""
        business = self.app.game_state.get_player_entity()
        if not business or not isinstance(business, Business):
            return
            
        # Get selected competitor
        competitor = self.selected_competitor.get()
        
        if competitor == "Select a competitor":
            messagebox.showerror("No Competitor", "Please select a competitor first")
            return
            
        # Different costs and effects based on action type
        if action == "Price War":
            cost = business.cash * 0.15  # 15% of cash
            risk = "High"
            success_chance = 0.6
        elif action == "Aggressive Marketing":
            cost = business.cash * 0.1  # 10% of cash
            risk = "Medium"
            success_chance = 0.7
        elif action == "Product Innovation":
            cost = business.cash * 0.08  # 8% of cash
            risk = "Low"
            success_chance = 0.8
        elif action == "Talent Acquisition":
            cost = business.cash * 0.05  # 5% of cash
            risk = "Medium"
            success_chance = 0.65
        elif action == "Market Expansion":
            cost = business.cash * 0.12  # 12% of cash
            risk = "Medium"
            success_chance = 0.75
        else:
            cost = business.cash * 0.1
            risk = "Medium"
            success_chance = 0.7
        
        # Check if business has enough cash
        if cost > business.cash:
            messagebox.showerror("Insufficient Funds", 
                               f"This action requires {self.format_currency(cost)} "
                               f"but you only have {self.format_currency(business.cash)}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm Action", 
                                    f"Execute {action} against {competitor}?\n\n"
                                    f"Cost: {self.format_currency(cost)}\n"
                                    f"Risk Level: {risk}\n"
                                    f"Success Chance: {success_chance*100:.0f}%")
        
        if not confirm:
            return
            
        # Execute the action
        business.cash -= cost
        
        # Determine success (random chance)
        success = random.random() < success_chance
        
        if success:
            messagebox.showinfo("Action Successful", 
                             f"The {action} against {competitor} was successful! "
                             f"You've gained market share and competitive advantage.")
        else:
            messagebox.showinfo("Action Failed", 
                             f"The {action} against {competitor} failed to achieve the desired results. "
                             f"The investment did not pay off as expected.")
        
        self.update_display()
    
    def format_currency(self, value):
        """Format a currency value based on size"""
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        else:
            return f"${value:.2f}"

class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.app = master
        
        self.title("Settings")
        self.geometry("400x500")
        self.transient(self.app)  # Make it a modal dialog
        self.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        ctk.CTkLabel(
            self,
            text="Game Settings",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(20, 10))
        
        # Settings container with tabs
        tabview = ctk.CTkTabview(self)
        tabview.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Add tabs
        tabview.add("Display")
        tabview.add("Game")
        tabview.add("Audio")
        tabview.add("About")
        
        # Display settings
        display_frame = ctk.CTkFrame(tabview.tab("Display"))
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Theme selection
        theme_frame = ctk.CTkFrame(display_frame)
        theme_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(theme_frame, text="UI Theme:").pack(side=tk.LEFT, padx=10)
        
        theme_var = tk.StringVar(value=ctk.get_appearance_mode())
        
        ctk.CTkRadioButton(
            theme_frame,
            text="Light",
            variable=theme_var,
            value="Light",
            command=self.change_theme
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkRadioButton(
            theme_frame,
            text="Dark",
            variable=theme_var,
            value="Dark",
            command=self.change_theme
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkRadioButton(
            theme_frame,
            text="System",
            variable=theme_var,
            value="System",
            command=self.change_theme
        ).pack(side=tk.LEFT, padx=10)
        
        # UI scaling
        scale_frame = ctk.CTkFrame(display_frame)
        scale_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(scale_frame, text="UI Scaling:").pack(side=tk.LEFT, padx=10)
        
        scale_var = ctk.DoubleVar(value=1.0)
        
        scale_slider = ctk.CTkSlider(
            scale_frame,
            from_=0.5,
            to=2.0,
            number_of_steps=15,
            variable=scale_var
        )
        scale_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        scale_label = ctk.CTkLabel(scale_frame, text="100%")
        scale_label.pack(side=tk.LEFT, padx=10)
        
        # Update scale label when slider moves
        def update_scale_label(value):
            scale_label.configure(text=f"{int(value*100)}%")
            
        scale_slider.configure(command=update_scale_label)
        
        # Apply scaling button
        ctk.CTkButton(
            scale_frame,
            text="Apply",
            width=80,
            command=lambda: self.change_scaling(scale_var.get())
        ).pack(side=tk.RIGHT, padx=10)
        
        # Game settings
        game_frame = ctk.CTkFrame(tabview.tab("Game"))
        game_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Difficulty selection
        difficulty_frame = ctk.CTkFrame(game_frame)
        difficulty_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(difficulty_frame, text="Game Difficulty:").pack(side=tk.LEFT, padx=10)
        
        difficulty_var = tk.StringVar(value="Normal")
        
        ctk.CTkRadioButton(
            difficulty_frame,
            text="Easy",
            variable=difficulty_var,
            value="Easy"
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkRadioButton(
            difficulty_frame,
            text="Normal",
            variable=difficulty_var,
            value="Normal"
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkRadioButton(
            difficulty_frame,
            text="Hard",
            variable=difficulty_var,
            value="Hard"
        ).pack(side=tk.LEFT, padx=10)
        
        # Game speed default
        speed_frame = ctk.CTkFrame(game_frame)
        speed_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(speed_frame, text="Default Game Speed:").pack(side=tk.LEFT, padx=10)
        
        speed_var = tk.IntVar(value=1)
        
        ctk.CTkRadioButton(
            speed_frame,
            text="1×",
            variable=speed_var,
            value=1
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkRadioButton(
            speed_frame,
            text="2×",
            variable=speed_var,
            value=2
        ).pack(side=tk.LEFT, padx=10)
        
        ctk.CTkRadioButton(
            speed_frame,
            text="5×",
            variable=speed_var,
            value=5
        ).pack(side=tk.LEFT, padx=10)
        
        # Tutorial toggle
        tutorial_frame = ctk.CTkFrame(game_frame)
        tutorial_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(tutorial_frame, text="Show Tutorial:").pack(side=tk.LEFT, padx=10)
        
        tutorial_var = tk.BooleanVar(value=True)
        
        tutorial_switch = ctk.CTkSwitch(
            tutorial_frame,
            text="",
            variable=tutorial_var
        )
        tutorial_switch.pack(side=tk.LEFT, padx=10)
        
        # Save settings button
        ctk.CTkButton(
            game_frame,
            text="Apply Game Settings",
            command=self.apply_game_settings
        ).pack(pady=20)
        
        # Audio settings
        audio_frame = ctk.CTkFrame(tabview.tab("Audio"))
        audio_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Master volume
        volume_frame = ctk.CTkFrame(audio_frame)
        volume_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(volume_frame, text="Master Volume:").pack(side=tk.LEFT, padx=10)
        
        volume_var = ctk.DoubleVar(value=0.7)
        
        volume_slider = ctk.CTkSlider(
            volume_frame,
            from_=0,
            to=1,
            number_of_steps=10,
            variable=volume_var
        )
        volume_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        volume_label = ctk.CTkLabel(volume_frame, text="70%")
        volume_label.pack(side=tk.LEFT, padx=10)
        
        # Update volume label when slider moves
        def update_volume_label(value):
            volume_label.configure(text=f"{int(value*100)}%")
            
        volume_slider.configure(command=update_volume_label)
        
        # Sound effects toggle
        sfx_frame = ctk.CTkFrame(audio_frame)
        sfx_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(sfx_frame, text="Sound Effects:").pack(side=tk.LEFT, padx=10)
        
        sfx_var = tk.BooleanVar(value=True)
        
        sfx_switch = ctk.CTkSwitch(
            sfx_frame,
            text="",
            variable=sfx_var
        )
        sfx_switch.pack(side=tk.LEFT, padx=10)
        
        # Music toggle
        music_frame = ctk.CTkFrame(audio_frame)
        music_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(music_frame, text="Background Music:").pack(side=tk.LEFT, padx=10)
        
        music_var = tk.BooleanVar(value=True)
        
        music_switch = ctk.CTkSwitch(
            music_frame,
            text="",
            variable=music_var
        )
        music_switch.pack(side=tk.LEFT, padx=10)
        
        # About tab
        about_frame = ctk.CTkFrame(tabview.tab("About"))
        about_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            about_frame,
            text=f"{APP_NAME} {APP_VERSION}",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        ctk.CTkLabel(
            about_frame,
            text="A complex economic + business simulator.",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        ctk.CTkLabel(
            about_frame,
            text="© 2025 myEconomy",
            font=("Helvetica", 10)
        ).pack(pady=20)
        
        # Credits
        credits_text = """
        Thank you to everyone that has played myEconomy!
        We are currently focusing on fixing new bugs before implementing new feature :D
        """
        
        credits = ctk.CTkTextbox(about_frame, height=150)
        credits.pack(fill=tk.X, padx=10, pady=10)
        credits.insert("1.0", credits_text)
        credits.configure(state="disabled")
        
        # Close button
        close_button = ctk.CTkButton(
            self,
            text="Close",
            command=self.destroy
        )
        close_button.pack(pady=10)
    
    def change_theme(self):
        """Change the UI theme"""
        theme = ctk.get_appearance_mode()
        ctk.set_appearance_mode(theme)
    
    def change_scaling(self, scale_factor):
        """Change the UI scaling factor"""
        try:
            ctk.set_widget_scaling(scale_factor)
            messagebox.showinfo("UI Scaling", f"UI scale changed to {int(scale_factor*100)}%")
        except Exception as e:
            messagebox.showerror("Scaling Error", str(e))
    
    def apply_game_settings(self):
        """Apply various game settings"""
        messagebox.showinfo("Settings Applied", "Game settings have been updated.")

def main():
    app = MyEconomyApp()
    app.mainloop()

if __name__ == "__main__":
    main()
