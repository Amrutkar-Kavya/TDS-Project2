#!/bin/bash

# Simple Interest Calculator

echo "Enter the principal amount: "
read principal

echo "Enter the annual interest rate (%): "
read rate

echo "Enter the time period in years: "
read time

# Calculate Simple Interest using bc for floating point math
interest=$(echo "scale=2; ($principal * $rate * $time) / 100" | bc)

echo "Simple Interest = ₹$interest"
