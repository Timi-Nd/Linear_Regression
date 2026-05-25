# My Journey into Machine Learning

**Author: Timi Ndubuisi**

This guide explains my first steps into understanding how math and algorithms can solve real-world problems. It documents my process of learning Linear Regression using Saskatoon's infrastructure data.

---

# Saskatoon Road Project: Simple Guide

This guide explains what this project is, how the data works, and what I found. It’s designed to be easy to read for anyone interested in how computers can help manage a city.

---

## 1. What is this project?
I built a "Predictive Maintenance System" for the City of Saskatoon. 
*   **The Problem**: Roads wear out at different speeds. It’s hard to know which ones to fix first.
*   **The Solution**: I used a computer model (Machine Learning) to look at road data (like age, traffic, and lanes) to predict which roads are under the most stress.

---

## 2. The Data
I combined four different city files to create one master list. All data was downloaded from the **City of Saskatoon Open Data Portal**.

### What’s in the data?
*   **Road Age**: How old the street is (older roads usually need more help).
*   **Traffic Volume**: How many cars use the road every day.
*   **Lane Count**: More lanes usually mean more traffic and faster wear.
*   **Speed Limit**: Faster roads often face different types of stress.
*   **Road Type**: Is it a big highway or a small residential street?
*   **Bus Routes**: Does a bus run on this street? (Buses are heavy and add more wear).

### How I cleaned it:
*   I fixed impossible dates (like a road built in the year 9999).
*   I filled in missing info with "best guesses" based on other roads.
*   I matched streets to bus routes by looking for street names inside the route descriptions.

---

## 3. What the Model Found
After the computer studied the data, here is what it learned:

*   **Lanes are the Big Driver**: The number of lanes is the biggest clue for traffic volume. More lanes = more cars = more wear.
*   **Age Matters**: Older streets consistently carry a heavy load, especially in established neighborhoods.
*   **Buses add Stress**: Streets with transit routes are predicted to have more complex maintenance needs.

### Looking at the Charts:
*   **Feature Importance**: This shows which factors (like lanes or age) the computer cared about most.
*   **Actual vs. Predicted**: This shows how "right" the computer was. If the dots are close to the line, the computer did a good job!
*   **Residuals (The Errors)**: This shows us where the computer was wrong. These "errors" are actually useful—they tell us which streets are behaving strangely and might need a human to go look at them.

---

## 4. Why this isn't perfect
No computer model is 100% right. Here’s where this one could be better:
*   **It assumes "straight lines"**: The model thinks if you double the traffic, you double the wear. In real life, wear can happen slowly for years and then everything breaks at once.
*   **It doesn't see everything**: It doesn't know about the weather (like a really bad Saskatoon winter) or if a road was recently closed for construction.
*   **Limited Bus Data**: I only matched about 12% of the bus routes. If I had better maps, the model would be smarter.

---

## 5. Why is this useful anyway?
Even with its flaws, this tool is great for **"Triage"**. 
Instead of trying to check 4,000 streets one by one, a city engineer can use this list to find the top 50 "High Risk" streets and start there. It saves time, money, and makes the roads safer for everyone.

---

## 💡 Final Summary
This project shows that I can take messy city data and turn it into a smart tool. It’s not about replacing humans—it’s about giving humans a better "map" to make smarter decisions for Saskatoon.
