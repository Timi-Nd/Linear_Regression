# My Journey into Machine Learning

**Author: Timi Ndubuisi**

This guide explains my first steps into understanding how math and algorithms can solve real-world problems. It documents my process of learning Linear Regression using Saskatoon's infrastructure data.

---

# Saskatoon Road Project: Simple Guide

This guide explains what this project is, how the data works, and what I found. It’s designed to be easy to read for anyone interested in how computers can help manage a city.

---

## 1. What is this project?
I built this project to practice **Data Analysis and Machine Learning**. 
*   **The Problem**: It is easy to look at a spreadsheet and make a quick decision. But if your analysis is shallow, your decision will be wrong.
*   **The Goal**: I am using Saskatoon’s road data to see if a computer can help us understand infrastructure—and to prove why we still need human experience to check the computer's work.

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
*   **Age and Traffic**: Older streets generally carry *less* traffic volume, but they face a different kind of stress—long-term structural neglect.
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
Instead of trying to check 4,000+ streets one by one, a city engineer can use this list to find the top 50 "High Risk" streets and start there. It saves time, money, and makes the roads safer for everyone.

---

## 💡 The Big Lesson: Misguided Decisions
This project taught me a very important lesson in ML: **The data doesn't always tell the whole story.**

If I followed the computer blindly, I would say: *"Old roads have less traffic, so they aren't a priority."* 

**This would be a misguided decision.** Why? Because an old road with low traffic can still be a safety hazard if it is neglected. This project proves that my goal as a data analyst is not just to run models, but to understand the reality behind the numbers.


---

## 6. From Math to Reality: How this helps
I know what you are thinking: "How does a computer's guess help a real city?"

### How these insights translate to the real world:
*   **The "High-Stress" Map**: My model found that Lane Count and Road Surface Type are significant clues for traffic. In the real world, this means our newest, widest roads are working the hardest. I can now create a list of streets where the "Predicted Traffic" is much higher than what the road was built for.
*   **Finding the "Quiet Giants"**: Some roads might have very old pavement but low traffic. My model helps identify these so I don't waste money over-engineering a quiet residential street just because it is old—while still keeping an eye on their maintenance needs.

### How this specifically helps the City:
*   **Budgeting (Smart Spending)**: Instead of spreading the budget thin across every street, the City can use my `Saskatoon_Predictions.csv` to see exactly which segments are under the most pressure.
*   **Safety First**: By predicting traffic intensity, I can highlight where wear-and-tear is likely to cause potholes or cracks before they happen, making the roads safer for families and bus drivers.
*   **Objective Decisions**: It takes the guesswork out of maintenance. Instead of picking a street because of a complaint, the City can pick a street because the math shows it is a priority.

---

## 7. The "Neglect Gap": Why some old roads feel so bad
You might notice that many old roads in Saskatoon are neglected or in terrible shape, even if the math says they have less traffic. This is a very important point!

### Usage vs. Condition:
*   **Usage (What my model sees)**: This is how many cars drive on the road. Newer highways have more cars.
*   **Condition (What you see)**: This is the physical state of the road (potholes, cracks). An old road might have low traffic, but because it hasn't been touched in 40 years, it is still falling apart.

### How my model helps find these neglected streets:
By looking at the **Residuals** (the "Prediction Errors"), I can find streets where the math doesn't add up. If a road is very old and the model predicts it should be quiet, but it is actually starting to fail, that's a sign of **Neglect**. My tool helps the City see these "hidden" problems by highlighting where a road's physical age is finally catching up to it, regardless of how many cars are on it.

---

## 8. Deep Dive: What is a Heatmap?
To understand my model better, I used a **Correlation Heatmap**. Here is a breakdown of what that actually means and what it revealed.

### What is a Heatmap?
A heatmap is just a color-coded table. Instead of reading every number, you use color to quickly spot patterns:
*   **Dark Red** means a strong positive relationship.
*   **Dark Blue** means a strong negative relationship.
*   **White/Light** means basically no relationship.

### What is "Correlation"?
Correlation measures how much two things move together, on a scale from -1.0 to +1.0:
*   **+1.0**: When one goes up, the other always goes up too.
*   **-1.0**: When one goes up, the other always goes down.
*   **0**: They have nothing to do with each other.

### What My Heatmap is Saying (The Honest Truth)
The big story here is about predicting **TRAFFIC_VOLUME** (bottom row). When I look at that row, almost everything is near 0. This means:
*   **Weak Predictors**: Most of my features are actually not very good at predicting traffic volume. The strongest ones are only -0.11 (Road Age) and 0.08 (Road Surface Type)—which are mathematically very small.

### Other Notable Findings:
Beyond traffic, the heatmap revealed some very logical (and some redundant) patterns:
*   🔴 **Redundancy (0.88)**: `Road_Surface_Type` and `Road_Structure_Type` are almost identical. In the math of ML, having both is redundant; one is enough to tell the story.
*   🔴 **Maintenance vs. Age (0.26)**: Older roads tend to belong to higher maintenance groups. This makes intuitive sense—the older the road, the more attention it needs from city crews.
*   🔵 **Speed vs. Age (-0.25)**: Older roads tend to have lower speed limits. This is logical because our oldest streets are often narrow residential ones, while newer roads are built as wide, fast-moving corridors.

### 💡 The Honest Bottom Line
This was a huge lesson for me: **My model is trying to predict traffic using features that don't strongly correlate with it.** 

This doesn't mean the code is wrong; it means the dataset is missing key "clues" (like proximity to downtown or time of day). A model trained on weak features will produce unreliable predictions. This is a vital lesson in Machine Learning—it’s not just about the code, it’s about the quality of the data!

---

## 🏁 Final Summary
This project shows that I can take messy city data and turn it into a smart tool. It’s not about replacing humans—it’s about giving humans a better "map" to make smarter decisions for Saskatoon. My goal as a data analyst is not just to run models, but to understand the reality behind the numbers.

**#MachineLearningJourney #TimiNdubuisi #LinearRegression #DataScience #SmartCity #AI**