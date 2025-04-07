![alt text](https://www.basketball-reference.com/req/202106291/images/headshots/adamsst01.jpg)

# Overview

**This repo features NBA only and is the public sample of the full private funaki system, which includes NBA/NCAAMB/NCAAWB/WNBA/NBL/Euroleague/Eurocup**

Funaki is a basketball methodology based on analysis of play by play data to create Bayesian player offensive/defensive ratings and team pace tendencies based on analysis of shot clock usage.  

The player ratings consider every posession a player was on the court, as well as the offensive ratings of their 4 teammates and the defensive ratings of their 5 opponents, and the outcome of the possession.  These ratings are updated through a Kalman filtering approach, built with tensorflow probability, to keep player ratings up to date while still robust to swings due to short term noise.  

The use of shot clock to determine pace tendencies allows to specifically measure team behavior without skewing the signal with opposing team pace, as is the case with most traditional pace metrics.  Using these shot clock pace metrics as well as team efficiency predictions aggregated from each team's player's individual offensive/defensive skill ratings, with projected play-time considerations, yields effective prediction models for spread, moneyline, and totals markets.  

Funaki is built with the first principles of basketball gameplay in mind, and thus is generalizable to any basketball league where play by play and court tracking information is available.  

## Current Player Ratings

The interpretation of these ratings is the added efficiency from that players presence on the court over 100 possessions versus a league average replacement level player.  For example, Jokic as the highest rated player by net rating, has an offensive rating of 7.26 and a defensive rating of 3.6.  This means given 4 additional replacement level teammates and 5 opposing replacement level players, we expect the Jokic team to be 7.26 points per 100 possessions better than a league average efficiency and their opponent to be 3.6 points per 100 possesions worse than league average efficiency.  

![image](https://github.com/user-attachments/assets/ca9cf950-0611-4400-b254-e99f4e19d955)

## Time Series of Player Ratings from 2015 to Present
![image](https://github.com/user-attachments/assets/b703bfc0-2c82-4992-a553-422f36821d7e)

## Top player ratings within the last 10 years
Only allowing individual players to show up once 
![image](https://github.com/user-attachments/assets/2a506b28-2bbf-4d1f-b1f2-03afff36b24a)

# Results Archive (Updated 04/07/2025)

## NBA

### 2024-2025

Totals: 38.75 (325/582) (55.8%)  
Spread: -14.65 (161/322) (50.0%)  
Moneyline: 2.0 (187/580)

### 2023-2024
  
Totals: 23.1 (249/462) (53.9%)  
Spread: 2.63 (48/89) (53.9%)  
Moneyline: 30.76 (143/423)

## NCAAMB

### 2024-2025
  
Totals: 16.59 (531/1005) (52.8%)

### 2023-2024
  
Totals: 44.57 (663/1235) (53.7%)  

## NCAAWB 

### 2024-2025

Totals: 36.99 (277/481) (57.6%)  
Spread: 19.38 (274/493) (55.6%)

### 2023-2024
  
Totals: -5.63 (97/189) (51.3%)  
Spread: 6.14 (54/97) (55.7%)  

## WNBA

### 2024

Totals: -7.13 (57/116) (49.1%)  
Moneyline: -14.0 (27/104)  
Spread: 4.36 (18/30) (60.0%)  

Points Props: 12.9 (57.4% accuracy on 155 bets)  
Assists Props: 3.17 (58.6% accuracy on 29 bets)  
Rebounds Props: -9.6 (47.8% accuracy on 67 bets) 

## EuroLeague

### 2024-2025 

Totals: Totals: 27.09 (120/216) (55.6%)   
Spread: 5.18 (9/12) (75%)   
Moneyline: 6.81 (12/31)

## EuroCup

### 2024-2025 

Totals: -1.32 (36/70) (51.4%)  
Spread: 0.36 (7/13) (53.8%)    
Moneyline: -6.9 (5/24)

## NBL

### 2024-2025 

Totals: -4.24 (33/63) (52.4%)   
Spread: 3.45 (6/8) (75%)   
Moneyline: 6.51 (9/23)
