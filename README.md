# Retail Recommendation System

A personalized recommendation system for an e-commerce platform, which suggests products based on user preferences, browsing history, and contextual factors. The system uses a hybrid approach of collaborative filtering, content-based filtering, and association rule mining to generate recommendations.

This project is developed as part of the third-year first-semester project for the Information Retrieval and Web Analytics module.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributers](#contributers)
- [License](#license)

  ## Features

  **Content-Based Filtering:**
  Recommends items based on their features.
  Similar products are suggested based on user interactions.
  
  **Item-based Collaborative Filtering:**
  Suggests items based on user purchase patterns.
  Items frequently bought together are recommended.
  
  **Association Rule-Based Filtering**
  Find patterns of frequently co-purchased items.
  Recommends items based on past transactions.
  
  **Hybrid Approach:**
  Combines all methods for better accuracy.
  Provides personalized and frequent co-purchase recommendations.
  
  **Monthly Support Filtering:**
  Recommends items with high support from previous months.
  For example, items with significant support in September 2023 are recommended to users in September 2024.
  It helps highlight seasonal or trending products based on historical popularity.

  ## Technologies Used

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: My SQL

  ## Installation

**Clone the repository**
  ```bash
  git clone https://github.com/ChaLinP/recommendation-system.git
  cd recommendation-system



  
