# Learning Platform Recommendation System

This project is a content-based recommendation system for a learning platform using machine learning.

## Setup

1. Install required packages

   ```
   pip install -r requirements.txt
   ```

2. Run the API

   ```
   python recommendation_api.py
   ```

## Usage

GET `/recommendations/<user_id>`

#### Parameters

- `user_id`: The user ID for which to generate recommendations.

#### Returns

List of recommended content for the user with the given user ID.

Example:

GET `/recommendations/1`

#### Response

[
  {
    "content": "Machine Learning Fundamentals",
    "similarity": 0.857
  },
  {
    "content": "Introduction to Data Science",
    "similarity": 0.821
  },
  ...
]

## Model

The model is trained using a content-based algorithm and saved as a serialized pickle object.

## Testing

The API contains a test suite for unit testing.

pytest

## Dependencies

- pandas
- scikit-learn
- Flask
- pickle
- pytest
