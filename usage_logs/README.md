# Usage Statistics Dashboard

This feature tracks and visualizes your API usage when interacting with OpenRouter models.

## Overview

The Usage Statistics Dashboard provides detailed analytics about your OpenRouter API usage, helping you understand:

- Total token consumption
- Number of requests and conversations
- Model-specific usage metrics
- Task type analysis
- Estimated costs

## Features

### Monthly Usage Tracking

- View usage data for the current month or select previous months
- Summary statistics showing total tokens, requests, and conversations
- Estimated cost calculations based on token usage

### Model Analysis

- Compare usage across different models
- See which models you use most frequently
- Analyze token consumption by model

### Task Type Analysis

- Track usage across different task types (general, coding, reasoning, vision)
- Compare token consumption by task type
- Identify which task types consume most resources

### Cost Estimation

- Calculate estimated API costs based on token usage
- Adjust pricing parameters for different model tiers
- View cost breakdown by model

### Data Export

- Export usage data as CSV or JSON files
- Save and share usage reports

## How It Works

1. The `UsageTracker` class logs information about:
   - API requests (tokens, models used, task types)
   - New conversations
   - Data is stored in monthly JSON files in the `usage_logs` directory

2. The dashboard reads these logs and presents:
   - Interactive visualizations
   - Filterable data tables
   - Cost calculations

## Accessing the Dashboard

You can access the dashboard in two ways:

1. Click on the "Stats" tab in the sidebar and then click "View Detailed Dashboard"
2. Navigate directly to the Usage Statistics page from the navigation menu

## Future Enhancements

- Time-based analytics (daily, weekly trends)
- User-specific tracking for multi-user deployments
- Budget alerts and notifications
- Integration with OpenRouter's official billing API
