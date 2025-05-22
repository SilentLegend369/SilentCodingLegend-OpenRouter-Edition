# SCL-OpenRouter AI Chat Interface

A sophisticated chat interface that leverages various AI models through OpenRouter's API to provide a versatile conversational experience.

![SCL-OpenRouter AI Chat Interface](https://raw.githubusercontent.com/username/scl-openrouter/main/assets/screenshot.png)

## Features

- **Multi-Model Support**: Access AI models from different providers (Anthropic, OpenAI, Google, etc.) through OpenRouter
- **Task-Specific Model Selection**: Automatically selects the best models for different tasks (general, coding, reasoning, vision)
- **Dark Mode UI**: Sleek, modern dark interface built with Streamlit
- **Conversation Management**: Save, load, and switch between multiple conversations
- **File Analysis**: Upload and analyze various file formats
- **Vision Capabilities**: Process and analyze images with vision-enabled models
- **Usage Statistics**: Track token usage and estimated costs
- **Multiple Specialized Interfaces**: Dedicated pages for different tasks like coding and image analysis

## Pages

- **Chat Interface**: Main interface for text conversations with AI
- **Vision Analysis**: Upload and analyze images
- **Coding**: Specialized interface for programming assistance
- **Deep Thinking/Reasoning**: Interface optimized for complex reasoning tasks
- **Chat History**: View and manage saved conversations
- **Usage Stats**: Monitor your usage statistics and costs

## Setup

### Prerequisites

- Python 3.8+
- OpenRouter API Key (get one from [OpenRouter](https://openrouter.ai))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/scl-openrouter.git
   cd scl-openrouter
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

### Running the Application

```bash
streamlit run Chat_Interface.py
```

## Project Structure

```
scl-openrouter/
├── Chat_Interface.py          # Main application entry point
├── requirements.txt           # Project dependencies
├── .env                       # Environment variables (not in repo)
├── dark_theme.css             # CSS for dark theme
├── Chat_History/              # Saved chat history files
├── pages/                     # Streamlit pages
│   ├── Chat_History.py        # History viewer interface
│   ├── Coding.py              # Coding assistance interface
│   ├── Deep_Thinking-Reasoning.py  # Complex reasoning interface
│   ├── Vision.py              # Image analysis interface
│   └── Usage_Stats.py         # Usage statistics dashboard
└── src/                       # Source code modules
    ├── agents/                # AI agent implementations
    │   └── openrouter_agent.py  # OpenRouter API client
    ├── model_config.py        # Model configuration
    ├── utils.py               # Utility functions
    └── usage_tracker.py       # Usage tracking functionality
```

## Configuration

You can configure the default models and providers in `src/model_config.py`.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [OpenRouter](https://openrouter.ai) for providing the API
- [Streamlit](https://streamlit.io) for the web framework
