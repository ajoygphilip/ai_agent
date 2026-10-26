from .models import Tool


def get_weather(location: str, unit: str = "celsius") -> dict:
    """Return mock weather data for a location."""
    return {
        "location": location,
        "temperature": 28,
        "unit": unit,
        "condition": "partly cloudy",
    }


weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a given location.",
    input_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location to check.",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit.",
            },
        },
        "required": ["location"],
        "additionalProperties": False,
    },
    handler=get_weather,
    metadata={"category": "weather"},
)
