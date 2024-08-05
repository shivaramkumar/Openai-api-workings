import os
from typing import Type

from langchain.tools import BaseTool
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor
from pydantic import BaseModel, Field
from pyowm.owm import OWM
from pyowm.utils.config import get_default_config

from ai_apps.langchain_ollama.core import get_llm

api_key = os.environ.get("OPENAI_API_KEY")


def get_weather(city):
    config_dict = get_default_config()
    config_dict["language"] = "en"  # your language here, eg. French
    owm = OWM("16e73d2a4456086bcfdc469c323351d5", config_dict)
    mgr = owm.weather_manager()
    observation = mgr.weather_at_place("Neuss, Germany")
    temperature = str(observation.weather.temperature("celsius")["temp"]) + "°C"
    humidity = str(observation.weather.humidity) + "%"
    wind = str(observation.weather.wind()) + " m/s"
    status = observation.weather.detailed_status
    return {"temp": temperature, "humid": humidity, "wind": wind, "status": status}


class GetWeatherInput(BaseModel):
    """Inputs for get_weather"""

    city: str = Field(description="City name with country separated by comma")


class GetWeatherTool(BaseTool):
    name = "get_weather_details_of_a_city"
    description = """
    Useful to get weather details of a city.
    Mandatory input format is 'city, country'.
    """
    args_schema: Type[BaseModel] = GetWeatherInput

    def _run(self, city: str):
        weather = get_weather(city)
        return weather

    def _arun(self, city: str):
        raise NotImplementedError("this tool doesn't support async")


llm = get_llm()
tools = [GetWeatherTool()]
model = get_llm()
planner = get_llm()
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
agent.run("Do I need a jacket to go out now in Neuss ?")
